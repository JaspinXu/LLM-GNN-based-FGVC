import os
from itertools import chain
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor
import torchvision.models as models
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from networks.relation_net import DisjointRelationNet, Mapper,Attention
from networks.gcn import GCN
from utils import constants
from utils.enhanced import all_en
import clip
from utils.cate import CATE
# import visdom



def graphgen(node_embeddings):
    device = node_embeddings.device
    _, num_nodes, _ = node_embeddings.shape
    sims = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
    sims = sims * torch.ones(num_nodes, num_nodes).fill_diagonal_(0).to(device)  # disregard self-similarities
    directed: Tensor = sims > (sims.sum(dim=2) / num_nodes - 1).unsqueeze(dim=2)  # average only over non-zero elms
    undirected = directed + directed.transpose(1, 2)
    assert torch.all(undirected == undirected.transpose(1, 2)).item()  # validate symmetrization
    edges = undirected.nonzero()

    edge_lists = []
    offset = 0
    graphs = []
    for i, sample in enumerate(undirected):
        num_edges = undirected[i].sum()
        edges_i = edges[offset: offset + num_edges]
        # Edge list in COO format
        edges_i = edges_i[:, 1:].T
        edge_lists.append(edges_i)
        offset = offset + num_edges
        graphs.append(Data(x = node_embeddings[i], edge_index=edges_i))

    return graphs


def cdist(set1, set2):
    ''' Pairwise Distance between two matrices
    Input:  x is a Nxd matrix
            y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # dist = set1.unsqueeze(1) - set2.unsqueeze(0)
    dist = set1.unsqueeze(1) - set2.unsqueeze(2)
    return dist.abs()


def hausdorff_distance(g1, g2):
    g1_local, g2_local = g1[:, 1:], g2[:, 1:]
    g1_global, g2_global = g1[:, 0], g2[:, 0]

    dist_matrix = cdist(g1_local, g2_local).pow(2.).sum(-1)

    d1 = 0.5 + g1_global.abs()
    d2 = 0.5 + g2_global.abs()

    pw_cost_1, _ = dist_matrix.min(1)
    pw_cost_2, _ = dist_matrix.min(2)

    cost_g1, _ = torch.min(torch.cat([d1, pw_cost_2], dim=-1), dim=-1)
    cost_g2, _ = torch.min(torch.cat([d2, pw_cost_1], dim=-1), dim=-1)

    return (cost_g1.sum() + cost_g2.sum()) / (len(g1) * 2)


class MultiViewHausdorff(nn.Module):
    def __init__(self, backbone, num_classes, logdir, train_backbone, local_weight, recovery_epoch):
        super(MultiViewHausdorff, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR
        self.local_weight = local_weight
        self.recovery_epoch = recovery_epoch
        self.resmodel = models.resnet50(pretrained=True)
        num_classes = 200
        self.resmodel.fc = torch.nn.Linear(self.resmodel.fc.in_features, num_classes).to('cuda')
        self.backbone = backbone
        self.aggregator = GCN(num_in_features=self.feature_dim, num_out_features=self.feature_dim)
        self.relation_net = DisjointRelationNet(feature_dim=self.feature_dim , out_dim=self.feature_dim, num_classes=num_classes)
        self.global_net = Mapper(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes)
        self.agg_net = Mapper(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes)
        self.text_net = Mapper(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes)
        self.cate = CATE()
        # self.vis = visdom.Visdom()
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            trainable_params = chain(self.cate.parameters(),self.relation_net.parameters(), self.global_net.parameters(), self.agg_net.parameters(),self.text_net.parameters())
        else:
            trainable_params = chain(backbone.parameters(), self.cate.parameters(),self.relation_net.parameters(), self.global_net.parameters(),
                                     self.agg_net.parameters(), self.aggregator.parameters(),self.text_net.parameters())
        trainable_params1 = chain(self.cate.parameters(),self.relation_net.parameters())
        #self.optimizer = torch.optim.Adam(self.resmodel.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
        self.optimizer = torch.optim.SGD(trainable_params1, lr=self.lr, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)
        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 9, 512))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.writer = SummaryWriter(logdir)
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    def train_one_epoch(self, trainloader, epoch, save_path):
        print('Training %d epoch' % epoch)
        self.train()
        device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
        recovery_weight =  self.local_weight
        epoch_state = {'loss': 0, 'correct': 0}
        for i, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)
            self.optimizer.zero_grad()
            
            
            #global_logits, local_logits, sem_logits,text_logits, semrel_graphs,info_loss,kl = self.compute_reprs(im)
            sem_logits,info_loss,kl = self.compute_reprs(im)
           

            loss = self.criterion(sem_logits , labels)+info_loss+kl
            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict( sem_logits,labels, epoch_state)
        self.scheduler.step()
        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            print('Testing %d epoch' % epoch)
            self.eval()
            device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
            epoch_state = {'loss': 0, 'correct': 0}
            for i, data in enumerate(tqdm(testloader)):
                im, labels = data
                im, labels = im.to(device), labels.to(device)
                relation_logits,info_loss,kl = self.compute_reprs(im)
                #epoch_state = self.predict(global_repr, local_repr, relation_logits, text_logits,labels, epoch_state)
                epoch_state = self.predict( relation_logits,labels, epoch_state)
                loss = self.criterion(relation_logits, labels)
                epoch_state['loss'] += loss.item()

            self.post_epoch('Test', epoch, epoch_state, len(testloader.dataset), None)

    def compute_reprs(self, im):
        B =im.shape[0]
        enhanced_patch = []
        path = "/root/autodl-tmp/fine/my_method/word_embeddings_512.pt"
        for j in ["random","4","9","16","25"]:

            global_embed, local_embeds, global_view, local_views = self.backbone.forward_with_views(im,crop_mode=j)
            global_embed,local_embeds = global_embed.float(),local_embeds.float()
            one_patch,p_c,n_c = all_en(global_embed,local_embeds,path)
            enhanced_patch.append(one_patch)
        # Step 1: 将所有张量堆叠成一个新的张量，形状变为 [5, 8, 512]
        stacked_patches = torch.stack(enhanced_patch)
        mean_patch = stacked_patches.mean(dim=0)
        stacked_patches = stacked_patches.transpose(0, 1)  # 交换第0维和第1维，形状变为 [8, 5, 512]
        stacked_patches = stacked_patches.squeeze(2)  # 去掉第2维（即维度为1的维度）

        enhance_patch,kloss1,info_loss1 = self.cate.ib(mean_patch,p_c,n_c)
        text_feature = p_c.mean(dim=1)
        text_feature =text_feature.float()
        full_embed = self.model.encode_image(im)
        full_embed =full_embed.float()
        global_embed = global_embed.unsqueeze(1)  # 变为 [b, 1, 512]
        enhance_patch = enhance_patch.unsqueeze(1)  # 变为 [b, 1, 512]
        text_feature = text_feature.unsqueeze(1)  # 变为 [b, 1, 512]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feature = torch.cat([global_embed, enhance_patch,text_feature,stacked_patches,cls_tokens], dim=1)
        feature = feature + self.position_embeddings
        sem_logits = self.relation_net(feature) +self.global_net(full_embed)
        return sem_logits,info_loss1,kloss1
    def compute_views(self, im, num_views=2):
        path = "/root/autodl-tmp/fine/my_method/word_embeddings_512.pt"
        views = []
        for _ in range(num_views):
            global_embed, local_embeds, global_view, local_views = self.backbone.forward_with_views(im)
            _, full_embed, _ = self.backbone.extractor(im)
            global_embed,local_embeds = global_embed.float(),local_embeds.float()
            _,local_embeds = all_en(global_embed,local_embeds,path)
            graphs = graphgen(local_embeds)
            graph_loader = DataLoader(graphs, batch_size=len(im))
            semrel_graphs = self.aggregator(next(iter(graph_loader)))
            view = semrel_graphs.reshape(len(im), local_embeds.shape[1], -1)  # batch_size x num_local
            views.append(view)

        return views

    @torch.no_grad()
    def predict(self, relation_logits,labels, epoch_state):
        pred = (relation_logits).max(1, keepdim=True)[1]
        epoch_state['correct'] += pred.eq(labels.view_as(pred)).sum().item()

        return epoch_state
    

    @torch.no_grad()
    def post_epoch(self, phase, epoch, epoch_state, num_samples, save_path):
        accuracy = epoch_state['correct'] / num_samples
        loss = epoch_state['loss']

        print(f'{phase} Loss: {loss}')
        print(f'{phase} Accuracy: {accuracy * 100}%')
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)
        # if phase == 'Train':
        #     self.vis.line(X=[epoch], Y=[accuracy], win='Train Accuracy', update='append', name='Train Accuracy', opts=dict(title='Train Accuracy'))
        # else:
        #     self.vis.line(X=[epoch], Y=[accuracy], win='Test Accuracy', update='append', name='Test Accuracy', opts=dict(title='Test Accuracy'))
        if (phase == 'Train') and ((epoch % constants.SAVE_EVERY == 0) or (epoch == constants.END_EPOCH)):
            self.scheduler.step()
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'learning_rate': self.lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

    def post_job(self):
        """Post-job actions"""
        self.writer.flush()
        self.writer.close()

