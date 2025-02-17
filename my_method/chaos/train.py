import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# import clip
import torch
from model import graphgen,GCN,Mapper,DisjointRelationNet,DisjointRelationNet_old
from chaos.cub_dataset import CUBImageDataset
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import view_ex as ve
from longclip.model import longclip

class MultiViewHausdorff(nn.Module):
    def __init__(self, input_dim, feature_dim,device, clip_model,lr,extractor):
        super(MultiViewHausdorff, self).__init__()
        self.device = device
        self.clip = clip_model.to(self.device)
        self.lr = lr
        self.extractor = extractor.to(self.device)
        self.aggregator = GCN(num_in_features=input_dim, num_out_features=feature_dim).to(self.device)
        self.ad_net=Mapper(feature_dim=feature_dim,out_dim=feature_dim,num_classes=200).to(self.device)
        self.global_net = Mapper(feature_dim=feature_dim, out_dim=feature_dim, num_classes=200).to(self.device)
        self.text_net = Mapper(feature_dim=feature_dim, out_dim=feature_dim, num_classes=200).to(self.device)
        self.relation_net = DisjointRelationNet_old(feature_dim=feature_dim * 3, out_dim=feature_dim, num_classes=200).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        trainable_params = chain(self.ad_net.parameters(), self.aggregator.parameters(),self.text_net.parameters(),self.global_net.parameters(),self.relation_net.parameters())
        self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[20, 50, 100], gamma=0.1)
        self.recovery_weight = 0.0001
        self.num_loacal  =32
    def train_one_epoch(self, trainloader, epoch,save_path):
        print('Training %d epoch' % epoch)
        save_interval=10
        self.train()
        for i, data in enumerate(tqdm(trainloader)):
            (im,im_text), labels = data
            #print(labels)
            batch_size = im.size(0)
            im, labels = im.to(self.device), labels.to(self.device)
            global_im =im.to(self.device)
            '''im = im.reshape(batch_size,3,8,224,8,224)
            im = im.permute(0, 2, 4, 1, 3, 5)
            im = im.reshape(batch_size, 64, 3, 224, 224)
            im_input = im.view(-1,3,224,224)'''
            #text = "bird"
            bird_list = list(im_text)
            text_descriptions = [desc[:248] for desc in bird_list]
            text_input = longclip.tokenize(text_descriptions).to(self.device)

            with torch.no_grad(): 
                global_im = ve.extract_global(global_im,self.extractor)
                #print(global_im.shape)
                lacal_im = ve.extract_local(global_im,num_local=self.num_loacal,divide=3,crop_mode="random")
                lacal_im = lacal_im.to(self.device)
                #print(lacal_im.shape)
                image_features = self.clip.encode_image(lacal_im)
                #print(image_features.shape)
                #text_features = self.clip.encode_text(text_input)
                global_im = global_im.to(self.device)
                global_features = self.clip.encode_image(global_im)
                #print(im_text)
                global_text_feature = self.clip.encode_text(text_input)
            output = image_features.view(batch_size,self.num_loacal,512)
            similarity = []
            for i in range(batch_size):
                #print(output[i].shape)
                #print(global_features.shape) 
                image_part_similarities = self.cosine_similarity(output[i],global_features[i]) # 每个子图与全局图的相似度
                similarity.append(image_part_similarities)
            similarity = torch.stack(similarity)
            max_similarity_indices = torch.argmax(similarity, dim=-1)
            selected_features = output[torch.arange(batch_size), max_similarity_indices]
            output_filter = self.select_top_features(output,selected_features,top_k=20)
            #print(output_filter.shape)
            local_logits,global_logits,text_logits,all_logits = self.compute_reprs(output_filter,global_features,global_text_feature)
            #print(logits)
            loss = self.criterion(all_logits + (self.recovery_weight * local_logits) + global_logits + text_logits, labels)
            #print(loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # 每save_interval轮保存一次权重
        # if (epoch + 1) % save_interval == 0:
        #     torch.save(self.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")
        #     print(f"Model weights saved for epoch {epoch + 1} at {save_path}/model_epoch_{epoch + 1}.pth")
        
    @torch.no_grad()
    def test(self, testloader, epoch):
        loss_list = []
        correct = 0  # 初始化正确预测的数量
        total = 0    # 初始化总的样本数
        print('Testing %d epoch' % epoch)
        self.eval()  # hacky, but keeps the arg list clean
        for i, data in enumerate(tqdm(testloader)):
            (im,im_text), labels = data
            #print(labels)
            batch_size = im.size(0)
            im, labels = im.to(self.device), labels.to(self.device)
            global_im =im.to(self.device)
            '''im = im.reshape(batch_size,3,8,224,8,224)
            im = im.permute(0, 2, 4, 1, 3, 5)
            im = im.reshape(batch_size, 64, 3, 224, 224)
            im_input = im.view(-1,3,224,224)'''
            #text = "bird"
            #text_input = clip.tokenize([text]).to(self.device)
            bird_list = list(im_text)
            text_descriptions = [desc[:248] for desc in bird_list]
            text_input = longclip.tokenize(text_descriptions).to(self.device)
            with torch.no_grad(): 
                global_im = ve.extract_global(global_im,self.extractor)
                lacal_im = ve.extract_local(global_im,num_local=self.num_loacal,divide=3,crop_mode="random")
                lacal_im = lacal_im.to(self.device)
                image_features = self.clip.encode_image(lacal_im)
                #text_features = self.clip.encode_text(text_input)
                global_im = global_im.to(self.device)
                global_features = self.clip.encode_image(global_im)
                global_text_feature = self.clip.encode_text(text_input)
            output = image_features.view(batch_size,self.num_loacal,512)
            similarity = []
            for i in range(batch_size):
                image_part_similarities = self.cosine_similarity(output[i],global_features[i]) # 每个部分与词汇的相似度
                similarity.append(image_part_similarities)
            similarity = torch.stack(similarity)
            max_similarity_indices = torch.argmax(similarity, dim=-1)
            selected_features = output[torch.arange(batch_size), max_similarity_indices]
            output_filter = self.select_top_features(output,selected_features,top_k=20)
            #print(output_filter.shape)
            local_logits,global_logits,text_logits,all_logits = self.compute_reprs(output_filter,global_features,global_text_feature)
            #print(logits)
            loss = self.criterion(all_logits + (self.recovery_weight * local_logits) + global_logits + text_logits, labels)
            loss_list.append(loss)
             # 计算正确率
            predicted = self.predict(global_logits,local_logits,text_logits,all_logits)  # 获取每个样本的预测标签
            correct += (predicted == labels).sum().item()  # 统计预测正确的数量
            total += labels.size(0)  # 总样本数
            accuracy = 100 * correct / total
        mean_loss = sum(loss_list) / len(loss_list)
        print(f"Test Loss: {mean_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def compute_reprs(self,im,glbal_in,text_feature):
        graphs = graphgen(im)
        graph_loader = DataLoader(graphs, batch_size=len(im))
        #print(next(iter(graph_loader))._num_graphs)
        #print(graphs)
        semrel_graphs = self.aggregator(next(iter(graph_loader)))
        semrel_graphs = semrel_graphs.reshape(len(im), im.shape[1], -1)  # batch_size x num_local
        agg_embed = semrel_graphs.mean(dim=1)
        local_logits = self.ad_net(agg_embed)
        glbal_in = glbal_in.float()
        text_feature = text_feature.float()
        global_logits = self.global_net(glbal_in)
        text_logits = self.text_net(text_feature)
        #all_embed = agg_embed+glbal_in+text_feature
        sem_logits = self.relation_net(glbal_in, agg_embed,text_feature)

        #logits = self.ad_net(all_embed)
        return local_logits,global_logits,text_logits,sem_logits
    
    def cosine_similarity(self,a, b):
        return F.cosine_similarity(a, b, dim=-1)
    
    def calculate_information_density(self,embedding_tensor, selected_tensor):
        """
        计算一个特征与被选择特征之间的信息密度。
        """
        euclidean_distance = torch.norm(embedding_tensor - selected_tensor, dim=-1)  # 计算欧几里得距离
        information_density = 1 / (1 + euclidean_distance)  # 计算信息密度
        return information_density
    

    def select_top_features(self,embedding_tensor, selected_tensor, top_k=32):
        """
        选择每个批次中信息密度得分最高的特征。
        
        Args:
            embedding_tensor (torch.Tensor): 形状为 (batch_size, 64, 512) 的嵌入特征张量。
            selected_tensor (torch.Tensor): 形状为 (batch_size, 512) 的被选择特征张量。
            top_k (int): 每个批次选择得分最高的特征数量。
            
        Returns:
            torch.Tensor: 形状为 (batch_size, top_k, 512) 的选择特征张量。
        """
        batch_size, num_features, feature_dim = embedding_tensor.shape
        
        # 计算每个特征与被选择特征的信息密度，得到一个形状为 (batch_size, 64) 的信息密度张量
        information_density = torch.zeros(batch_size, num_features)
        for i in range(batch_size):
            information_density[i] = self.calculate_information_density(embedding_tensor[i], selected_tensor[i])
        
        # 对每个批次，按照信息密度降序排序，选择前 top_k 个特征
        _, topk_indices = torch.topk(information_density, top_k, dim=-1, largest=True)
        topk_indices = topk_indices.to(self.device)
        # 使用 topk_indices 来选择对应的信息密度得分最高的特征
        selected_features = torch.gather(embedding_tensor, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feature_dim))
        
        return selected_features
    torch.no_grad()
    def predict(self, global_logits, local_logits, text_logits, relation_logits):
        pred = (global_logits + (self.recovery_weight * local_logits) + relation_logits+text_logits).max(1, keepdim=True)[1]
        return pred