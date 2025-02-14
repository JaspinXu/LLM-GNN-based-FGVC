# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from torch_geometric.loader import DataLoader
import torch
from model import graphgen,GCN,Mapper,DisjointRelationNet,GAT_
from chaos.cub_dataset import CUBImageDataset
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import view_ex as ve
from longclip.model import longclip
from enhanced import gen_patch
from cate import CATE
import time

class MultiViewFGVC(nn.Module):
    def __init__(self, input_dim, feature_dim, num_classes, device, clip_model,lr,extractor):
        super(MultiViewFGVC, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.clip = clip_model.to(self.device)
        self.lr = lr
        self.extractor = extractor.to(self.device)
        self.aggregator = GCN(num_in_features=input_dim, num_out_features=feature_dim).to(self.device)
        self.ad_net=Mapper(feature_dim=feature_dim,out_dim=feature_dim,num_classes=self.num_classes).to(self.device)
        self.global_net = Mapper(feature_dim=feature_dim, out_dim=feature_dim, num_classes=self.num_classes).to(self.device)
        self.text_net = Mapper(feature_dim=feature_dim, out_dim=feature_dim, num_classes=self.num_classes).to(self.device)
        self.concept_net = Mapper(feature_dim=feature_dim, out_dim=feature_dim, num_classes=self.num_classes).to(self.device)
        self.relation_net = DisjointRelationNet(feature_dim=feature_dim * 4, out_dim=feature_dim, num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.cate = CATE()
        trainable_params = chain(self.cate.parameters(), self.ad_net.parameters(), self.aggregator.parameters(),self.text_net.parameters(),self.concept_net.parameters(),self.global_net.parameters(),self.relation_net.parameters())
        # self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[20, 50, 100], gamma=0.1)
        self.optimizer = Adam(trainable_params, lr=1e-3)  
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',     
            factor=0.1,  
            patience=3,     
            verbose=True,   
            min_lr=1e-6  
        )
        self.recovery_weight = 1
        # self.num_loacal  = 5
        # self.select_concepts  = 10
        # self.select_locals = 3
        self.num_loacal  = 36
        self.select_concepts  = 40
        self.select_locals = 20
        self.concept_path =  "/root/autodl-tmp/fine/my_method/data/cub2011/des_and_concept/cub_concepts__512_longclip.pt"
        # self.concept_path =  "/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft_concepts__512_longclip.pt"
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
                image_features = self.clip.encode_image(lacal_im) # 16*32  ###timecost
                #print(image_features.shape)
                #text_features = self.clip.encode_text(text_input)
                global_im = global_im.to(self.device)
                global_features = self.clip.encode_image(global_im)
                #print(im_text)
                global_text_feature = self.clip.encode_text(text_input)
                p_c,n_c,_,weights = self.select_coarse_concepts(global_features, self.concept_path, k=self.select_concepts) ##得到最相似的概念
            output = image_features.view(batch_size,self.num_loacal,512)
            similarity = []
            for i in range(batch_size):
                #print(output[i].shape)
                #print(global_features.shape) 
                image_part_similarities = self.cosine_similarity(output[i],global_features[i]) # 每个子图与全局图的相似度
                similarity.append(image_part_similarities)
            similarity = torch.stack(similarity)
            max_similarity_indices = torch.argmax(similarity, dim=-1)
            selected_features = output[torch.arange(batch_size), max_similarity_indices] #output 16*32*512,selected_features 16*512
            output_filter = self.select_top_features(output,selected_features,top_k=self.select_locals) #
            concepts_filter = self.find_most_similar_concepts(p_c, output_filter)
            
            self.optimizer.zero_grad()
            local_logits,global_logits,text_logits,concept_logits,all_logits = self.compute_reprs(output_filter,global_features,global_text_feature,concepts_filter)
            cc_loss = self.concept_contrastive_loss(global_im,p_c,n_c,weights)
            # del im, im_text, labels, image_features, global_features, global_text_feature
            # torch.cuda.empty_cache()

            loss = self.criterion(all_logits + (self.recovery_weight * local_logits) + global_logits + text_logits + (self.recovery_weight * concept_logits), labels)+cc_loss
            #print(loss
            loss.backward()
            self.optimizer.step()
        # 每save_interval轮保存一次权重
        if (epoch + 1) % save_interval == 0:
            torch.save(self.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")
            print(f"Model weights saved for epoch {epoch + 1} at {save_path}/model_epoch_{epoch + 1}.pth")
        
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
                p_c,n_c,_,weights = self.select_coarse_concepts(global_features, self.concept_path, k=self.select_concepts)
            output = image_features.view(batch_size,self.num_loacal,512)
            similarity = []
            for i in range(batch_size):
                image_part_similarities = self.cosine_similarity(output[i],global_features[i]) # 每个部分与词汇的相似度
                similarity.append(image_part_similarities)
            similarity = torch.stack(similarity)
            max_similarity_indices = torch.argmax(similarity, dim=-1)
            selected_features = output[torch.arange(batch_size), max_similarity_indices]
            output_filter = self.select_top_features(output,selected_features,top_k=self.select_locals)
            concepts_filter = self.find_most_similar_concepts(p_c, output_filter)
            local_logits,global_logits,text_logits,concept_logits,all_logits = self.compute_reprs(output_filter,global_features,global_text_feature,concepts_filter)
            #print(logits)
            cc_loss = self.concept_contrastive_loss(global_im,p_c,n_c,weights)
            loss = self.criterion(all_logits + (self.recovery_weight * local_logits) + global_logits + text_logits + (self.recovery_weight * concept_logits), labels) + cc_loss
            loss_list.append(loss)
             # 计算正确率
            predicted = self.predict(global_logits,local_logits,text_logits,all_logits,concept_logits)  # 获取每个样本的预测标签
            correct += (predicted.reshape(-1) == labels).sum().item()  # 统计预测正确的数量
            total += labels.size(0)  # 总样本数
            accuracy = 100 * correct / total
        mean_loss = sum(loss_list) / len(loss_list)
        self.scheduler.step(mean_loss) 
        print(f"Test Loss: {mean_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def compute_reprs(self,im,glbal_in,text_feature,concepts_filter):
        agg_embed = self.gen_agg_graph(im)
        local_logits = self.ad_net(agg_embed)
        concept_agg_embed = self.gen_agg_graph(concepts_filter)
        concept_logits = self.concept_net(concept_agg_embed)
        glbal_in = glbal_in.float()
        text_feature = text_feature.float()
        global_logits = self.global_net(glbal_in)
        text_logits = self.text_net(text_feature)
        sem_logits = self.relation_net(glbal_in, agg_embed,text_feature, concept_agg_embed)

        return local_logits,global_logits,text_logits,concept_logits,sem_logits
    
    def gen_agg_graph(self,filter_set):
        graphs = graphgen(filter_set)
        graph_loader = DataLoader(graphs, batch_size=len(filter_set))
        semrel_graphs = self.aggregator(next(iter(graph_loader)))
        semrel_graphs = semrel_graphs.reshape(len(filter_set), filter_set.shape[1], -1)  # batch_size x num_local
        agg_embed = semrel_graphs.mean(dim=1)

        return agg_embed
    
    def concept_contrastive_loss(self,global_imm,p_c,n_c,weights):
        B =global_imm.shape[0]
        local_number = 36
        with torch.no_grad():
            local_embeds = self.clip.encode_image(ve.extract_local(global_imm,num_local=local_number,divide=3,crop_mode="random").to(self.device)).reshape(B,local_number,-1)
        main_patch = gen_patch(local_embeds,p_c,weights)

        _,kloss,info_loss = self.cate.ib(main_patch.float(),p_c,n_c)
        cc_loss = kloss + info_loss
        return cc_loss
    
    def cosine_similarity(self,a, b):
        return F.cosine_similarity(a, b, dim=-1)
    
    def calculate_information_density(self,embedding_tensor, selected_tensor):
        """
        计算一个特征与被选择特征之间的信息密度。
        """
        euclidean_distance = torch.norm(embedding_tensor - selected_tensor, dim=-1)  # 计算欧几里得距离
        information_density = 1 / (1 + euclidean_distance)  # 计算信息密度
        return information_density
    

    def select_top_features(self,embedding_tensor, selected_tensor, top_k):

        batch_size, num_features, feature_dim = embedding_tensor.shape
        
        information_density = torch.zeros(batch_size, num_features)
        for i in range(batch_size):
            information_density[i] = self.calculate_information_density(embedding_tensor[i], selected_tensor[i])

        _, topk_indices = torch.topk(information_density, top_k, dim=-1, largest=True)
        topk_indices = topk_indices.to(self.device)

        selected_features = torch.gather(embedding_tensor, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feature_dim))
        
        return selected_features
    
    def select_coarse_concepts(self, features, concept_path, k):

        concepts = torch.load(concept_path,weights_only=True)
        mean = concepts.mean(dim=0)  # 每列的均值
        std = concepts.std(dim=0)    # 每列的标准差
        concepts = (concepts-mean)/std

        # 计算余弦相似度
        concepts_expanded = concepts.unsqueeze(0)  # [1, num_concepts, 512]
        features_expanded = features.unsqueeze(1)  # [batch_size, 1, 512]

        # 计算所有特征与所有concept的余弦相似度
        cosine_similarities = F.cosine_similarity(features_expanded, concepts_expanded, dim=2)  # [batch_size, num_concepts]

        # 对每个特征选择前k个最相似的concept索引
        top_k_similarities, top_k_indices = torch.topk(cosine_similarities, k, dim=1, largest=True)

        _, un_top_k_indices = torch.topk(cosine_similarities, k, dim=1, largest=False)

        # 根据选择的索引从concepts中提取出最相似的k个concept
        selected_concepts = concepts[top_k_indices]  # [batch_size, k, 512]
        un_selected_concepts = concepts[un_top_k_indices]
        return selected_concepts, un_selected_concepts ,top_k_indices, top_k_similarities

    def find_most_similar_concepts(self, p_c, output_filter):

        # 计算余弦相似度
        similarity = F.cosine_similarity(
            output_filter.unsqueeze(2),  # (batch_size, num_locals, 1, 512)
            p_c.unsqueeze(1),            # (batch_size, 1, num_concepts, 512)
            dim=-1
        )  # 输出形状: (batch_size, num_locals, num_concepts)

        # 找到最相似的概念索引
        max_similarity_indices = torch.argmax(similarity, dim=-1)  # 形状: (batch_size, num_locals)

        # 根据索引选择最相似的概念
        batch_size, num_locals = output_filter.shape[0], output_filter.shape[1]
        concepts = p_c[torch.arange(batch_size).unsqueeze(1), max_similarity_indices]  # 形状: (batch_size, num_locals, 512)

        return concepts
    
    torch.no_grad()
    def predict(self, global_logits, local_logits, text_logits, relation_logits, concept_logits):
        pred = (global_logits + (self.recovery_weight * local_logits) + relation_logits+text_logits + (self.recovery_weight * concept_logits) ).max(1, keepdim=True)[1]
        return pred
    