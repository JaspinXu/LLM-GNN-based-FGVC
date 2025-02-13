import os
from itertools import chain

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import sys
sys.path.append('/root/autodl-tmp/my_method')
import clip
from networks.relation_net import DisjointRelationNet, Mapper
from networks.gcn import GCN
from utils import constants
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def select_coarse_concepts(features, concepts, k=20):
    """
    选择每个特征与concept中最相似的前k个concept，并返回它们的余弦相似度。
    
    参数：
    features (Tensor): 形状为 [batch_size, 512] 的特征张量。
    concepts (Tensor): 形状为 [num_concepts, 512] 的概念张量。
    k (int): 选择的最相似concept的数量（默认为10）。

    返回：
    selected_concepts (Tensor): 形状为 [batch_size, k, 512] 的最终选择的概念张量。
    selected_indices (Tensor): 形状为 [batch_size, k] 的索引，表示每个特征对应的最相似concept的索引。
    top_k_similarities (Tensor): 形状为 [batch_size, k] 的张量，表示每个特征对应的最相似concept的余弦相似度。
    """
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






import torch

def normalize_weights(weights):
    """
    归一化每个样本的概念权重，使得每个样本的权重和为 1。
    
    参数：
    weights (Tensor): 形状为 [batch_size, k] 的权重张量。
    
    返回：
    normalized_weights (Tensor): 形状为 [batch_size, k] 的归一化后的权重张量。
    """
    # 对每个样本的权重进行归一化
    weight_sums = weights.sum(dim=1, keepdim=True)  # [batch_size, 1]
    normalized_weights = weights / weight_sums  # 每个样本的权重和为 1
    return normalized_weights

def weighted_cosine_similarity(local_features, concept_anchors, weights):
    """
    计算加权的局部特征与概念锚点之间的余弦相似度。
    
    参数：
    local_features (Tensor): 形状为 [batch_size, num_local_views, 512] 的局部特征张量。
    concept_anchors (Tensor): 形状为 [batch_size, k, 512] 的概念锚点张量。
    weights (Tensor): 形状为 [batch_size, k] 的概念权重张量。
    
    返回：
    weighted_similarities (Tensor): 形状为 [batch_size, num_local_views] 的加权相似度。
    """
    # 归一化权重
    normalized_weights = normalize_weights(weights)  # [batch_size, k]
    
    # 扩展张量以便计算余弦相似度
    concept_expanded = concept_anchors.unsqueeze(1)  # [batch_size, 1, k, 512]
    local_expanded = local_features.unsqueeze(2)    # [batch_size, num_local_views, 1, 512]

    # 计算余弦相似度，dim=3 计算 512 维度的余弦相似度
    cosine_similarities = F.cosine_similarity(local_expanded, concept_expanded, dim=3)  # [batch_size, num_local_views, k]
    
    # 对相似度进行加权
    weighted_similarities = cosine_similarities * normalized_weights.unsqueeze(1)  # [batch_size, num_local_views, k]
    
    # 对每个局部特征按概念维度求和，得到最终的加权相似度
    final_similarities = weighted_similarities.sum(dim=2)  # [batch_size, num_local_views]
    

    return final_similarities

def select_important_vector(a, b):
    """
    根据输入的权重向量 b，选择 a 中每个 batch 中最重要的一个向量。
    
    参数：
    a (Tensor): 形状为 [batch_size, num_local_views, 512] 的张量，表示每个样本的多个局部特征。
    b (Tensor): 形状为 [batch_size, num_local_views] 的张量，表示每个局部特征的重要性权重。
    
    返回：
    Tensor: 形状为 [batch_size, 512] 的张量，表示每个样本选择的最重要的向量。
    """
    # 在 b 中找到每个 batch 中最大值的索引
    _, topk_indices = torch.max(b, dim=1)  # topk_indices 形状为 [batch_size]
    values, indices = torch.topk(b, 2, dim=1)  # 获取每行前两个最大值和它们的索引
    second_largest_indices = indices[:, 1]
    # 使用这些索引从 a 中选择对应的向量
    # 使用 gather 从 a 中按 topk_indices 选择对应的向量
    selected_vectors = torch.gather(a, dim=1, index=topk_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 512))
    #seconed_vectors = torch.gather(a, dim=1, index=second_largest_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 512))

    return selected_vectors

def all_en(global_feature,local_feature,concept_path):
    c = torch.load(concept_path,weights_only=True)
    mean = c.mean(dim=0)  # 每列的均值
    std = c.std(dim=0)    # 每列的标准差
    c = (c-mean)/std
    p_c,n_c,_,weights = select_coarse_concepts(global_feature, c, k=20) ##得到与concept的pt最相似的概念
    distance = weighted_cosine_similarity(local_feature, p_c, weights)
    one_patch = select_important_vector(local_feature, distance)
    return one_patch,p_c,n_c


        