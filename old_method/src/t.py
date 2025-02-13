import torch
import torch.nn as nn
import clip
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

path = "/root/autodl-tmp/fine/my_method/word_embeddings.pt"
c = torch.load(path)
mean = c.mean(dim=0)  # 每列的均值
std = c.std(dim=0)    # 每列的标准差
c = (c-mean)/std
print(c.shape)
import clip
import torch
from PIL import Image

# 加载CLIP模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# 读取图像
image = Image.open("/root/autodl-tmp/fine/Dataset/CUB/train/070.Green_Violetear/Green_Violetear_0047_795677.jpg")  # 替换为你的图像路径

# 对图像进行预处理
image_input = preprocess(image).unsqueeze(0).to(device)

# 获取图像的特征向量
with torch.no_grad():
    image_features = model.encode_image(image_input)

# 输出图像的特征向量
print(image_features.shape)


p_c,n_c,ind,weights = select_coarse_concepts(image_features, c, k=20)
print(ind)
w = normalize_weights(weights)
w = torch.reciprocal(w)
print(w)