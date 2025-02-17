import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
# from model import graphgen,GCN,compute_reprs
from openai import OpenAI
import os
import base64
import json

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
class CUBImageDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        """
        初始化数据集

        参数:
        - root_dir: 包含小文件夹的大文件夹路径
        - label_file: 包含文件夹名称和标签的TXT文件路径
        - transform: 图像的变换操作（如 transforms.Compose([...])）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []   #0-199
        self.label_mapping = self._load_label_mapping(label_file)

        # 遍历大文件夹，收集所有图像路径及其对应的标签
        for sub_dir in sorted(os.listdir(root_dir)):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                label = self.label_mapping.get(sub_dir, None)
                if label is None:
                    raise ValueError(f"在标签文件中找不到文件夹 '{sub_dir}' 的标签")
                
                for file_name in os.listdir(sub_dir_path):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 过滤图像文件
                        file_path = os.path.join(sub_dir_path, file_name)
                        self.image_paths.append(file_path)
                        self.labels.append(label-1)

    def _load_label_mapping(self, label_file):
        """从标签文件中加载文件夹名称和标签的映射关系"""
        label_mapping = {}
        with open(label_file, 'r') as file:
            for line in file:
                label, folder_name = line.strip().split(' ', 1)
                label_mapping[folder_name] = int(label)  # 确保标签是整数
        return label_mapping

    def __len__(self):
        """返回数据集中图像的数量"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """根据索引 idx 返回一对图像和标签"""
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        with open('/root/autodl-tmp/fine/my_method/data/cub2011/des_and_concept/cub_image_text.json', 'r') as file:
            data = json.load(file)  # 将JSON加载为字典
        description = data.get(image_name, "未找到与该图像名称匹配的解释。")
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
        if self.transform:
            image = self.transform(image)

        return (image,description), label
    

device = 'cuda'
root_dir = '/root/autodl-tmp/fine/my_method/data/cub2011/CUB_200_2011/images'
label_file = '/root/autodl-tmp/fine/my_method/data/cub2011/CUB_200_2011/classes.txt'
# 数据变换操作
# 数据变换操作
transform = transforms.Compose([
    transforms.Resize((672, 672)),  # 将图像调整为 32x32 大小
    transforms.ToTensor(),        # 转换为张量
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # 归一化
])
bs=64
dataset = CUBImageDataset(root_dir=root_dir, label_file=label_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
dataset.__getitem__(0) 
'''
torch.Size([3, 672, 672])
((tensor([[[ 1.3172,  1.3172,  1.3318,  ...,  1.6092,  1.6092,  1.6092],
         [ 1.3172,  1.3172,  1.3172,  ...,  1.6092,  1.6238,  1.6238],
         [ 1.3172,  1.3172,  1.3172,  ...,  1.6092,  1.6238,  1.6238],
         ...,
         [ 1.3464,  1.3464,  1.3464,  ...,  0.8063,  0.7771,  0.7771],
         [ 1.3756,  1.3756,  1.3756,  ...,  0.7771,  0.7479,  0.7333],
         [ 1.4048,  1.4048,  1.3902,  ...,  0.7333,  0.7041,  0.6895]],

        [[ 1.4446,  1.4446,  1.4596,  ...,  1.8648,  1.8498,  1.8498],
         [ 1.4446,  1.4446,  1.4446,  ...,  1.8648,  1.8648,  1.8648],
         [ 1.4446,  1.4446,  1.4446,  ...,  1.8798,  1.8798,  1.8798],
         ...,
         [ 1.5946,  1.5946,  1.5946,  ...,  0.6642,  0.6491,  0.6341],
         [ 1.6247,  1.6247,  1.6247,  ...,  0.6341,  0.6041,  0.5891],
         [ 1.6547,  1.6547,  1.6397,  ...,  0.5891,  0.5591,  0.5441]],

        [[ 1.6624,  1.6624,  1.6766,  ...,  2.0037,  2.0179,  2.0179],
         [ 1.6624,  1.6624,  1.6624,  ...,  2.0037,  2.0179,  2.0321],
         [ 1.6624,  1.6624,  1.6624,  ...,  2.0037,  2.0179,  2.0179],
         ...,
         [ 1.8757,  1.8757,  1.8615,  ..., -0.2431, -0.2573, -0.2573],
         [ 1.9042,  1.9042,  1.8757,  ..., -0.2715, -0.2857, -0.2857],
         [ 1.9326,  1.9184,  1.8899,  ..., -0.3142, -0.3284, -0.3284]]]), 
         'This bird appears to resemble a type of seabird, particularly one similar to an albatross or a petrel. In the image, the bird has a predominantly gray body, with a lighter face and a distinct beak. It stands on a sandy surface, surrounded by green foliage. The posture suggests it is engaged in some form of activity, possibly walking or foraging. These characteristics are typical of seabirds that inhabit coastal areas.'), 0)
'''

'''
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# 迭代加载数据
for images, labels in dataloader:
    print(f'图像批量大小: {images.shape}, 标签批量大小: {labels.shape}')
    image = images.reshape(bs,3,3,224,3,224)
    iamge = image.permute(0, 2, 4, 1, 3, 5)  # [128, 3, 3, 3, 64, 64]

# 3. 合并前两个维度 3 x 3 = 9
    image = image.reshape(bs, 9, 3, 224, 224)  # [128, 9, 3, 64, 64]
    #print(image[0].shape)
    image_data = image.view(-1,3,224,224).to(device)
    print("----------------")
    #print(image_data.shape)
    with torch.no_grad():
        image_features = model.encode_image(image_data)
    output = image_features.view(64,9,512)
    print(output.shape)
    semrel_graphs = compute_reprs(output,512,350)
    print(semrel_graphs.shape)
    break  # 仅显示一个批次'''
'''
'''