import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from openai import OpenAI
import os
import base64
import json
from utils import constants
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = OpenAI()


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torchvision.transforms as transforms

class CUBImageDataset(Dataset):
    def __init__(self, split_file,label_file, is_train=True, transform=None):
        """
        初始化数据集，加载划分信息，并初始化必要的转换。

        参数:
        - split_file: 数据集划分文件路径（如 'data_split.json'）
        - is_train: 是否为训练集。True 表示加载训练集，False 表示加载测试集
        - transform: 预处理转换操作（如图像缩放、标准化等）
        """
        with open(split_file, 'r') as file:
            data = json.load(file)
        
        # 根据 is_train 选择加载训练集或测试集
        self.image_paths = data['train_indices']+data['test_indices'] if is_train else data['test_indices']
        self.transform = transform
        self.label_mapping = self._load_label_mapping(label_file)
    def __len__(self):
        """
        返回数据集的大小，即图像的数量。
        """
        return len(self.image_paths)
    def _load_label_mapping(self, label_file):
        """从标签文件中加载文件夹名称和标签的映射关系"""
        label_mapping = {}
        with open(label_file, 'r') as file:
            for line in file:
                label, folder_name = line.strip().split(' ', 1)
                label_mapping[folder_name] = int(label)  # 确保标签是整数
        return label_mapping

    def __getitem__(self, idx):
        """
        根据索引获取数据样本。
        
        参数:
        - idx: 数据集中的索引

        返回:
        - 图像和对应的标签（如果需要）
        """
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为 RGB 格式
        with open('/root/autodl-tmp/my_method/cub_image_text.json', 'r') as file:
            data = json.load(file)  # 将JSON加载为字典
        description = data.get(image_name, "未找到与该图像名称匹配的解释。")
        # 如果定义了转换操作，应用转换
        if self.transform:
            image = self.transform(image)
        
        # 从路径中提取标签（假设文件夹结构中标签为文件夹名称）
        k = img_path.split(os.sep)[-2]  # 假设标签在路径的倒数第二个目录中
        label = self.label_mapping.get(k, None)
        label= label-1

        return (image,description), label

import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir,label_file, transform=None):
        """
        Args:
            root_dir (str): 根目录，包含所有类别的文件夹。
            transform (callable, optional): 可选的变换函数，应用于样本。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []
        self.label_mapping = self._load_label_mapping(label_file)

        for i, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)
    
    def _load_label_mapping(self, label_file):
        """从标签文件中加载文件夹名称和标签的映射关系"""
        label_mapping = {}
        with open(label_file, 'r') as file:
            for line in file:
                label, folder_name = line.strip().split(' ', 1)
                label_mapping[folder_name] = int(label)  # 确保标签是整数
        return label_mapping

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        k = img_path.split(os.sep)[-2]  # 假设标签在路径的倒数第二个目录中
        label = self.label_mapping.get(k, None)
        label= label-1
        with open('/root/autodl-tmp/fine/my_method/cub_image_text.json', 'r') as file:
            data = json.load(file)  # 将JSON加载为字典
        description = data.get(image_name, "未找到与该图像名称匹配的解释。")

        return (image,description), label

    
'''
device = 'cuda'
root_dir = '/root/autodl-tmp/my_method/data/CUB_200_2011/images'
label_file = '/root/autodl-tmp/my_method/data/CUB_200_2011/classes.txt'
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
