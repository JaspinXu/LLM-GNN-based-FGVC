import resnet
import torch
import torch.nn as nn
import torch.optim as optim
from cub_dataset import CUBImageDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

# 定义 Mapper 类
class Mapper(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(feature_dim * 2, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, features):
        return self.layers(features)

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为 224x224 大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # 归一化
])

# 数据集路径和标签文件路径
root_dir = '/root/autodl-tmp/my_method/data/CUB_200_2011/images'
label_file = '/root/autodl-tmp/my_method/data/CUB_200_2011/classes.txt'

# 加载数据集
print("Loading dataset...")
dataset = CUBImageDataset(root_dir=root_dir, label_file=label_file, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 模型定义
print("Initializing model...")
backbone = resnet.resnet50(pretrained=False, pth_path="")
num_features = backbone.fc.in_features  # 提取ResNet的最后一层的特征维度
  # 移除全连接层，只保留特征提取部分

# 定义 Mapper 分类头
num_classes = 200  # 假设数据集包含 200 个类
mapper = Mapper(feature_dim=num_features, out_dim=2048, num_classes=num_classes)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
mapper = mapper.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(backbone.parameters()) + list(mapper.parameters()), lr=0.001)

# 训练函数
def train(model, mapper, train_loader, criterion, optimizer, epoch):
    model.train()
    mapper.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        _,features,_ = model(images)  # 提取特征
        outputs = mapper(features)  # 分类头输出
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 10 == 0:  # 每 10 个 batch 打印一次损失
            print(f'Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    print(f'Train Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# 测试函数
def test(model, mapper, test_loader, criterion):
    model.eval()
    mapper.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            _,features,_ = model(images)  # 提取特征
            outputs = mapper(features)  # 分类头输出
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Loss: {running_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%')

# 训练和测试
num_epochs = 100  # 训练 100 个 epoch
for epoch in range(num_epochs):
    train(backbone, mapper, train_loader, criterion, optimizer, epoch)
    test(backbone, mapper, test_loader, criterion)

# 保存最后一轮的权重
torch.save({
    'backbone_state_dict': backbone.state_dict(),
    'mapper_state_dict': mapper.state_dict()
}, 'final_model.pth')

print("Training and testing completed.")

