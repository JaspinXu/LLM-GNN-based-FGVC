from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from chaos.cub_dataset import CUBImageDataset
from options import Options
from train_ import MultiViewFGVC
from torch.utils.data import random_split
import resnet
from longclip.model import longclip
import time
from fgvc_dataset.cub2011 import Cub2011
from fgvc_dataset.aircraft import Aircraft

def main():
    args = Options().parse()
    device = 'cuda'

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 将图像调整为 224x224 大小
    #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为 0.5
    #     transforms.ToTensor(),        # 转换为张量
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # 归一化
    # ])
    # dataset = CUBImageDataset(root_dir=args.root_dir, label_file=args.label_file, transform=transform)
    # print(len(dataset.image_paths))
    #     # 指定训练集和测试集的大小
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_dataset = Cub2011('/root/autodl-tmp/fine/my_method/data/cub2011', train=True)
    # test_dataset = Cub2011('/root/autodl-tmp/fine/my_method/data/cub2011', train=False)
    train_dataset = Aircraft('/root/autodl-tmp/fine/my_method/data/aircraft', train=True)
    test_dataset = Aircraft('/root/autodl-tmp/fine/my_method/data/aircraft', train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,drop_last=True)   

    # for i, data in enumerate(train_loader):
    #     (im,im_text), labels = data
    model, _ = longclip.load("/root/autodl-tmp/fine/my_method/longclip/checkpoints/longclip-B.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    # model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load(args.ckpt_path)
    # model.load_state_dict(state_dict, strict=True)
    extractor = resnet.resnet50(pretrained=True, pth_path="/root/autodl-tmp/fine/my_method/resnet50-19c8e357.pth")
    train_model = MultiViewFGVC(input_dim=512,feature_dim=512,num_classes = 100,device='cuda',clip_model=model,lr=args.lr,extractor=extractor)
    # train_model = EnhancedMultiViewHausdorff(input_dim=512,feature_dim=512,device='cuda',clip_model=model,lr=args.lr,extractor=extractor,circle_layers=5, circle_step=20)    
    for epoch in range(args.epochs):
        train_model.train_one_epoch(train_loader, epoch,args.savepath)
        train_model.test(test_loader, epoch)

if __name__ == "__main__":
    main()