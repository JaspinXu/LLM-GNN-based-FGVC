import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage import measure


def extract_global(x, extractor):
    """
    Adopted from MMAL --
    Paper: https://arxiv.org/pdf/2003.09150.pdf
    Code: https://github.com/ZF4444/MMAL-Net
    """
    fms, _, fm1 = extractor(x)
    batch_size, channel_size, side_size, _ = fms.shape
    fm1 = fm1.detach()

    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()

    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(7, 7)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = ((component_labels == (max_idx + 1)).astype(int) + (M1[i][0].cpu().numpy() == 1).astype(
            int)) == 2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            # print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)

    # SCDA
    global_view = torch.zeros([batch_size, 3, 224, 224])  # [N, 3, 448, 448]
    for i in range(batch_size):
        [x0, y0, x1, y1] = coordinates[i]
        global_view[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                             mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
    return global_view


import torch
import torchvision.transforms as T

def split_into_parts(image, num_parts):
    """
    将图像切割为指定数量的均等部分，并将每个部分调整为 224x224。

    参数：
    - image: 输入图像，形状为 [B, C, H, W]。
    - num_parts: 需要切割的总份数，可以是 4, 9, 16, 或 25。

    返回：
    - 计算出的裁剪区域列表，每个裁剪区域都被调整为 224x224，形状为 [B, C, 224, 224]。
    """
    B, C, H, W = image.shape  # 获取批量大小，通道数和图像的高宽
    n = int(num_parts ** 0.5)  # n 是切割网格的大小，例如 num_parts = 16 时，n = 4，表示 4x4 网格

    # 计算每个区域的大小
    crop_height, crop_width = H // n, W // n

    crops = []
    resize = T.Resize((224, 224))  # 初始化 Resize 操作

    # 逐行逐列裁剪每个网格的区域
    for i in range(n):  # 切成 n 行
        for j in range(n):  # 切成 n 列
            # 按照索引计算裁剪的区域
            crop = image[:, :, i*crop_height:(i+1)*crop_height, j*crop_width:(j+1)*crop_width]
            
            # 将每个裁剪区域调整为 224x224
            crop_resized = resize(crop)
            
            crops.append(crop_resized)

    # 返回裁剪的部分（B, C, 224, 224）的列表
    return crops



def extract_local(global_view, num_local, crop_mode):
    """
    根据 crop_mode 切割图像为多个局部区域。
    
    参数：
    - global_view: 输入图像，形状为 [C, H, W]。
    - num_local: 切割的数量，例如 4, 9, 16, 或 25。
    - crop_mode: 切割模式，可选 'random' 或 'five_crops'，或其他自定义模式。
    
    返回：
    - local_views: 切割后的局部图像列表。
    """
    if crop_mode == 'random':
        random_cropper = T.Compose([T.RandomCrop(size=global_view.shape[-1] // 3), T.Resize((224, 224))])
        local_views = [random_cropper(global_view) for _ in range(num_local)]
        
    elif crop_mode in ['4', '9', '16', '25']:  # 根据传入的切割模式
        local_views = split_into_parts(global_view, int(num_local))
    
    else:
        raise ValueError("Unsupported crop_mode. Supported modes are 'random', '4', '9', '16', or '25'.")
    
    local_views = torch.cat(local_views)
    return local_views

