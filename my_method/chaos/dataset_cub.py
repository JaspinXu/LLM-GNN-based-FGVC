import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.measure import regionprops
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm
def calculate_area(image):
    """计算图像的面积（像素数量）"""
    return image.size

def slic_single_image(image):
    max_area=0
    segments = slic(image, n_segments=5, enforce_connectivity=True, sigma=1, compactness=10.0)
    image_list=[]
# 使用掩码剪切原始图像并保存到result文件夹
    for region_number in range(1, len(regionprops(segments)) + 1):
        region = regionprops(segments)[region_number - 1]  # 注意：regionprops返回的区域编号从1开始
        minr, minc, maxr, maxc = region.bbox
        cropped_image = image[minr:maxr, minc:maxc]
        image_list.append(cropped_image)
    for image in image_list:
        area = calculate_area(image)
        if area > max_area:
            max_area = area
            max_image = image
    image = Image.fromarray(max_image.astype('uint8'))
    image.save('output_image.png')
    new_size = (192, 192)
    image = image.resize(new_size)
    image = np.array(image)
    rows = np.vsplit(image, 3)  # 按高度分割图像为3部分
    blocks = [np.hsplit(row, 3) for row in rows]  # 然后对每一部分按宽度分割为3部分
    blocks = [item for sublist in blocks for item in sublist]  # 展平列表
    return blocks

def find_images(directory):
    """
    递归查找指定目录及其子目录中的所有图像文件。
    
    参数:
    - directory: 要搜索的根目录路径。
    
    返回:
    - 一个列表，包含所有找到的图像文件的完整路径。
    """
    images = []
    # 遍历目录及其子目录
    for root, dirs, files in tqdm(os.walk(directory), desc="Scanning directory"):
        for dir in dirs:
            #print(dir)
            # 构造文件的完整路径
            file_path_all = os.path.join(root, dir)
            for filename in os.listdir(file_path_all):
                # 构造完整的文件路径
                file_path = os.path.join(file_path_all, filename)
                print(file_path)
                # 检查文件是否为图像文件（根据文件扩展名判断，可以根据需要调整）
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # 读取图像文件
                    image = imageio.imread(file_path)
                    images.append(image)

    return images
'''
# 确保结果文件夹存在
output_folder = '/root/autodl-tmp/my_method/data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


image_folder = '/root/autodl-tmp/my_method/data'

# 确保路径存在
if not os.path.isdir(image_folder):
    raise ValueError(f"The folder {image_folder} does not exist.")

# 初始化一个空列表来存储读取的图像
images = []

# 遍历文件夹中的所有文件
for filename in os.listdir(image_folder):
    # 构造完整的文件路径
    file_path = os.path.join(image_folder, filename)
    
    # 检查文件是否为图像文件（根据文件扩展名判断，可以根据需要调整）
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 读取图像文件
        image = imageio.imread(file_path)
        images.append(image)
input_data=[]
for img in tqdm(images, desc="Processing images"):
    block = slic_single_image(img)
    input_data.append(block)
print(len(input_data))
print(len(input_data[0]))'''
def process_images(images):
    """
    处理图像列表中的所有图像。
    
    参数:
    - images: 一个包含图像文件路径的列表。
    """
    input_data = []
    for img_path in tqdm(images, desc="Processing images"):
        # 读取图像文件
        image = imageio.imread(img_path)
        # 假设equalize_single_image是一个处理单个图像的函数
        block = slic_single_image(image)
        input_data.append(block)
    return input_data

def main():
    # 定义包含小文件的文件夹的路径
    folder_with_small_files = '/root/autodl-tmp/my_method/data/CUB_200_2011/images'
    
    # 查找所有图像文件
    images = find_images(folder_with_small_files)
    
    # 处理所有找到的图像
    input_data = process_images(images)
    print(len(input_data))

if __name__ == "__main__":
    main()