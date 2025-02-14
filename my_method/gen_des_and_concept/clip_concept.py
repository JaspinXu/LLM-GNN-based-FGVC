import torch
import sys
import os
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))
from longclip.model import longclip

def extract_features(file_path):
    # 用于存储所有行的特征
    all_features = []

    # 打开并读取txt文件
    with open(file_path, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 去除每行的首尾空白字符
            line = line.strip()

            # 每行的特征短语是用逗号分隔的，并且每个短语在方括号中
            features = [feature.strip('[] ').strip() for feature in line.split('],')]

            # 将每行的特征短语添加到总列表中
            all_features.append(features)

    return all_features

def generate_word_embeddings(word_list, model, preprocess):
    # 使用模型的文本编码器来获取嵌入
    text_inputs = longclip.tokenize(word_list).to("cuda")  # 转换单词为CLIP输入
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)  # 获取文本的嵌入
    return text_features

def save_embeddings_to_file(embeddings, file_path):
    # 保存嵌入到.pt文件
    torch.save(embeddings, file_path)

# 调用函数并输出结果
if __name__ == "__main__":
    # model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = longclip.load("/root/autodl-tmp/fine/my_method/longclip/checkpoints/longclip-B.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    file_path = '/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft.txt'  # 替换为你的txt文件路径
    features = extract_features(file_path)
    feature_list = [item for sublist in features for item in sublist]
    print(len(feature_list))
    feature_list=list(set(feature_list))
    embeddings = generate_word_embeddings(feature_list, model, preprocess)

    # 保存嵌入到 .pt 文件
    save_embeddings_to_file(embeddings, "/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft_concepts__512_longclip.pt")
    print(len(feature_list))
    

