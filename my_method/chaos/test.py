import os
import torch
import clip
from PIL import Image
def filter_top_50_percent(features, scores):
    """
    根据特征分数筛选出前 50% 的特征，并打印排序后的前 10 名特征及其位置。
    
    Args:
        features (list): 每个图像的特征列表，格式为 [feature1, feature2, ..., featureN]。
        scores (list): 每个图像的特征分数，格式为 [score1, score2, ..., scoreN]。
        
    Returns:
        list: 排序后的前 50% 特征列表。
    """
    # 将特征和分数结合为一个列表的元组形式 [(score1, feature1, original_index), ...]
    combined = list(zip(scores, features, range(len(features))))
    
    # 根据分数对结合的列表进行排序，分数越大排得越前
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # 打印排序后的前 10 名特征和它们的原始位置
    print("排序后的前 10 名特征及其原始位置:")
    for i, (score, feature, original_index) in enumerate(combined[:20]):
        print(f"排名 {i+1}: '，原始位置 {original_index}，分数 {score}")
    print("排序后的后 10 名特征及其原始位置:")
    for i, (score, feature, original_index) in enumerate(combined[-10:]):
        print(f"排名 {len(combined)-9+i}: '，原始位置 {original_index}，分数 {score}")

    
    # 计算前 50% 的数量
    num_top = len(features) // 2
    
    # 获取前 50% 的特征
    top_50_percent_features = [feature for _, feature, _ in combined[:num_top]]
    
    return top_50_percent_features

def calculate_information_density(embedding_tensor, cluster_center_tensor):
    """
    计算一个节点嵌入与其簇中心之间的信息密度。
    
    Args:
        embedding_tensor (torch.Tensor): 节点的嵌入向量。
        cluster_center_tensor (torch.Tensor): 节点所属簇的簇中心向量。
        
    Returns:
        float: 节点信息密度。
    """
    # 计算嵌入向量与簇中心之间的欧几里得距离
    euclidean_distance = torch.norm(embedding_tensor - cluster_center_tensor)
    
    # 计算信息密度，1 / (1 + 距离)
    information_density = 1 / (1 + euclidean_distance.item())
    
    return information_density

class ClipImageSimilarity:
    """
    使用 CLIP 模型计算图像与给定文本（如 "bird"）的相似度。
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 CLIP 模型和设备。
        """
        self.device = device
        print(f"使用的设备: {self.device}")
        
        # 加载 CLIP 模型和其对应的标记器
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def calculate_similarity(self, image_path: str, text: str = "bird") -> float:
        """
        计算图像与文本的相似度。
        
        Args:
            image_path (str): 图像的路径。
            text (str): 要比较的文本，默认为“bird”。
            
        Returns:
            float: 图像与文本的相似度分数。
        """
        try:
            # 读取图像并使用CLIP的预处理方法进行转换
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # 将文本标记化
            text_input = clip.tokenize([text]).to(self.device)

            # 使用CLIP对图像和文本编码
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                #print(image_features.shape)
                text_features = self.model.encode_text(text_input)

            # 归一化特征向量
            #image_features /= image_features.norm(dim=-1, keepdim=True)
            #text_features /= text_features.norm(dim=-1, keepdim=True)

            # 计算相似度（点积形式）
            similarity = (image_features @ text_features.T).item()
            return similarity,image_features
        except Exception as e:
            print(f"处理图像 {image_path} 时发生错误: {e}")
            return 0.0

    def find_most_similar_image(self, folder_path: str, text: str = "bird") -> str:
        """
        遍历文件夹，计算其中每张图像与指定文本的相似度，并找出相似度最高的图像。
        
        Args:
            folder_path (str): 包含图像的文件夹路径。
            text (str): 要比较的文本，默认为“bird”。
            
        Returns:
            str: 相似度最高的图像路径。
        """
        highest_similarity = -1
        most_similar_image = None
        most_similar_feature = None
        feature_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                similarity,feature = self.calculate_similarity(image_path, text)
                feature_list.append(feature)
                print(f"图像: {image_path}, 与 \"{text}\" 的相似度: {similarity:.4f}")
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_image = image_path
                    most_similar_feature = feature
        
        return most_similar_image, highest_similarity,feature_list,most_similar_feature


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用 CLIP 计算图像与文本的相似度，并找出最相似的图像。")
    parser.add_argument('--folder', type=str, required=True, help="包含图像的文件夹路径。")
    parser.add_argument('--text', type=str, default="bird", help="用于与图像比较的文本，默认为 'bird'。")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="使用的设备 (cuda 或 cpu)")

    args = parser.parse_args()
    folder_path = args.folder
    text = args.text
    device = args.device
    score =[]
    # 初始化 CLIP 模型
    clip_similarity = ClipImageSimilarity(device=device)

    # 找出与文本相似度最高的图像
    most_similar_image, similarity ,latent_list,best_feature= clip_similarity.find_most_similar_image(folder_path, text)

    if most_similar_image:
        print(f"\n最相似的图像是: {most_similar_image}, 相似度: {similarity:.4f}")
    else:
        print("没有找到任何图像。")
    for i in latent_list:
        s = calculate_information_density(i,best_feature)
        score.append(s)
    top_features = filter_top_50_percent(latent_list, score)
