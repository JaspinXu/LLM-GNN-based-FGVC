from openai import OpenAI
import os
import base64
from tqdm import tqdm
import json

# 设置 API 密钥
api_key = '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'
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# 编码图片为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 处理单个图片文件并保存结果到 JSON
def process_image(image_path, output_json_file):
    try:
        # 加载现有的 JSON 数据
        if os.path.exists(output_json_file):
            with open(output_json_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # 如果图片已经处理过，跳过
        if os.path.basename(image_path) in data:
            print(f"Skipping {image_path} (already processed)")
            return

        # 调用 OpenAI API 处理图片
        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 使用正确的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is a picture of an aircraft. Please start with a one-sentence description of the aircraft as a whole."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content
        print(f"Processed {image_path}: {text}")

        # 更新数据并保存到 JSON 文件
        data[os.path.basename(image_path)] = text
        with open(output_json_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results updated in {output_json_file}")
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

# 处理文件夹中的所有图片
def des_gen(output_json_file, file_path):
    try:
        # 检查 file_path 是文件还是文件夹
        if os.path.isfile(file_path):
            # 如果是单个文件，直接处理
            process_image(file_path, output_json_file)
        elif os.path.isdir(file_path):
            # 如果是文件夹，遍历所有图片文件
            supported_formats = (".jpg", ".jpeg", ".png")
            image_files = [f for f in os.listdir(file_path) if f.lower().endswith(supported_formats)]
            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(file_path, image_file)
                process_image(image_path, output_json_file)
        else:
            print(f"Invalid path: {file_path}")
            return
    except Exception as e:
        print(f"Failed: {e}")

# 示例调用
output_json_file = "/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft_image_text.json"
file_path = "/root/autodl-tmp/fine/my_method/data/aircraft/fgvc-aircraft-2013b/data/images/"
des_gen(output_json_file, file_path)