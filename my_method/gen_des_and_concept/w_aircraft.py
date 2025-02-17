from openai import OpenAI
from tqdm import tqdm

# 创建 OpenAI 客户端
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

# 读取飞机型号列表文件
with open("/root/autodl-tmp/fine/my_method/data/aircraft/fgvc-aircraft-2013b/data/variants.txt", "r") as file:
    lines = file.readlines()

    # 使用 tqdm 添加进度条
    for line in tqdm(lines, desc="Processing Aircraft", unit="model"):

        # 构造飞机特征查询
        user_query = f"What are the key visual characteristics that distinguish a {line} aircraft from others? Focus on distinctive features like engine configuration, wing shape, livery patterns, tail design, etc. Provide short feature phrases separated by commas, formatted like [high-wing design], [two turbofan engines]. Only list features, no explanations."

        # 发起API请求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an aviation expert specializing in aircraft recognition."},
                {"role": "user", "content": user_query},
            ],
            stream=False
        )

        # 获取并保存结果
        content = response.choices[0].message.content
        
        with open("/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft.txt", "a") as output_file:
            output_file.write(f"{content}\n")

        # 打印进度
        print(f"Processed: {line}")