from openai import OpenAI
from tqdm import tqdm

# 创建 OpenAI 客户端
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

# 读取txt文件，假设文件名为 "bird_list.txt"
with open("classes.txt", "r") as file:
    # 获取文件的所有行并为每一行添加进度条
    lines = file.readlines()

    # 使用 tqdm 为 lines 添加进度条
    for line in tqdm(lines, desc="Processing birds", unit="bird"):
        # 假设每一行的格式是：编号 名称（如：1 001.Black_footed_Albatross）
        parts = line.strip().split(" ")
        bird_id = parts[0]
        bird_name = parts[1]

        # 构造查询内容
        user_query = f"What are the distinguishable characteristics that can be used to differentiate a {bird_name} from other birds based on just a photo? Please provide descriptions of the characteristics. The text should be presented in a similar form like [blue eyes], making sure the answers are made up of feature phrases, with different feature phrases separated by commas.Note that only [features] appear, no other information."

        # 发起请求获取生成的文本
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Now you're a bird expert."},
                {"role": "user", "content": user_query},
            ],
            stream=False
        )

        # 获取生成的文本内容
        content = response.choices[0].message.content

        # 将生成的文本保存到文件，文件名可以按需更改
        with open("cub.txt", "a") as output_file:
            output_file.write(content + "\n")
        
        # 打印当前查询的鸟类特征
        print(f"Processed bird: {bird_name}, ID: {bird_id}")
