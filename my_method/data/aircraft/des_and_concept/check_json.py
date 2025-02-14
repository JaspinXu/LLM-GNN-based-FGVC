import json
import re
import chardet  # 用于检测文件编码

def detect_file_encoding(file_path):
    """
    检测文件的编码格式。

    :param file_path: 文件路径
    :return: 检测到的编码格式
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def is_valid_string(s):
    """
    检查字符串是否只包含常用符号（ASCII可打印字符、中文、标点符号等）。

    :param s: 需要检查的字符串
    :return: 如果字符串合法返回True，否则返回False
    """
    # 定义允许的字符范围：ASCII可打印字符、中文、常见标点符号
    allowed_pattern = re.compile(
        r'[\u4e00-\u9fa5]'  # 中文字符
        r'|[a-zA-Z0-9]'      # 字母和数字
        r'|[ \!\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]'  # 常见标点符号
    )
    # 检查字符串中是否有不允许的字符
    for char in s:
        if not allowed_pattern.match(char):
            return False
    return True

def check_json_for_invalid_symbols(json_file_path):
    """
    检查JSON文件中的字符串值是否包含非常用符号，并返回有问题的行号。

    :param json_file_path: JSON文件的路径
    :return: 包含非常用符号的行号列表
    """
    problematic_lines = []

    try:
        # 检测文件编码
        encoding = detect_file_encoding(json_file_path)
        print(f"Detected file encoding: {encoding}")

        with open(json_file_path, 'r', encoding=encoding) as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    # 检查字典中的字符串值
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, str) and not is_valid_string(value):
                                problematic_lines.append(line_number)
                                print(f"Invalid symbol found in line {line_number}, key: {key}, value: {value}")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in line {line_number}: {e}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return problematic_lines

# 示例调用
if __name__ == "__main__":
    json_file_path = '/root/autodl-tmp/fine/my_method/data/aircraft/des_and_concept/aircraft_image_text.json'  # 替换为你的JSON文件路径
    problematic_lines = check_json_for_invalid_symbols(json_file_path)
    if problematic_lines:
        print(f"Problematic lines: {problematic_lines}")
    else:
        print("No problematic lines found.")