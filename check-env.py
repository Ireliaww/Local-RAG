import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 从环境变量读取API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("错误: 请在.env文件中设置GOOGLE_API_KEY环境变量")
    exit(1)

client = genai.Client(api_key=api_key)

try:
    print("--- 正在查询你的可用模型列表 ---")
    models = client.models.list()
    for m in models:
        print(f"可用模型名称: {m.name}")
except Exception as e:
    print(f"查询失败: {e}")