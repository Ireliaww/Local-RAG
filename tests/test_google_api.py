import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. 从.env文件加载API Key
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("错误: 请在.env文件中设置GOOGLE_API_KEY环境变量")
    exit(1)

def quick_check():
    print("--- 正在进行极简环境测试 ---")
    
    # 2. 初始化模型 (使用 Flash 版本，速度快且限额更高)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    try:
        # 3. 发送最简单的请求
        print("⏳ 正在等待 Google 响应...")
        response = llm.invoke("Hi! Are you ready?")
        
        print("\n✅ 测试成功！")
        print(f"🤖 AI 回复: {response.content}")
        
    except Exception as e:
        if "429" in str(e):
            print("\n❌ 触发频率限制 (429)。请等待 60 秒后再试。")
        else:
            print(f"\n❌ 报错了: {e}")

if __name__ == "__main__":
    quick_check()