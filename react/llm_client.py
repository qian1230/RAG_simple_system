# llm_client.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为 "Hello Agents" 定制的LLM客户端，兼容OpenAI接口（如火山方舟）
    """
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = None):
        """
        初始化客户端：优先使用传入参数，否则从.env读取
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 校验必要参数
        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须在.env中配置！")

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        调用LLM生成响应（流式输出）
        :param messages: 对话消息列表，格式[{"role": "user/system", "content": "内容"}]
        :param temperature: 生成温度，0-1之间
        :return: 完整响应文本，失败返回None
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                max_tokens=2048
            )

            # 处理流式响应
            print("✅ LLM响应中:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()  # 流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"\n❌ LLM调用失败: {str(e)}")
            return None

# 测试代码（可选）
if __name__ == '__main__':
    try:
        llm = HelloAgentsLLM()
        test_msg = [{"role": "user", "content": "你好，测试一下！"}]
        llm.think(test_msg)
    except ValueError as e:
        print(e)