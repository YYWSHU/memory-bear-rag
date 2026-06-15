import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""生成模块 - DeepSeek API调用 + 作者风格Prompt工程"""
import json
from typing import Dict, List, Generator
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


SYSTEM_PROMPT = """你正在扮演"碧树西风"——微信公众号"记忆承载"的作者。

## 你的身份特征
你是一位深谙中国社会运行逻辑的思考者，擅长用利益分析法、历史周期律、人性底层逻辑来剖析问题。
你的文章涵盖经济、房地产、教育、职场、社会阶层、投资等领域。

## 你的思维特征
- **利益分析法**: 一切问题的本质都是利益分配问题，从"谁受益、谁受损"的角度分析
- **历史周期律**: 太阳底下没有新鲜事，用历史规律解释当下现象
- **人性底层逻辑**: 看透人性中的贪婪、恐惧、从众心理
- **先破后立**: 先推翻读者的常见认知，再给出自己的见解
- **极端场景推演**: 把一个问题推到极端，规律就浮出水面
- **利益计算**: 把模糊的问题量化为利益得失
- **从众批判**: 对主流观点保持警惕，强调独立思考

## 你的语言风格
- 口语化但有深度，像跟老朋友聊天
- 善用类比和故事来解释复杂概念
- 标题常常带有悬念和反直觉
- 常用"说到底"、"本质上"、"你以为...其实..."等句式
- 敢于谈敏感话题，但不碰红线

## 回答要求
1. 基于提供的文章内容回答问题，优先使用原文中的观点和论证
2. 模仿作者的思维方式和语言风格
3. 展示你的推理过程：前提→分析→结论
4. 引用具体的文章观点作为依据
5. 如果信息不足以回答，诚实地说明
6. 回答末尾标注引用的文章标题
"""


class ThinkingStyleGenerator:
    """带思维风格的生成器"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.model = model or DEEPSEEK_MODEL

    def generate(self, query: str, context: str,
                 chat_history: List[Dict] = None) -> str:
        """生成回答（非流式）"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加上下文
        messages.append({
            "role": "user",
            "content": f"""以下是"记忆承载"文章中的相关内容，请基于这些内容回答用户问题。

{context}

---
用户问题：{query}

请模仿碧树西风的思维方式和语言风格，基于上述文章内容来回答。展示你的分析过程。"""
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[生成错误] {str(e)}"

    def generate_stream(self, query: str, context: str,
                        chat_history: List[Dict] = None) -> Generator[str, None, None]:
        """流式生成回答"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        messages.append({
            "role": "user",
            "content": f"""以下是"记忆承载"文章中的相关内容，请基于这些内容回答用户问题。

{context}

---
用户问题：{query}

请模仿碧树西风的思维方式和语言风格，基于上述文章内容来回答。展示你的分析过程。"""
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[生成错误] {str(e)}"


if __name__ == "__main__":
    gen = ThinkingStyleGenerator()
    test_context = "测试上下文：作者认为房地产问题的本质是利益分配问题..."
    test_query = "怎么看当前的房地产市场？"
    result = gen.generate(test_query, test_context)
    print(result)
