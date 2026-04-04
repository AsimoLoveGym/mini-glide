from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
# override=True：若终端里曾 export 过旧 key，仍优先使用项目根目录 .env
load_dotenv(_ROOT / ".env", override=True)
client = OpenAI()  # 自动读取 OPENAI_API_KEY


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """统一的 LLM 调用函数，方便后续换模型"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def translate_v1(en_text: str) -> str:
    """Step 1：基础翻译，无任何优化"""
    system = (
        "你是一名专业翻译，专注于将英文内容翻译成自然流畅的中文。"
        "翻译风格要符合短视频平台：口语化、简洁、有表现力，不要过于书面化。"
    )
    user = f"请翻译以下英文内容：\n\n{en_text}\n\n只输出译文，不加任何说明。"
    return call_llm(system, user)


# 快速测试
if __name__ == "__main__":
    test = "The quick brown fox jumps over the lazy dog."
    result = translate_v1(test)
    print(f"EN: {test}")
    print(f"ZH: {result}")
