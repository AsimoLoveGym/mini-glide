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


def critique(en_text: str, zh_v1: str) -> str:
    """Step 2：分析 v1 的具体问题"""
    system = (
        "你是一名资深翻译质量评审员。"
        "你的工作是找出译文中的具体问题，帮助改进翻译质量。"
        "请简洁、具体地列出问题，不要泛泛而谈。"
    )
    user = (
        f"原文：{en_text}\n\n"
        f"当前译文：{zh_v1}\n\n"
        f'请从三个维度简洁列出问题（每条一行，如无问题就写"无"）：\n'
        f"1. 准确性问题：\n"
        f"2. 流畅性问题：\n"
        f"3. 风格问题（是否符合短视频平台）："
    )
    return call_llm(system, user, temperature=0.2)


def revise(en_text: str, zh_v1: str, critique_text: str) -> str:
    """Step 3：根据 critique 生成改进版 v2"""
    system = "你是一名专业翻译。你的任务是根据评审意见改进现有译文。"
    user = (
        f"原文：{en_text}\n\n"
        f"当前译文（需要改进）：{zh_v1}\n\n"
        f"评审意见：{critique_text}\n\n"
        f"请根据以上意见输出改进后的译文。只输出译文本身，不加任何说明。"
    )
    return call_llm(system, user, temperature=0.3)


# 快速测试：从 data/raw 读前 5 条，跑 v1 → critique → v2
if __name__ == "__main__":
    import json

    data_path = _ROOT / "data/raw/flores_en_zh.jsonl"
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            samples.append(json.loads(line))

    for s in samples:
        v1 = translate_v1(s["en"])
        crit = critique(s["en"], v1)
        v2 = revise(s["en"], v1, crit)
        print("--- Sample", s["id"], "---")
        print("EN:      ", s["en"][:80])
        print("V1:      ", v1[:80])
        print("Critique:", crit[:120])
        print("V2:      ", v2[:80])
        print()
