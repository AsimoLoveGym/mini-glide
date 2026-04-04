# pipeline/evaluate.py
import json
import re
import time
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=True)
client = OpenAI()


def judge_pairwise(en: str, zh_a: str, zh_b: str) -> dict:
    """
    让模型比较两个译文（A=v1, B=v2），输出各维度分数和胜者。
    返回 dict，失败时返回 {'error': ...}
    """
    prompt = f"""你是一名专业翻译质量评审员。请评判以下两个中文译文哪个更好。

原文（英文）：{en}

译文A：{zh_a}

译文B：{zh_b}

请从以下三个维度分别给 A 和 B 打分（1–5分）：
- 准确性：译文是否忠实于原文意思
- 流畅性：译文读起来是否自然流畅
- 风格：是否适合短视频平台（口语化、简洁）

最后给出总体判断。

严格按以下 JSON 格式输出，不要加其他任何内容：
{{
  "accuracy_a": <1-5>,
  "fluency_a": <1-5>,
  "style_a": <1-5>,
  "accuracy_b": <1-5>,
  "fluency_b": <1-5>,
  "style_b": <1-5>,
  "winner": "A" | "B" | "tie",
  "reason": "一句话理由"
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def run_judge(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    n: int = 200,
) -> None:
    input_path = Path(input_path) if input_path is not None else _ROOT / "data/generated/pipeline_output.jsonl"
    output_path = Path(output_path) if output_path is not None else _ROOT / "data/eval/judge_results.jsonl"

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                data.append(r)
    data = data[:n]
    print(f"Judging {len(data)} samples...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for item in tqdm(data, desc="Judging"):
        result = judge_pairwise(item["en"], item["zh_v1"], item["zh_v2"])
        result["id"] = item["id"]
        results.append(result)
        time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = [r for r in results if "error" not in r]
    print(f"Done: {len(valid)}/{len(results)} valid judge results")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    run_judge()
