# pipeline/run_pipeline.py
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm

# 保证从项目根执行 `python pipeline/run_pipeline.py` 时也能 import translate
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=True)

from translate import critique, revise, translate_v1  # noqa: E402


def process_one(sample: dict) -> dict:
    """处理单条，带错误捕获"""
    try:
        v1 = translate_v1(sample["en"])
        time.sleep(0.3)  # 避免触发限速
        crit = critique(sample["en"], v1)
        time.sleep(0.3)
        v2 = revise(sample["en"], v1, crit)
        return {**sample, "zh_v1": v1, "critique": crit, "zh_v2": v2, "status": "ok"}
    except Exception as e:
        print(f'Error on id={sample.get("id")}: {e}')
        return {**sample, "status": "error", "error": str(e)}


def run_pipeline(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    max_workers: int = 3,
) -> None:
    input_path = Path(input_path) if input_path is not None else _ROOT / "data/raw/flores_en_zh.jsonl"
    output_path = Path(output_path) if output_path is not None else _ROOT / "data/generated/pipeline_output.jsonl"

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(samples), desc="Pipeline"):
            results.append(future.result())

    results.sort(key=lambda x: x["id"])

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"Done: {ok}/{len(samples)} succeeded")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    run_pipeline()
