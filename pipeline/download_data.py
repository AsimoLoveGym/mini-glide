from pathlib import Path

from datasets import load_dataset

# Hub 上不存在 facebook/flores-200；facebook/flores 在 datasets>=3 下会因 flores.py 报错。
# 使用 pipeline/requirements.txt 中的 datasets<3，并保留 trust_remote_code（脚本数据集需要）。
ds = load_dataset(
    "facebook/flores",
    "eng_Latn-zho_Hans",
    split="devtest",
    trust_remote_code=True,
)

PREVIEW_N = 5
_HERE = Path(__file__).resolve().parent
PREVIEW_PATH = _HERE / "flores_eng_zho_devtest_preview.jsonl"

preview = ds.select(range(min(PREVIEW_N, len(ds))))
preview.to_json(str(PREVIEW_PATH))

print(f"共 {len(ds)} 条（devtest）。预览前 {len(preview)} 条已写入：\n  {PREVIEW_PATH}\n")
for i in range(len(preview)):
    eng = preview[i]["sentence_eng_Latn"]
    zho = preview[i]["sentence_zho_Hans"]
    eng_short = (eng[:120] + "…") if len(eng) > 120 else eng
    zho_short = (zho[:120] + "…") if len(zho) > 120 else zho
    print(f"--- [{i}] ---")
    print(f"EN: {eng_short}")
    print(f"ZH: {zho_short}\n")

# 每条数据：
# ds[0]['sentence_eng_Latn']  → 英文原文（给 API 翻译）
# ds[0]['sentence_zho_Hans']  → 中文参考译文（留着对比用）
