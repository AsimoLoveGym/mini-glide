# pipeline/download_data.py
import json
import os

from datasets import load_dataset


def download_flores(n=200, output_path="data/raw/flores_en_zh.jsonl"):
    """
    下载 FLORES 英→中（Hub: facebook/flores, eng_Latn-zho_Hans）devtest 的前 n 条。
    每条格式：{id, en, zh_ref}
    zh_ref 是人工参考译文，用于最后对比评估

    需 datasets<3 且 trust_remote_code（脚本数据集）。完整依赖见根目录 requirements.txt。
    """
    print("Downloading FLORES-200...")
    ds = load_dataset(
        "facebook/flores",
        "eng_Latn-zho_Hans",
        split="devtest",
        trust_remote_code=True,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        samples.append(
            {
                "id": i,
                "en": row["sentence_eng_Latn"],
                "zh_ref": row["sentence_zho_Hans"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Example: {samples[0]}")
    return samples


if __name__ == "__main__":
    download_flores()
