# pipeline/vv_sampling.py
"""Simulate VV (view-count) weighted sampling vs uniform random on judge results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def simulate_vv_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    模拟视频播放量分布（长尾）：
    少量「高 VV」、大量低 VV。使用 Pareto 形状，再归一化为采样概率。
    """
    weights = rng.pareto(1.5, size=n) + 1.0
    return weights / weights.sum()


def _load_judge_records(judge_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with judge_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                records.append(r)
    return records


def _avg_scores(subset: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(subset)
    return {
        "win_rate_v2": float((df["winner"] == "B").mean()),
        "avg_accuracy_delta": float((df["accuracy_b"] - df["accuracy_a"]).mean()),
        "avg_fluency_delta": float((df["fluency_b"] - df["fluency_a"]).mean()),
        "avg_style_delta": float((df["style_b"] - df["style_a"]).mean()),
    }


def _plot_comparison(
    random_stats: dict[str, float],
    weighted_stats: dict[str, float],
    out_path: Path,
) -> None:
    metrics = list(random_stats.keys())
    x = np.arange(len(metrics))
    width = 0.35
    r_vals = [random_stats[m] for m in metrics]
    w_vals = [weighted_stats[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, r_vals, width, label="Uniform random")
    ax.bar(x + width / 2, w_vals, width, label="VV-weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Judge metrics: uniform vs VV-weighted subsample (simulated)")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_sampling_strategies(
    judge_path: str | Path | None = None,
    n_sample: int = 50,
    seed: int = 42,
    plot_path: Path | None = None,
    save_plot: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    比较随机采样 vs 模拟 VV 加权采样的 judge 汇总指标。
    若高 VV 与「revision 更有效」相关，加权子样本的 win_rate / 正向 delta 可能更高（单次抽样有随机性，宜多次重复或增大 n_sample）。
    """
    path = Path(judge_path) if judge_path is not None else _ROOT / "data/eval/judge_results.jsonl"
    records = _load_judge_records(path)
    n_total = len(records)
    if n_total < 2:
        raise ValueError(f"Need at least 2 judge rows, got {n_total} from {path}")
    if n_sample > n_total:
        raise ValueError(f"n_sample ({n_sample}) cannot exceed n_total ({n_total})")

    rng = np.random.default_rng(seed)
    weights = simulate_vv_weights(n_total, rng)

    pool = np.arange(n_total)
    random_idx = rng.choice(pool, size=n_sample, replace=False)
    weighted_idx = rng.choice(pool, size=n_sample, replace=False, p=weights)

    random_stats = _avg_scores([records[i] for i in random_idx])
    weighted_stats = _avg_scores([records[i] for i in weighted_idx])

    print("Uniform random:     ", random_stats)
    print("VV-weighted (sim): ", weighted_stats)
    print()
    print(
        "提示（非结论）：若业务上高播放量内容更依赖好翻译，加权子样本的指标可能更高；"
        "也可能因单次抽样波动而接近。可改 seed 或增大 n_sample 对照。"
    )

    if save_plot:
        out = plot_path if plot_path is not None else _ROOT / "data/eval/vv_sampling_comparison.png"
        _plot_comparison(random_stats, weighted_stats, out)
        print(f"Saved plot: {out}")

    return random_stats, weighted_stats


def main() -> None:
    p = argparse.ArgumentParser(description="VV-weighted vs random sampling on judge_results.jsonl")
    p.add_argument(
        "--judge",
        type=Path,
        default=_ROOT / "data/eval/judge_results.jsonl",
        help="Path to judge_results.jsonl",
    )
    p.add_argument("--n-sample", type=int, default=50, dest="n_sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write comparison PNG",
    )
    args = p.parse_args()
    compare_sampling_strategies(
        judge_path=args.judge,
        n_sample=args.n_sample,
        seed=args.seed,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
