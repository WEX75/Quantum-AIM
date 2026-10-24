"""Diagnostic: how many false alarms does MOCO raise to catch a given share of the frauds?
For the official MOCO models (single 1:10, seeds 0..4, scores cached by exp_moco), the maximum test precision at
recall >= 0.5 .. 0.9, and the operating point given by the protocol's threshold (max F1 on validation). MOCO's
ranking is identical at every noise level, so these numbers hold for every p of the protocol. Test set used for
reporting only; nothing is selected with it.

    python3 -m qai.experiments.exp_precision_at_recall   # -> results/task2/precision_at_recall.csv, figures/n_pr_curve.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from qai import moco
from qai.protocol import SEEDS, choose_threshold, common_split, load_df, load_scores

OUT = Path("results/task2")
RECALLS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    split = common_split(load_df())
    rows, curves = [], []
    for seed in SEEDS:
        scores = load_scores(OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{seed}.npz")
        s_val, s_test = scores[0.0]
        prec, rec, _ = precision_recall_curve(split.y_test, s_test)
        curves.append((prec, rec))
        for r in RECALLS:
            rows.append({"seed": seed, "kind": "max precision at recall", "recall target": r,
                         "precision": prec[rec >= r].max()})
        t = choose_threshold(split.y_val, s_val)
        flag = s_test >= t
        tp = int((flag & (split.y_test == 1)).sum())
        rows.append({"seed": seed, "kind": "operating point (max-F1 threshold on validation)", "recall target": np.nan,
                     "precision": tp / flag.sum(), "recall": tp / split.y_test.sum(), "flagged": int(flag.sum()),
                     "true frauds caught": tp, "false alarms": int(flag.sum()) - tp, "threshold": t})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "precision_at_recall.csv", index=False)

    summary = df[df.kind == "max precision at recall"].groupby("recall target").precision.agg(["mean", "std"])
    op = df[df.kind.str.startswith("operating")]
    print("max test precision at recall >= r (MOCO, 5 seeds):\n", summary.round(3).to_string())
    print("\noperating point:\n", op[["precision", "recall", "flagged", "true frauds caught", "false alarms"]]
          .agg(["mean", "std"]).round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for prec, rec in curves:
        ax.plot(rec, prec, color="#2a78d6", linewidth=1.5, alpha=0.6)
    ax.scatter(op.recall, op.precision, color="#0b0b0b", s=60, zorder=3)
    ax.annotate("operating points (threshold\nchosen on validation)", (op.recall.mean(), op.precision.mean()),
                xytext=(25, 25), textcoords="offset points", fontsize=14, color="#0b0b0b")
    m = summary["mean"]
    ax.scatter(m.index, m.values, marker="s", color="#eb6834", s=50, zorder=3)
    for r, v in m.items():
        ax.annotate(f"{v:.2f}", (r, v), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=13)
    ax.axhline(split.y_test.mean(), color="#898781", linestyle=":", linewidth=1.5)
    ax.annotate(f"chance level ({split.y_test.mean():.4f})", (0.02, split.y_test.mean()), xytext=(0, 6),
                textcoords="offset points", fontsize=13, color="#898781")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (share of test frauds caught)")
    ax.set_ylabel("precision (share of alerts that are fraud)")
    ax.set_title(f"At 80% recall MOCO keeps {m[0.8]:.0%} precision (orange: mean max precision, 5 seeds)",
                 loc="left", fontsize=16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "n_pr_curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
