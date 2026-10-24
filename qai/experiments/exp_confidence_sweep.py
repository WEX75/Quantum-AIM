"""Confidence sweep of the adaptive triage, VALIDATION only (the selected alpha = 0.05 is not changed).

MOCO kit encoding, plug-in Bernstein interval, b = 16, S_max = 4096, p in {0.01, 0.05},
alpha in {0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49} (confidence 0.999 down to 0.51), 5 seeds, full validation set.
Same random numbers for every alpha (common random numbers), and the same generator as exp_adaptive_shots, so the
alpha = 0.05 rows must equal its cached validation rows (checked).

Hypotheses (written before the run):
  C1  Mean shots fall steadily as the confidence drops (the interval width grows like sqrt(log(2 / alpha))), by
      about 2x from 0.999 to 0.51.
  C2  F1 and recall stay within 0.01 of the alpha = 0.05 value down to confidence 0.9, and drop slightly below.
  C3  The empirical rate of wrong early decisions stays <= alpha at every alpha (by construction), and far below it.

    python3 -m qai.experiments.exp_confidence_sweep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qai import adaptive as ad
from qai.experiments.exp_adaptive_shots import OUT, P_LEVELS, SEEDS, SeedData, problem, run_triage

MODEL, CI, B, CAP = "MOCO kit", "plugin-bernstein", 16, 4096
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49]
P_SWEEP = [0.01, 0.05]
SELECTED_ALPHA = 0.05
FIG = OUT / "figures" / "l_confidence_sweep.png"


def sweep_seed(seed):
    d = SeedData(seed, smoke=False)
    rows = []
    for p in P_SWEEP:
        i = P_LEVELS.index(p)
        P, w, exact, t = problem(d, MODEL, p, "val")
        for alpha in ALPHAS:
            rng = np.random.default_rng([seed, i, CAP, 7])          # the generator of exp_adaptive_shots, stage 2b
            r, row = run_triage(P, w, t, (CI, B, alpha), [CAP], rng, d.y_val, exact)[CAP]
            flip = r["fraud"] != (exact >= t)
            rows.append(dict(seed=seed, p=p, alpha=alpha, confidence=1 - alpha, **row,
                             **{"wrong early decisions among stopped": float(np.mean(flip[r["decided"]]))}))
    return pd.DataFrame(rows)


def check_against_cache(df):
    """alpha = 0.05 must reproduce exp_adaptive_shots' cached validation rows (same generator, same rows)."""
    err = 0.0
    for (seed, p), g in df[df.alpha == SELECTED_ALPHA].groupby(["seed", "p"]):
        cached = pd.read_csv(OUT / f"adaptive_val_chosen_seed{seed}_p{p}.csv")
        c = cached[(cached.model == MODEL) & (cached.cap == CAP)].iloc[0]
        assert (c.ci, int(c.b), float(c.alpha)) == (CI, B, SELECTED_ALPHA)
        err = max(err, *(abs(float(g[k].iloc[0]) - float(c[k])) for k in ("mean shots", "F1", "recall")))
    return err


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    shots_color, f1_color = "#2a78d6", "#eb6834"                   # validated pair (dataviz reference palette)
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "figure.facecolor": surface,
                         "axes.facecolor": surface, "savefig.facecolor": surface})
    agg = df.groupby(["p", "alpha"])[["mean shots", "F1"]].agg(["mean", "std"])
    x = np.log10(ALPHAS)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax2 = ax.twinx()
    for p, style in zip(P_SWEEP, ("--", "-")):
        a = agg.loc[p]
        ax.errorbar(x, a[("mean shots", "mean")], yerr=a[("mean shots", "std")], color=shots_color, linestyle=style,
                    linewidth=2, marker="o", markersize=5, label=f"mean shots per transaction, p = {p:g}")
        ax2.errorbar(x, a[("F1", "mean")], yerr=a[("F1", "std")], color=f1_color, linestyle=style, linewidth=2,
                     marker="s", markersize=5, label=f"F1, p = {p:g}")
        ax.annotate(f"{a[('mean shots', 'mean')].iloc[-1]:.0f} shots", (x[-1], a[("mean shots", "mean")].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=ink)
    ax.axvline(np.log10(SELECTED_ALPHA), color=muted, linewidth=1.2, linestyle=":")
    ax.annotate("selected: confidence 0.95", (np.log10(SELECTED_ALPHA), 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8, color=muted)
    ax.set_xticks(x, [f"{1 - a:g}" for a in ALPHAS])
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.45)
    ax.set_xlabel("confidence 1 - alpha (decreasing to the right; spacing logarithmic in alpha)")
    ax.set_ylabel("mean shots per transaction (blue)")
    ax2.set_ylabel("validation F1 (orange)")
    ax.set_ylim(0, None)
    ax2.set_ylim(0.6, 0.85)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for a_ in (ax, ax2):
        a_.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Adaptive triage, MOCO kit, plug-in Bernstein, b = 16, S_max = 4096: confidence sweep\n"
                 "validation only, mean ± std over 5 seeds; the selected alpha = 0.05 is unchanged", fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


def main():
    df = pd.concat([sweep_seed(s) for s in SEEDS])
    df.to_csv(OUT / "confidence_sweep.csv", index=False)
    err = check_against_cache(df)
    plot(df)
    cols = ["mean shots", "F1", "recall", "capped", "wrong early decisions", "wrong early decisions among stopped"]
    g = df.groupby(["p", "confidence"], sort=False)[cols]
    pd.set_option("display.width", 220, "display.max_columns", None)
    print((g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).to_string())
    worst = (df["wrong early decisions"] - df.alpha).max()
    print(f"\nmax over seeds/p/alpha of (wrong early decisions - alpha): {worst:.4f}  ->  "
          f"{'PASS' if worst <= 0 else 'FAIL'} (must be <= 0)")
    print(f"alpha = 0.05 vs cached exp_adaptive_shots validation rows: max |difference| = {err:.2e}")


if __name__ == "__main__":
    main()
