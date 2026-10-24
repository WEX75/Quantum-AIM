"""Deterministic shot tax from the measured effective compression c(p) (exp_noise_shots, exact expectation values).

A score is estimated from S shots with standard error ~ 1/sqrt(S), while noise compresses the signal |s - 1/2| by c(p).
To keep the same signal-to-noise ratio as without noise, a model needs 1/c^2 times more shots.
  MOCO      c = lambda^2 exactly (product state, k = 2 gates per qubit), so the tax is lambda^-4 (asserted below).
  baseline  1/c^2 is only a LOWER bound on its cost: its compression is not uniform over transactions, and it loses
            ranking even with infinite shots (Spearman rho 0.686 at p = 0.05), which no number of shots recovers.
Diagnostic, computed from test scores, not used for any choice.

    python3 -m qai.experiments.shot_tax          # reads results/task2/noise_shots.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/task2")
MODELS = {"MOCO": "#2a78d6", "R2 (moco)": "#1baf7a", "baseline VQC": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def lam(p):
    return 1 - 4 * p / 3


def shot_tax_table(noise_shots):
    exact = noise_shots[np.isinf(noise_shots.shots)]
    with np.errstate(divide="ignore"):
        exact = exact.assign(tax=1 / exact.compression ** 2)
    g = exact.groupby(["model", "p"])
    c_mean = g.compression.mean()
    with np.errstate(divide="ignore"):
        tax = 1 / c_mean ** 2                      # from the mean compression over seeds
    table = pd.DataFrame({"compression_mean": c_mean, "compression_std": g.compression.std(), "shot_tax": tax,
                          "shot_tax_seed_min": g.tax.min(), "shot_tax_seed_max": g.tax.max()}).reset_index()
    table["note"] = np.where(table.model == "MOCO", "exact: lambda^-4",
                             np.where(table.model == "baseline VQC", "lower bound (non-uniform compression, ranking lost)",
                                      "1/c^2 from median compression"))
    moco = exact[exact.model == "MOCO"]
    err = np.abs(moco.tax - lam(moco.p) ** -4).max()
    assert err < 1e-6, f"MOCO shot tax differs from lambda^-4 by {err}"
    return table, err


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    levels = sorted(table.p.unique())
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ends = {}
    for model, color in MODELS.items():
        t = table[table.model == model].set_index("p").reindex(levels)
        y = t.shot_tax.to_numpy()
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=color, linewidth=2, marker="o", markersize=5)
        ends[model] = (x[ok][-1], y[ok][-1])
        if not ok.all():
            ax.annotate(f"at p = {levels[np.flatnonzero(~ok)[0]]:g}: infinite (compression exactly 0)",
                        (x[ok][-1], y[ok][-1]), xytext=(6, -12),
                        textcoords="offset points", fontsize=8, color=MUTED)
    for model, (xe, ye) in ends.items():
        label = {"MOCO": "MOCO (= λ⁻⁴)", "baseline VQC": "baseline VQC (lower bound)"}.get(model, model)
        ax.annotate(label, (xe, ye), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{p:g}" for p in levels])
    ax.set_xlabel("depolarizing p per gate (ordinal axis, not to scale)")
    ax.set_ylabel("shot tax 1/c² (× shots needed for the same signal-to-noise)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.3, len(levels) + 1.2)
    ax.set_title("Shot tax from the measured score compression c(p), mean over 5 seeds\n"
                 "diagnostic, measured on test, not used for any choice", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "shot_tax.png", dpi=160)
    plt.close(fig)


def main():
    table, err = shot_tax_table(pd.read_csv(OUT / "noise_shots.csv"))
    table.to_csv(OUT / "shot_tax.csv", index=False)
    plot(table)
    pd.set_option("display.width", 200)
    print(table[table.model.isin(MODELS)].pivot(index="p", columns="model", values="shot_tax").to_string(
        float_format=lambda v: f"{v:.4g}"))
    print(f"\nMOCO: max |1/c^2 - lambda^-4| = {err:.2e} (assert < 1e-6 passed)")


if __name__ == "__main__":
    main()
