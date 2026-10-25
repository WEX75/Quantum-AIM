"""Margins of MOCO (kit encoding) on the full validation set, from the cached circuits (no new circuit simulation).

For every seed: median |<H>_0 - t_h| over validation transactions (t_h = 1 - 2 t0, t0 the max-F1 threshold on the
noiseless validation scores), the 5th and 95th percentiles of <H>_0 over legitimate transactions, and the median
per-shot standard deviation of h = sum_q w_q z_q at p = 0. Explains why fixed-width intervals need many shots.

    python3 -m qai.experiments.exp_margins       # -> results/task2/margins.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qai.experiments.exp_adaptive_shots import SEEDS, SeedData, problem

OUT = Path("results/task2/margins.csv")


def main():
    rows = []
    for seed in SEEDS:
        d = SeedData(seed, smoke=False)
        P, w, exact, t = problem(d, "MOCO kit", 0.0, "val")
        mu, t_h = 1 - 2 * exact, 1 - 2 * t
        legit = mu[d.y_val == 0]
        sd = np.sqrt(((1 - (2 * P - 1) ** 2) * w ** 2).sum(1))
        rows.append({"seed": seed, "t_h": t_h, "legit <H> p5": np.quantile(legit, 0.05),
                     "legit <H> p95": np.quantile(legit, 0.95), "median margin |<H> - t_h|": np.median(np.abs(mu - t_h)),
                     "median per-shot sd": np.median(sd)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))
    print(df.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
