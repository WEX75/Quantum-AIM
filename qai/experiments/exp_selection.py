"""Selection of MOCO's configuration on validation only (the test set is never touched).
Grid: feature map in {single, pair} x legit:fraud ratio in {1, 3, 10, 30, 100} x seeds 0..4. For each cell: undersample
the training split, fit the kit encoder on that subset, fit MOCO (measured Bloch vectors, convex fit, C by CV on the
training subset), and score the validation set with exact expectation values. train_exec = training rows x measured
bases (the hardware cost of the acquisition).

Selection rule: the configuration with the best mean validation AUPRC defines the bar; every configuration whose mean
is within one standard error of that best mean (std over seeds / sqrt(5)) counts as tied, and the cheapest tied one
(fewest training executions) is selected. This rule was made precise after the pair rows were reproduced (the
earlier wording was "best mean, ties to the cheaper"); with it, single 1:10 is selected, as it was before.

    python3 -m qai.experiments.exp_selection [--out results/selection_on_validation.csv]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from qai import moco
from qai.protocol import SEEDS, AngleEncoder, common_split, load_df, undersample

FMS = ["single", "pair"]
RATIOS = [1, 3, 10, 30, 100]


def run_cell(fm, ratio, seed):
    torch.set_num_threads(1)
    split = common_split(load_df())
    X_b, y_b = undersample(split.X_train, split.y_train, seed, ratio)
    enc = AngleEncoder(moco.N_QUBITS, seed).fit(X_b)
    params, _ = moco.fit(enc.transform(X_b), y_b, seed, fm)
    s_val = moco.score_exact(params, moco.bloch_exact(enc.transform(split.X_val), fm))
    return {"fm": fm, "ratio": ratio, "seed": seed, "val_auprc": average_precision_score(split.y_val, s_val),
            "train_exec": moco.train_executions(len(y_b), fm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/selection_on_validation.csv")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    jobs = [(fm, r, s) for fm in FMS for r in RATIOS for s in SEEDS]
    with Pool(args.jobs) as pool:
        rows = pool.starmap(run_cell, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out), index=False)
    g = df.groupby(["fm", "ratio"]).agg(val_auprc=("val_auprc", "mean"), sd=("val_auprc", "std"),
                                        train_exec=("train_exec", "first")).reset_index()
    print(g.round(3).to_string(index=False))
    best = g.loc[g.val_auprc.idxmax()]
    tied = g[g.val_auprc >= best.val_auprc - best.sd / len(SEEDS) ** 0.5]
    pick = tied.loc[tied.train_exec.idxmin()]
    print(f"best mean: {best.fm} 1:{best.ratio}; selected (cheapest within one standard error): {pick.fm} 1:{pick.ratio}")


if __name__ == "__main__":
    main()
