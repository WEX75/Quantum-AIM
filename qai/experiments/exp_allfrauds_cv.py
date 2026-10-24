"""All 492 frauds as test points, without leakage: an outer stratified 5-fold split of the full dataset. In each
fold the other 80% is split 75/25 into train and validation (the same 60/20/20 proportions as the common split),
and every model is trained and evaluated with the official protocol. The official test set has only 98 frauds,
so this tells whether differences of a few AUPRC points (MOCO vs logistic regression, noisy vs noiseless) are real.

    python3 -m qai.experiments.exp_allfrauds_cv            # ~ as long as the 5-seed baseline, background
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from qai import kit_vqc as kit
from qai import moco
from qai.protocol import (NOISE_LEVELS, SEED, AngleEncoder, circuit_scorer, load_df, load_scores, mean_std,
                          protocol_table, save_scores, scores_under_noise, undersample)

OUT = Path("results/task2/cv")
LEVELS = [*NOISE_LEVELS, 0.1]


def fold_split(df, fold):
    X, y = df.drop(columns="Class"), df["Class"].to_numpy()
    trval_idx, test_idx = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))[fold]
    X_tv, y_tv = X.iloc[trval_idx], y[trval_idx]
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED)
    return X_tr, y_tr, X_val, y_val, X.iloc[test_idx], y[test_idx]


def run_fold(fold):
    torch.set_num_threads(3)
    X_tr, y_tr, X_val, y_val, X_te, y_te = fold_split(load_df(), fold)
    seed, scores = fold, {}

    X_b, y_b = undersample(X_tr, y_tr, seed)                                   # baseline: 1:1, kit encoder
    enc = AngleEncoder(kit.N_QUBITS, seed).fit(X_b)
    Z_b, Z_val, Z_te = enc.transform(X_b), enc.transform(X_val), enc.transform(X_te)
    cache = OUT / f"baseline_fold{fold}.npz"
    if not cache.exists():
        weights, _ = kit.train_vqc(Z_b, y_b, Z_val, y_val, seed)
        save_scores(cache, scores_under_noise(circuit_scorer(kit.vqc, weights), Z_val, Z_te, LEVELS))
    scores["baseline VQC"] = load_scores(cache)
    lr = LogisticRegression(max_iter=1000).fit(Z_b, y_b)
    scores["logistic regression 1:1"] = {p: (lr.predict_proba(Z_val)[:, 1], lr.predict_proba(Z_te)[:, 1])
                                         for p in [0.0, *LEVELS]}

    X_m, y_m = undersample(X_tr, y_tr, seed, moco.NEG_PER_POS)                # MOCO: 1:10, same encoder recipe
    enc = AngleEncoder(moco.N_QUBITS, seed).fit(X_m)
    Z_m, Z_val, Z_te = enc.transform(X_m), enc.transform(X_val), enc.transform(X_te)
    cache = OUT / f"moco_fold{fold}.npz"
    if not cache.exists():
        params, _ = moco.fit(Z_m, y_m, seed)
        save_scores(cache, scores_under_noise(circuit_scorer(moco.make_circuit(params), None, chunk=20000),
                                              Z_val, Z_te, LEVELS))
    scores["MOCO"] = load_scores(cache)
    lr = LogisticRegression(max_iter=1000).fit(Z_m, y_m)
    scores["logistic regression 1:10"] = {p: (lr.predict_proba(Z_val)[:, 1], lr.predict_proba(Z_te)[:, 1])
                                          for p in [0.0, *LEVELS]}

    tables = []
    for model, s in scores.items():
        t = protocol_table(s, y_val, y_te)
        t["model"], t["fold"] = model, fold
        tables.append(t)
    test_scores = {model: {p: s[p][1] for p in s} for model, s in scores.items()}
    return pd.concat(tables), test_scores, y_te


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(5) as pool:
        results = pool.map(run_fold, range(5))
    per_fold = pd.concat([r[0] for r in results])
    per_fold.to_csv(OUT / "cv_per_fold.csv")
    pd.set_option("display.width", 220, "display.max_columns", None)
    for model, t in per_fold.groupby("model"):
        print(f"\n=== {model} (mean, std over 5 folds; every fraud tested once)")
        print(mean_std([t]).round(3))

    y_all = np.concatenate([r[2] for r in results])
    pooled = {model: {p: average_precision_score(y_all, np.concatenate([r[1][model][p] for r in results]))
                      for p in [0.0, *LEVELS]} for model in results[0][1]}
    pooled = pd.DataFrame(pooled)
    pooled.index.name = "p"
    pooled.to_csv(OUT / "cv_pooled_auprc.csv")
    print(f"\n=== pooled test AUPRC over all {int(y_all.sum())} frauds (scores of 5 fold models concatenated)\n",
          pooled.round(3))


if __name__ == "__main__":
    main()
