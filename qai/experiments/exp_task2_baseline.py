"""Task 2 requirement 1 + 3: the kit VQC and a logistic regression on the same four angles, 5 seeds, under the
official protocol (common split, depolarizing noise after every native gate, max-F1 threshold on validation).

    python3 -m qai.experiments.exp_task2_baseline            # full data, ~30-40 min, background
    python3 -m qai.experiments.exp_task2_baseline --fast     # 10% stratified sample, 3 epochs, smoke run

Scores are cached per seed in results/task2/ so later experiments (shots, CV) reuse them.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from qai import kit_vqc as kit
from qai.protocol import (SEEDS, AngleEncoder, circuit_scorer, common_split, gradient_ledger, load_df, load_scores,
                          mean_std, protocol_table, save_scores, scores_under_noise, tracked_executions_per_example,
                          training_examples, undersample, with_tracker)

OUT = Path("results/task2")


def run_seed(seed, fast=False):
    torch.set_num_threads(3)
    cache = OUT / f"baseline_seed{seed}{'_fast' if fast else ''}.npz"
    split = common_split(load_df(sample_fraction=0.1 if fast else 1.0))
    X_bal, y_bal = undersample(split.X_train, split.y_train, seed)
    enc = AngleEncoder(kit.N_QUBITS, seed).fit(X_bal)
    Z_bal, Z_val, Z_test = enc.transform(X_bal), enc.transform(split.X_val), enc.transform(split.X_test)

    if cache.exists():
        vqc_scores = load_scores(cache)
    else:
        weights, _ = kit.train_vqc(Z_bal, y_bal, Z_val, split.y_val, seed, epochs=3 if fast else kit.EPOCHS)
        vqc_scores = scores_under_noise(circuit_scorer(kit.vqc, weights), Z_val, Z_test)
        save_scores(cache, vqc_scores, weights=weights.numpy())

    ref = LogisticRegression(max_iter=1000).fit(Z_bal, y_bal)
    r_val, r_test = ref.predict_proba(Z_val)[:, 1], ref.predict_proba(Z_test)[:, 1]
    lr_scores = {p: (r_val, r_test) for p in vqc_scores}       # classical: noise does not apply

    tables = []
    for model, scores in [("baseline VQC", vqc_scores), ("logistic regression", lr_scores)]:
        t = protocol_table(scores, split.y_val, split.y_test)
        t["model"], t["seed"] = model, seed
        tables.append(t)
    return pd.concat(tables), len(y_bal), len(split.y_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(args.seeds)) as pool:
        results = pool.starmap(run_seed, [(s, args.fast) for s in args.seeds])
    per_seed = pd.concat([r[0] for r in results])
    suffix = "_fast" if args.fast else ""
    per_seed.to_csv(OUT / f"baseline_per_seed{suffix}.csv")

    pd.set_option("display.width", 220, "display.max_columns", None)
    for model, t in per_seed.groupby("model"):
        print(f"\n=== {model} (mean, std over {len(args.seeds)} seeds)")
        print(mean_std([t]).round(3))
    n_train, n_val = results[0][1], results[0][2]
    Z5, y5 = training_examples(seed=args.seeds[0])
    tracker = tracked_executions_per_example(kit.vqc, kit.init_weights(0), Z5, y5)     # 41: dead params skipped
    led = pd.DataFrame([with_tracker(gradient_ledger("baseline VQC", kit.EPOCHS, n_train, 32, n_val), tracker,
                                     kit.EPOCHS * n_train)])
    led.to_csv(OUT / f"ledger_baseline{suffix}.csv", index=False)
    print("\n", led.to_string(index=False))


if __name__ == "__main__":
    main()
