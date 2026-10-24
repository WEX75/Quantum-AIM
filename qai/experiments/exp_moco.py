"""Task 2 requirement 2: MOCO under the official protocol, 5 seeds, next to a logistic regression trained on the
same inputs (same 1:10 undersampled set, same four angles).

    python3 -m qai.experiments.exp_moco                    # final config (single, 1:10)
    python3 -m qai.experiments.exp_moco --fm linear_cnot   # entangling ablation, same training
    python3 -m qai.experiments.exp_moco --fast             # 10% stratified sample, smoke run
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from qai import moco
from qai.protocol import (SEEDS, AngleEncoder, RobustAngleEncoder, circuit_scorer, common_split, gate_stats, ledger, load_df,
                          load_scores, mean_std, protocol_table, save_scores, scores_under_noise, training_examples,
                          undersample, with_tracker)

OUT = Path("results/task2")
SELECTION_CSV = Path("results/selection_on_validation.csv")


def run_seed(seed, fm, neg_per_pos, fast=False, q=0.0):
    torch.set_num_threads(3)
    tag = f"moco_{fm}_{neg_per_pos}{f'_q{q:g}' if q else ''}_seed{seed}{'_fast' if fast else ''}"
    split = common_split(load_df(sample_fraction=0.1 if fast else 1.0))
    X_b, y_b = undersample(split.X_train, split.y_train, seed, neg_per_pos)
    enc = (RobustAngleEncoder(moco.N_QUBITS, seed, q) if q else AngleEncoder(moco.N_QUBITS, seed)).fit(X_b)
    Z_b, Z_val, Z_test = enc.transform(X_b), enc.transform(split.X_val), enc.transform(split.X_test)

    cache = OUT / f"{tag}.npz"
    if cache.exists():
        q_scores = load_scores(cache)
    else:
        params, C = moco.fit(Z_b, y_b, seed, fm)
        q_scores = scores_under_noise(circuit_scorer(moco.make_circuit(params), None, chunk=20000), Z_val, Z_test)
        save_scores(cache, q_scores, rot=params["rot"], w=params["w"], C=C)

    ref = LogisticRegression(max_iter=1000).fit(Z_b, y_b)
    r_val, r_test = ref.predict_proba(Z_val)[:, 1], ref.predict_proba(Z_test)[:, 1]
    tables = []
    enc_tag = f" robust q={q:g}%" if q else ""
    for model, scores in [(f"MOCO {fm} 1:{neg_per_pos}{enc_tag}", q_scores),
                          (f"logistic regression 1:{neg_per_pos}{enc_tag}", {p: (r_val, r_test) for p in q_scores})]:
        t = protocol_table(scores, split.y_val, split.y_test)
        t["model"], t["seed"] = model, seed
        tables.append(t)
    return pd.concat(tables), len(y_b), len(split.y_val)


def selection_cost(n_val):
    """One-off cost of the validation study that fixed (feature map, ratio): every candidate's training set,
    plus the validation Bloch vectors measured once in X, Y, Z (all candidates are scored from them)."""
    if not SELECTION_CSV.exists():
        return None
    sel = pd.read_csv(SELECTION_CSV)
    return int(sel.groupby(["fm", "ratio"])["train_exec"].first().sum()) + 3 * n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=moco.FEATURE_MAP)
    ap.add_argument("--ratio", type=int, default=moco.NEG_PER_POS)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--q", type=float, default=0.0,
                    help="robust angle scaling on the [q, 100-q] percentiles (protocol.RobustAngleEncoder); 0 = kit")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(args.seeds)) as pool:
        results = pool.starmap(run_seed, [(s, args.fm, args.ratio, args.fast, args.q) for s in args.seeds])
    per_seed = pd.concat([r[0] for r in results])
    suffix = f"_{args.fm}_{args.ratio}" + (f"_q{args.q:g}" if args.q else "") + ("_fast" if args.fast else "")
    per_seed.to_csv(OUT / f"moco_per_seed{suffix}.csv")

    pd.set_option("display.width", 220, "display.max_columns", None)
    for model, t in per_seed.groupby("model"):
        print(f"\n=== {model} (mean, std over {len(args.seeds)} seeds)")
        print(mean_std([t]).round(3))

    params = dict(rot=np.zeros((moco.N_QUBITS, 3)), w=np.ones(moco.N_QUBITS) / moco.N_QUBITS, fm=args.fm)
    print("\ncircuit:", gate_stats(moco.make_circuit(params), np.zeros(moco.N_QUBITS), None))
    n_train, n_val = results[0][1], results[0][2]
    Z5, _ = training_examples(seed=args.seeds[0], neg_per_pos=args.ratio)
    tracker = moco.tracker_check(Z5, args.fm)["executions_per_row"]
    rows = [with_tracker(ledger(f"MOCO {args.fm} 1:{args.ratio}" + (f" robust q={args.q:g}%" if args.q else ""), moco.train_executions(n_train, args.fm), 0,
                                executions_per_example=len(moco.measured_bases(args.fm))), tracker, n_train)]
    sel = selection_cost(n_val)
    if sel is not None and not args.fast:
        rows.append(ledger("one-off selection study (fm x ratio on validation)", 0, sel))
    led = pd.DataFrame(rows)
    led.to_csv(OUT / f"ledger_moco{suffix}.csv", index=False)
    print("\n", led.to_string(index=False))


if __name__ == "__main__":
    main()
