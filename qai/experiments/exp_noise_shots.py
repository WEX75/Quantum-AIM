"""The mentor's question: what happens when the noise grows? The protocol uses exact expectation values (infinite
shots), where depolarizing noise on MOCO only contracts the scores towards 1/2 and cannot change the ranking.
On hardware every score is estimated from S shots, so the contraction has to be paid for in shots.

For each model and seed (reusing the trained circuits cached by exp_task2_baseline and exp_moco):
  - exact noisy test scores for p up to 0.5,
  - finite-shot scores for S in SHOTS (sampled exactly: binomial on qubit 0 for the baseline; for MOCO the
    final state is a product state, so each Z_q is sampled independently from its exact noisy mean),
  - test AUPRC and Spearman rho against the exact noiseless scores, and the median contraction |s_p - 1/2| / |s_0 - 1/2|.

    python3 -m qai.experiments.exp_noise_shots
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from qai import kit_vqc as kit
from qai import moco
from qai.protocol import SEEDS, AngleEncoder, clean_predict, common_split, load_df, noisy_predict, undersample

OUT = Path("results/task2")
P_LEVELS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
SHOTS = [64, 256, 1024, 4096, 16384]


def baseline_zero_probability(seed, split):
    """{p: P(qubit 0 reads 1) on the test set} for the kit VQC, cached."""
    cache = OUT / f"baseline_extra_seed{seed}.npz"
    if cache.exists():
        data = np.load(cache)
        return {p: data[str(p)] for p in P_LEVELS}
    X_bal, _ = undersample(split.X_train, split.y_train, seed)
    Z_test = AngleEncoder(kit.N_QUBITS, seed).fit(X_bal).transform(split.X_test)
    w = torch.tensor(np.load(OUT / f"baseline_seed{seed}.npz")["weights"])
    out = {p: (clean_predict(Z_test, w, kit.vqc) if p == 0 else noisy_predict(Z_test, w, p, kit.vqc))
           for p in P_LEVELS}
    np.savez(cache, **{str(p): s for p, s in out.items()})
    return out


def moco_qubit_means(seed, split):
    """Exact noiseless <Z_q> per test transaction and the observable weights; with noise each mean is
    multiplied by lambda^2 (tests/test_protocol.py checks this against the kit noise model)."""
    X_b, _ = undersample(split.X_train, split.y_train, seed, moco.NEG_PER_POS)
    Z_test = AngleEncoder(moco.N_QUBITS, seed).fit(X_b).transform(split.X_test)
    data = np.load(OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{seed}.npz")
    th, ph = -data["rot"][:, 1], -data["rot"][:, 0]
    axes = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return (moco.bloch_exact(Z_test) * axes[None]).sum(-1), data["w"]


def run_seed(seed):
    torch.set_num_threads(3)
    split = common_split(load_df())
    y = split.y_test
    rng = np.random.default_rng(1000 + seed)
    rows = []

    base = baseline_zero_probability(seed, split)
    zq, w = moco_qubit_means(seed, split)
    moco_clean = (1 - zq @ w) / 2
    for p in P_LEVELS:
        lam2 = (1 - 4 * p / 3) ** 2
        exact = {"baseline VQC": base[p], "MOCO": (1 - lam2 * zq @ w) / 2}
        clean = {"baseline VQC": base[0.0], "MOCO": moco_clean}
        for model, s in exact.items():
            contraction = np.median(np.abs(s - 0.5) / np.clip(np.abs(clean[model] - 0.5), 1e-12, None))
            rows.append(dict(model=model, seed=seed, p=p, shots=np.inf, AUPRC=average_precision_score(y, s),
                             rho=spearmanr(clean[model], s).statistic, contraction=contraction))
        for S in SHOTS:
            sampled = {
                "baseline VQC": rng.binomial(S, base[p]) / S,
                "MOCO": (1 - (2 * rng.binomial(S, (1 + lam2 * zq) / 2) / S - 1) @ w) / 2,
            }
            for model, s in sampled.items():
                rows.append(dict(model=model, seed=seed, p=p, shots=S, AUPRC=average_precision_score(y, s),
                                 rho=spearmanr(clean[model], s).statistic, contraction=np.nan))
    return pd.DataFrame(rows)


def shots_needed(df, tol=0.01):
    """Smallest S on the grid whose mean AUPRC stays within tol of the model's exact noiseless AUPRC."""
    ref = df[(df.p == 0) & np.isinf(df.shots)].groupby("model").AUPRC.mean()
    m = df[np.isfinite(df.shots)].groupby(["model", "p", "shots"]).AUPRC.mean().reset_index()
    m["ok"] = m.AUPRC >= m.model.map(ref) - tol
    need = m[m.ok].groupby(["model", "p"]).shots.min().unstack("model")
    return need.reindex(P_LEVELS)


def main():
    with Pool(len(SEEDS)) as pool:
        df = pd.concat(pool.map(run_seed, SEEDS))
    df.to_csv(OUT / "noise_shots.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", None)
    exact = df[np.isinf(df.shots)].groupby(["model", "p"])[["AUPRC", "rho", "contraction"]].agg(["mean", "std"])
    print("=== exact expectation values (protocol), extended noise range\n", exact.round(4))
    grid = df[np.isfinite(df.shots)].pivot_table(index=["model", "p"], columns="shots", values="AUPRC")
    print("\n=== test AUPRC with finite shots (mean over seeds)\n", grid.round(3))
    need = shots_needed(df)
    need.to_csv(OUT / "shots_needed.csv")
    print("\n=== shots per transaction to stay within 0.01 of the noiseless AUPRC (NaN: not reached by 16384)\n", need)


if __name__ == "__main__":
    main()
