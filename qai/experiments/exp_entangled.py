"""R2: a shallow entangling VQC trained variationally, warm-started from MOCO.

Circuit: RY(x_q) -> Rot(alpha_q) -> CNOT(0,1) CNOT(1,2) CNOT(2,3) -> Rot(beta_q) -> readout
p_fraud = (1 - sum_q w_q <Z_q>) / 2 with w = softmax(v) (a classical rescaling of the circuit's outputs).
With alpha = 0 this is exactly MOCO with the "linear_cnot" feature map, so MOCO gives the global optimum of the
last layer in closed form, and Adam then trains the pre-entangling rotations from there instead of from a random
point (where gradients of deeper circuits vanish [13, 14]).

Variants, each 5 seeds under the official protocol:
  random       kit-style uniform init, trained without noise
  moco         MOCO warm start, trained without noise
  moco-noisy   MOCO warm start, trained inside the noise model at P_TRAIN (noise-aware training [16])

    python3 -m qai.experiments.exp_entangled [--variants moco moco-noisy] [--fast]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.model_selection import train_test_split

from qai import kit_vqc as kit
from qai import moco
from qai.protocol import (SEED, SEEDS, AngleEncoder, circuit_scorer, common_split, gate_stats, gradient_ledger,
                          ledger, load_df, load_scores, mean_std, noisy_qnode, protocol_table, save_scores,
                          scores_under_noise, undersample)

OUT = Path("results/task2")
N = 4
N_ROT = 2 * N * 3            # trainable rotation angles (alpha, beta)
EPOCHS = 10
P_TRAIN = 0.05
VAL_FRACTION = 0.2           # validate on a stratified 20% subset (cheaper than the kit's full validation)


def circuit(x, theta):
    rot = theta[:N_ROT].reshape(2, N, 3)
    for q in range(N):
        qml.RY(x[..., q], wires=q)
    for q in range(N):
        qml.Rot(*rot[0, q], wires=q)
    for q in range(N - 1):
        qml.CNOT(wires=[q, q + 1])
    for q in range(N):
        qml.Rot(*rot[1, q], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(N)]


def readout_for(theta):
    w = torch.softmax(torch.as_tensor(theta[N_ROT:]), 0)
    return lambda expvals: (1 - torch.stack(list(expvals), -1) @ w) / 2


def model_for(qnode):
    def model(x, theta):
        return (1 - torch.stack(list(qnode(x, theta)), -1) @ torch.softmax(theta[N_ROT:], 0)) / 2
    return model


def moco_init(Z_m, y_m, seed):
    params, _ = moco.fit(Z_m, y_m, seed, "linear_cnot")
    theta = np.zeros(N_ROT + N)
    theta[N * 3:N_ROT] = params["rot"].ravel()             # beta = MOCO's compiled rotations, alpha = identity
    theta[N_ROT:] = np.log(np.clip(params["w"], 1e-12, None))
    return lambda seed: torch.nn.Parameter(torch.tensor(theta))


def random_init(seed):
    g = torch.Generator().manual_seed(seed)
    theta = torch.zeros(N_ROT + N, dtype=torch.float64)
    theta[:N_ROT] = 2 * np.pi * torch.rand(N_ROT, generator=g, dtype=torch.float64)
    return torch.nn.Parameter(theta)


def run_seed(seed, variant, fast=False):
    torch.set_num_threads(3)
    tag = f"entangled_{variant}_seed{seed}{'_fast' if fast else ''}"
    split = common_split(load_df(sample_fraction=0.1 if fast else 1.0))
    X_m, y_m = undersample(split.X_train, split.y_train, seed, moco.NEG_PER_POS)   # warm start data (1:10)
    X_b, y_b = undersample(split.X_train, split.y_train, seed)                      # variational stage (1:1)
    enc = AngleEncoder(N, seed).fit(X_m)
    Z_m, Z_b, Z_val, Z_test = (enc.transform(X) for X in (X_m, X_b, split.X_val, split.X_test))
    Z_vs, _, y_vs, _ = train_test_split(Z_val, split.y_val, train_size=VAL_FRACTION, stratify=split.y_val,
                                        random_state=SEED)

    cache = OUT / f"{tag}.npz"
    if cache.exists():
        data = np.load(cache)
        scores, theta = load_scores(cache), torch.tensor(data["theta"])
    else:
        init = random_init if variant == "random" else moco_init(Z_m, y_m, seed)
        qnode = (noisy_qnode(circuit, P_TRAIN, N) if variant == "moco-noisy" else
                 qml.QNode(circuit, qml.device("default.qubit", wires=N), interface="torch", diff_method="backprop"))
        theta, _ = kit.train_vqc(Z_b, y_b, Z_vs, y_vs, seed, model=model_for(qnode), init=init,
                                 epochs=2 if fast else EPOCHS)
        scores = scores_under_noise(circuit_scorer(circuit, theta, readout_for(theta.numpy()), N, chunk=20000),
                                    Z_val, Z_test)
        save_scores(cache, scores, theta=theta.numpy())
    t = protocol_table(scores, split.y_val, split.y_test)
    t["model"], t["seed"] = f"R2 entangled ({variant})", seed
    return t, len(y_m), len(y_b), len(y_vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["random", "moco", "moco-noisy"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, v, args.fast) for v in args.variants for s in args.seeds]
    with Pool(min(len(jobs), 5)) as pool:
        results = pool.starmap(run_seed, jobs)
    per_seed = pd.concat([r[0] for r in results])
    suffix = "_fast" if args.fast else ""
    per_seed.to_csv(OUT / f"entangled_per_seed{suffix}.csv")
    pd.set_option("display.width", 220, "display.max_columns", None)
    for model, t in per_seed.groupby("model"):
        print(f"\n=== {model}")
        print(mean_std([t]).round(3))

    theta = torch.zeros(N_ROT + N, dtype=torch.float64)
    print("\ncircuit:", gate_stats(circuit, np.zeros(N), theta))
    n_m, n_b, n_vs = results[0][1], results[0][2], results[0][3]
    epochs = 2 if args.fast else EPOCHS
    rows = []
    for v in args.variants:
        row = gradient_ledger(f"R2 entangled ({v})", epochs, n_b, N_ROT, n_vs)
        if v != "random":   # + the MOCO warm start: X, Y, Z Bloch vectors of the 1:10 set, measured once
            row = ledger(row["method"], row["train executions"] + moco.train_executions(n_m, "linear_cnot"),
                         row["validation executions"], executions_per_example=row["executions_per_example"])
        rows.append(row)
    led = pd.DataFrame(rows)
    led.to_csv(OUT / f"ledger_entangled{suffix}.csv", index=False)
    print("\n", led.to_string(index=False))


if __name__ == "__main__":
    main()
