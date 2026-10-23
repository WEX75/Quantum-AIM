"""Hybrid torch model vs its same-size classical twin, clean vs noise-aware, across noise.

    python3 -m qai.experiments.exp_hybrid_torch --dataset creditcard --fast
    python3 -m qai.experiments.exp_hybrid_torch --dataset creditcard --qubits 4 --layers 2 --epochs 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qai.bench import load
from qai.metrics import PRIMARY
from qai.noise import NoiseModel
from qai.torch_hybrid import (ClassicalTwin, HybridNet, evaluate_torch, n_params, set_noise,
                              train_torch)


def run(dataset="creditcard", qubits=4, layers=2, epochs=20, lr=0.02, p_train=0.05,
        noise_levels=(0.0, 0.02, 0.05, 0.1), n_per_class=150, n_in=8, seed=0, data_path=None):
    # the classical "pre" layer compresses n_in features into n_qubits angles
    split, _ = load(dataset, n_in, 300, seed, n_per_class, data_path)
    metric = PRIMARY.get(dataset, PRIMARY["default"])
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test
    k = split.n_classes
    twin = ClassicalTwin(n_in, qubits, k)
    train_torch(twin, Xtr, ytr, epochs, lr, seed=seed)
    clean = HybridNet(n_in, qubits, layers, k, None)
    h_clean = train_torch(clean, Xtr, ytr, epochs, lr, seed=seed)
    aware = HybridNet(n_in, qubits, layers, k, NoiseModel.uniform(p_train))
    h_aware = train_torch(aware, Xtr, ytr, epochs, lr, seed=seed)
    twin_m = evaluate_torch(twin, Xte, yte)
    sweep = []
    for p in noise_levels:
        nm = NoiseModel.uniform(p)
        sweep.append({"p": p,
                      "hybrid_clean": evaluate_torch(set_noise(clean, nm), Xte, yte),
                      "hybrid_aware": evaluate_torch(set_noise(aware, nm), Xte, yte),
                      "classical_twin": twin_m})
    return {"config": {"dataset": dataset, "qubits": qubits, "layers": layers, "n_in": n_in,
                       "epochs": epochs, "p_train": p_train, "metric": metric,
                       "params_hybrid": n_params(clean), "params_twin": n_params(twin),
                       "n_train": len(Xtr), "n_test": len(Xte)},
            "sweep": sweep,
            "train_seconds": {"clean": h_clean["seconds"], "aware": h_aware["seconds"]}}


def summarize(res) -> str:
    c, m = res["config"], res["config"]["metric"]
    keys = ["hybrid_clean", "hybrid_aware", "classical_twin"]
    out = [f"[{c['dataset']} torch-hybrid] {c['n_in']}->{c['qubits']}q x {c['layers']}L, "
           f"params hybrid={c['params_hybrid']} twin={c['params_twin']}, metric={m}",
           "p      " + "  ".join(f"{k:>16}" for k in keys)]
    for r in res["sweep"]:
        out.append(f"{r['p']:<6} " + "  ".join(f"{r[k][m]:>16.3f}" for k in keys))
    out.append(f"train time (s): {res['train_seconds']}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="creditcard")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n-in", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--p-train", type=float, default=0.05)
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    kw = dict(dataset=a.dataset, qubits=a.qubits, layers=a.layers, epochs=a.epochs,
              p_train=a.p_train, n_per_class=a.n_per_class, n_in=a.n_in, seed=a.seed,
              data_path=a.data_path)
    if a.fast:
        kw.update(epochs=5, n_per_class=50, noise_levels=(0.0, 0.05, 0.1))
    res = run(**kw)
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"exp_hybrid_torch_{a.dataset}.json"
    out.write_text(json.dumps(res, indent=1))
    print(summarize(res))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
