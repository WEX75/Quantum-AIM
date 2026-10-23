"""End-to-end Quantum AI experiment. Writes results/qai_<dataset>.json and a plot.

Experiments:
  1. classical baselines (logreg, small MLP, RBF-SVM) with parameter counts
  2. VQC trained noiseless, evaluated across noise levels  -> how fragile is it?
  3. VQC trained *under* noise p_train (noise-aware)        -> does it adapt?
  4. noise-scaling extrapolation (ZNE) on model 2           -> how much can we recover?

    python -m qai.bench --fast
    python -m qai.bench --dataset file:data/kit.csv --qubits 6 --layers 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import run_baselines
from .data import load_dataset
from .mitigation import zne_accuracy
from .model import VQC, VQCConfig
from .noise import NoiseModel
from .train import train

RESULTS = Path("results")


def run(dataset="moons", qubits=4, layers=3, epochs=30, lr=0.05, encoding="reupload",
        noise_levels=(0.0, 0.01, 0.02, 0.05, 0.1), p_train=0.05, max_samples=300,
        seed=0, zne=True, verbose=False) -> dict:
    split = load_dataset(dataset, n_features=qubits, max_samples=max_samples, seed=seed)
    cfg = VQCConfig(n_qubits=qubits, n_layers=layers, n_classes=split.n_classes, encoding=encoding)
    res = {"config": {"dataset": dataset, "qubits": qubits, "layers": layers, "epochs": epochs,
                      "encoding": encoding, "p_train": p_train, "n_train": len(split.X_train),
                      "n_test": len(split.X_test), "n_classes": split.n_classes,
                      "vqc_params": cfg.n_params()}}
    res["classical"] = run_baselines(split, seed)

    clean = VQC(cfg, None, seed=seed)
    h_clean = train(clean, split.X_train, split.y_train, epochs, lr, seed=seed, verbose=verbose)
    aware = VQC(cfg, NoiseModel.uniform(p_train), seed=seed)
    h_aware = train(aware, split.X_train, split.y_train, epochs, lr, seed=seed, verbose=verbose)

    sweep = []
    for p in noise_levels:
        nm = NoiseModel.uniform(p)
        row = {"p": p,
               "clean_trained": clean.accuracy(split.X_test, split.y_test, nm),
               "noise_aware": aware.accuracy(split.X_test, split.y_test, nm)}
        if zne and p > 0:
            row["clean_trained+zne"] = zne_accuracy(clean, split.X_test, split.y_test, nm)
        sweep.append(row)
    res["sweep"] = sweep
    res["train_seconds"] = {"clean": h_clean["seconds"], "aware": h_aware["seconds"]}
    res["loss_curves"] = {"clean": h_clean["loss"], "aware": h_aware["loss"]}
    return res


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = [r["p"] for r in res["sweep"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, style in [("clean_trained", "o-"), ("noise_aware", "s-"), ("clean_trained+zne", "^--")]:
        pts = [(r["p"], r[key]) for r in res["sweep"] if key in r]
        if pts:
            ax[0].plot(*zip(*pts), style, label=key)
    best_cl = max(v["test_acc"] for v in res["classical"].values())
    ax[0].axhline(best_cl, color="gray", ls=":", label=f"best classical ({best_cl:.2f})")
    ax[0].set(xlabel="noise strength p", ylabel="test accuracy",
              title=f"{res['config']['dataset']}: accuracy vs noise")
    ax[0].legend(fontsize=8)
    for k, v in res["loss_curves"].items():
        ax[1].plot(v, label=k)
    ax[1].set(xlabel="epoch", ylabel="train loss", title="training")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="moons")
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--encoding", default="reupload", choices=["angle", "reupload"])
    ap.add_argument("--p-train", type=float, default=0.05)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--no-zne", action="store_true")
    ap.add_argument("--fast", action="store_true", help="tiny smoke run (~30s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(dataset=a.dataset, qubits=a.qubits, layers=a.layers, epochs=a.epochs, lr=a.lr,
              encoding=a.encoding, noise_levels=tuple(a.noise_levels), p_train=a.p_train,
              max_samples=a.max_samples, seed=a.seed, zne=not a.no_zne, verbose=a.verbose)
    if a.fast:
        kw.update(layers=2, epochs=8, max_samples=120, noise_levels=(0.0, 0.05, 0.1))
    res = run(**kw)
    RESULTS.mkdir(exist_ok=True)
    tag = a.dataset.replace("file:", "").replace("/", "_").replace(".", "_")
    out = RESULTS / f"qai_{tag}.json"
    out.write_text(json.dumps(res, indent=1))
    plot(res, out.with_suffix(".png"))
    from .summary import summarize
    print(summarize(res))
    print(f"\nsaved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
