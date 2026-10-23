"""End-to-end Quantum AI experiment. Writes results/qai_<dataset>.json and a plot.

Experiments:
  1. classical baselines (logreg, MLP, RBF-SVM, RF; + logreg on the full train set for fraud)
  2. VQC trained noiseless, evaluated across noise levels  -> how fragile is it?
  3. VQC trained *under* noise p_train (noise-aware)        -> does it adapt?
  4. noise-scaling extrapolation (ZNE) on model 2           -> how much can we recover?
Every model is scored with acc, bal_acc, precision, recall, f1, roc_auc, pr_auc; the summary
shows the primary metric (pr_auc for fraud, acc otherwise).

    python3 -m qai.bench --fast
    python3 -m qai.bench --dataset creditcard --qubits 4 --layers 3      # data/creditcard.csv
    python3 -m qai.bench --dataset file:data/kit.csv --qubits 6 --layers 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import run_baselines
from .data import load_dataset
from .metrics import PRIMARY
from .mitigation import zne_metrics
from .model import VQC, VQCConfig
from .noise import NoiseModel
from .train import train

RESULTS = Path("results")


def load(dataset, qubits, max_samples, seed, n_per_class, data_path):
    if dataset == "creditcard":
        from .fraud import DEFAULT_PATH, load_creditcard
        return load_creditcard(data_path or DEFAULT_PATH, n_features=qubits,
                               n_per_class=n_per_class, seed=seed, return_full_train=True)
    return load_dataset(dataset, n_features=qubits, max_samples=max_samples, seed=seed), None


def run(dataset="moons", qubits=4, layers=3, epochs=30, lr=0.05, encoding="reupload",
        noise_levels=(0.0, 0.01, 0.02, 0.05, 0.1), p_train=0.05, max_samples=300,
        seed=0, zne=True, verbose=False, metric=None, n_per_class=150, data_path=None) -> dict:
    split, full_train = load(dataset, qubits, max_samples, seed, n_per_class, data_path)
    metric = metric or PRIMARY.get(dataset, PRIMARY["default"])
    cfg = VQCConfig(n_qubits=qubits, n_layers=layers, n_classes=split.n_classes, encoding=encoding)
    res = {"config": {"dataset": dataset, "qubits": qubits, "layers": layers, "epochs": epochs,
                      "encoding": encoding, "p_train": p_train, "n_train": len(split.X_train),
                      "n_test": len(split.X_test), "n_classes": split.n_classes,
                      "vqc_params": cfg.n_params(), "metric": metric, "seed": seed}}
    for extra in ("features", "test_fraud_rate"):
        if hasattr(split, extra):
            res["config"][extra] = getattr(split, extra)
    res["classical"] = run_baselines(split, seed, full_train=full_train)

    clean = VQC(cfg, None, seed=seed)
    h_clean = train(clean, split.X_train, split.y_train, epochs, lr, seed=seed, verbose=verbose)
    aware = VQC(cfg, NoiseModel.uniform(p_train), seed=seed)
    h_aware = train(aware, split.X_train, split.y_train, epochs, lr, seed=seed, verbose=verbose)

    sweep = []
    for p in noise_levels:
        nm = NoiseModel.uniform(p)
        row = {"p": p,
               "clean_trained": clean.evaluate(split.X_test, split.y_test, nm),
               "noise_aware": aware.evaluate(split.X_test, split.y_test, nm)}
        if zne and p > 0:
            row["clean_trained+zne"] = zne_metrics(clean, split.X_test, split.y_test, nm)
        sweep.append(row)
    res["sweep"] = sweep
    res["train_seconds"] = {"clean": h_clean["seconds"], "aware": h_aware["seconds"]}
    res["loss_curves"] = {"clean": h_clean["loss"], "aware": h_aware["loss"]}
    return res


def _val(v, metric):
    return v if isinstance(v, (int, float)) else v.get(metric, float("nan"))


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = res["config"].get("metric", "acc")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, style in [("clean_trained", "o-"), ("noise_aware", "s-"), ("clean_trained+zne", "^--")]:
        pts = [(r["p"], _val(r[key], metric)) for r in res["sweep"] if key in r]
        if pts:
            ax[0].plot(*zip(*pts), style, label=key)
    name, best = max(((k, _val(v, metric)) for k, v in res["classical"].items()), key=lambda t: t[1])
    ax[0].axhline(best, color="gray", ls=":", label=f"best classical: {name} ({best:.2f})")
    ax[0].set(xlabel="noise strength p", ylabel=f"test {metric}",
              title=f"{res['config']['dataset']}: {metric} vs noise")
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
    ap.add_argument("--dataset", default="moons", help="moons|circles|iris|breast_cancer|digits01|creditcard|file:<csv>")
    ap.add_argument("--data-path", default=None, help="creditcard csv (default data/creditcard.csv)")
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--encoding", default="reupload", choices=["angle", "reupload"])
    ap.add_argument("--p-train", type=float, default=0.05)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--n-per-class", type=int, default=150, help="creditcard: balanced train size per class")
    ap.add_argument("--metric", default=None, help="primary metric (acc, f1, pr_auc, roc_auc, recall...)")
    ap.add_argument("--no-zne", action="store_true")
    ap.add_argument("--fast", action="store_true", help="tiny smoke run (~15s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(dataset=a.dataset, qubits=a.qubits, layers=a.layers, epochs=a.epochs, lr=a.lr,
              encoding=a.encoding, noise_levels=tuple(a.noise_levels), p_train=a.p_train,
              max_samples=a.max_samples, seed=a.seed, zne=not a.no_zne, verbose=a.verbose,
              metric=a.metric, n_per_class=a.n_per_class, data_path=a.data_path)
    if a.fast:
        kw.update(layers=2, epochs=8, max_samples=120, n_per_class=min(a.n_per_class, 60),
                  noise_levels=(0.0, 0.05, 0.1))
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
