"""Dataset loading + preprocessing into rotation angles.

`load_dataset("moons")` for toy runs; `load_dataset("file:data/kit.csv")` for the starter kit
(last column = label, or pass label_col). Features are reduced with PCA to `n_features`
(= number of qubits) and min-max scaled to [0, pi].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int
    name: str


def _raw(name: str, seed: int, label_col: int | str | None):
    if name == "moons":
        return datasets.make_moons(n_samples=400, noise=0.15, random_state=seed)
    if name == "circles":
        return datasets.make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=seed)
    if name == "iris":
        d = datasets.load_iris()
        return d.data, d.target
    if name == "breast_cancer":
        d = datasets.load_breast_cancer()
        return d.data, d.target
    if name == "digits01":
        d = datasets.load_digits(n_class=2)
        return d.data, d.target
    if name.startswith("file:"):
        return load_file(Path(name[5:]), label_col)
    raise ValueError(f"unknown dataset {name!r}")


def load_file(path: Path, label_col: int | str | None = None):
    """CSV (header ok) or NPZ with arrays X, y."""
    if path.suffix == ".npz":
        d = np.load(path)
        X, y = d["X"], d["y"]
        return X.reshape(len(X), -1), y.reshape(-1)
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    header = None
    try:
        float(rows[0][0])
    except ValueError:
        header, rows = rows[0], rows[1:]
    arr = np.array(rows)
    if isinstance(label_col, str):
        label_col = header.index(label_col)
    lc = -1 if label_col is None else label_col
    y_raw = arr[:, lc]
    X = np.delete(arr, lc if lc >= 0 else arr.shape[1] + lc, axis=1).astype(float)
    _, y = np.unique(y_raw, return_inverse=True)
    return X, y


def load_dataset(name: str = "moons", n_features: int = 4, test_size: float = 0.3,
                 max_samples: int | None = 300, seed: int = 0,
                 label_col: int | str | None = None) -> Split:
    X, y = _raw(name, seed, label_col)
    X, y = np.asarray(X, float), np.asarray(y).astype(int)
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=seed, stratify=y)
    # fit preprocessing on train only (no leakage)
    std = StandardScaler().fit(X_tr)
    X_tr, X_te = std.transform(X_tr), std.transform(X_te)
    if X_tr.shape[1] > n_features:
        pca = PCA(n_components=n_features, random_state=seed).fit(X_tr)
        X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
    elif X_tr.shape[1] < n_features:  # pad by repeating features (cheap re-encoding)
        reps = int(np.ceil(n_features / X_tr.shape[1]))
        X_tr = np.tile(X_tr, reps)[:, :n_features]
        X_te = np.tile(X_te, reps)[:, :n_features]
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(X_tr)
    X_tr, X_te = mm.transform(X_tr), np.clip(mm.transform(X_te), 0, np.pi)
    return Split(X_tr, X_te, y_tr, y_te, int(len(np.unique(y))), name)
