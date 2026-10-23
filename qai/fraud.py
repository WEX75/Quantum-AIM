"""Credit-card fraud data (OpenML 42175 / Kaggle mlg-ulb) prepared for a small quantum model.

The full set is 284,807 transactions with 0.17% fraud. A 4-8 qubit circuit can't train on that,
so we:
  1. split train/test stratified on the FULL data (no leakage, test keeps real frauds),
  2. undersample the train split to `n_per_class` frauds + normals (balanced training),
  3. build the test set from ALL test frauds + `test_neg_ratio` x as many normals
     (still imbalanced, so precision / PR-AUC stay meaningful; ratio is reported),
  4. select `n_features` features on train only (ANOVA F-score, or PCA), scale to [0, pi].
`load_full_train` returns the unsubsampled train split for the strong classical reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Split

DEFAULT_PATH = Path("data/creditcard.csv")


def read_creditcard(path: Path | str = DEFAULT_PATH):
    import pandas as pd
    df = pd.read_csv(path)
    y = df["Class"].astype(str).str.strip("'\" ").astype(int).to_numpy()
    X = df.drop(columns=["Class"]).copy()
    if "Amount" in X:
        X["Amount"] = np.log1p(X["Amount"].astype(float))
    if "Time" in X:
        X["Time"] = (X["Time"].astype(float) % 86400) / 3600.0  # hour of day
    return X.to_numpy(float), y, list(X.columns)


def make_synthetic(path: Path | str, n: int = 20_000, fraud_rate: float = 0.0017, seed: int = 0):
    """Same schema as the real file, for tests and offline dry runs."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[:max(40, int(n * fraud_rate))] = 1  # ensure enough frauds for a split
    V = rng.normal(0, 1, (n, 28))
    V[y == 1, :6] += rng.normal(1.0, 1.0, (int(y.sum()), 6)) * np.array([1, -1, 1, -1, 1, -1])
    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.uniform(0, 172_792, n))
    df["Amount"] = rng.lognormal(3, 1.5, n)
    df["Class"] = y
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_creditcard(path: Path | str = DEFAULT_PATH, n_features: int = 4, n_per_class: int = 150,
                    test_size: float = 0.3, test_neg_ratio: int = 10, select: str = "anova",
                    seed: int = 0, return_full_train: bool = False):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import f_classif
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X, y, cols = read_creditcard(path)
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed,
                                              stratify=y)
    X_full, y_full = X_tr, y_tr

    def sub(Xs, ys, n_pos, n_neg):
        pos, neg = np.flatnonzero(ys == 1), np.flatnonzero(ys == 0)
        idx = np.concatenate([rng.choice(pos, min(n_pos, len(pos)), replace=False),
                              rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        rng.shuffle(idx)
        return Xs[idx], ys[idx]

    X_tr, y_tr = sub(X_tr, y_tr, n_per_class, n_per_class)
    n_pos_te = int((y_te == 1).sum())
    X_te, y_te = sub(X_te, y_te, n_pos_te, n_pos_te * test_neg_ratio)

    std = StandardScaler().fit(X_tr)
    X_tr, X_te, X_full_s = std.transform(X_tr), std.transform(X_te), std.transform(X_full)
    if select == "pca":
        red = PCA(n_components=n_features, random_state=seed).fit(X_tr)
        f = red.transform
        chosen = [f"PC{i+1}" for i in range(n_features)]
    else:
        F, _ = f_classif(X_tr, y_tr)
        keep = np.argsort(np.nan_to_num(F))[::-1][:n_features]
        f = lambda A: A[:, keep]
        chosen = [cols[i] for i in keep]
    X_tr, X_te, X_full_s = f(X_tr), f(X_te), f(X_full_s)
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(X_tr)
    X_tr = mm.transform(X_tr)
    X_te = np.clip(mm.transform(X_te), 0, np.pi)
    X_full_s = np.clip(mm.transform(X_full_s), 0, np.pi)
    split = Split(X_tr, X_te, y_tr, y_te, 2, "creditcard")
    split.features = chosen
    split.test_fraud_rate = float(y_te.mean())
    if return_full_train:
        return split, (X_full_s, y_full)
    return split
