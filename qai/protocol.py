"""Task 2 evaluation protocol, reproduced from the kit notebook (kit/task1_guided_vqc_fraud_detection.ipynb).

Each function names the kit cell it copies, so that the baseline and our methods are measured exactly the
way the organizers measure them. Generalisations are marked; with default arguments they match the kit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42                     # master seed of the common split
SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS = [0.001, 0.01, 0.05]
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}
SHOTS_PER_EXECUTION = 1024
DATA_PATH = Path("data/creditcard.csv")
SPLIT_PATH = Path("data/split_indices.npz")
METRICS = ["AUPRC", "AUROC", "Spearman rho", "precision", "recall", "F1", "MCC", "threshold"]


# ------------------------------------------------------------------ data (cells 17, 33, 35, 38, 41-42, 123)
@dataclass
class CommonSplit:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def load_df(path=DATA_PATH, sample_fraction=1.0):
    """Cells 17 + 33. sample_fraction < 1 keeps a stratified fraction, for smoke runs only."""
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(int)
    if sample_fraction < 1.0:
        df, _ = train_test_split(df, train_size=sample_fraction, stratify=df["Class"], random_state=SEED)
    return df


def common_split(df, split_path=SPLIT_PATH):
    """Cell 35: stratified 60/20/20 with seed 42. On the full data it also writes split_indices.npz (cell 123)
    and, when the file already exists, checks that it matches."""
    X, y = df.drop(columns="Class"), df["Class"].to_numpy()
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval,
                                                      random_state=SEED)
    if len(df) == 284_807 and split_path is not None:
        idx = dict(train=X_train.index.to_numpy(), val=X_val.index.to_numpy(), test=X_test.index.to_numpy())
        if Path(split_path).exists():
            saved = np.load(split_path)
            assert all((saved[k] == v).all() for k, v in idx.items()), "split differs from split_indices.npz"
        else:
            np.savez(split_path, **idx)
    return CommonSplit(X_train, y_train, X_val, y_val, X_test, y_test)


def undersample(X_part, y_part, seed, neg_per_pos=1):
    """Cell 38. Generalised: neg_per_pos legit transactions per fraud (the kit uses 1)."""
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=neg_per_pos * len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]


class AngleEncoder:
    """Cells 41-42: StandardScaler on Time and Amount, PCA, MinMax to [0, pi]; fitted on training data only."""
    raw_columns = ["Time", "Amount"]

    def __init__(self, n_components, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame):
        out = frame.copy()
        out[self.raw_columns] = self.scaler.transform(frame[self.raw_columns])
        return out.to_numpy()

    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        self.angles.fit(self.pca.fit_transform(self._scaled(frame)))
        return self

    def transform(self, frame):
        return np.clip(self.angles.transform(self.pca.transform(self._scaled(frame))), 0, np.pi)


class RobustAngleEncoder(AngleEncoder):
    """NOT in the kit. Like AngleEncoder, but the angle scale is fitted on the [q, 100 - q] percentiles of each PCA
    component of the training data instead of its full range, then clipped to [0, pi]. q = 0 is the kit encoder."""

    def __init__(self, n_components, seed, q=0.0):
        super().__init__(n_components, seed)
        self.q = q

    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        components = self.pca.fit_transform(self._scaled(frame))
        self.lo, self.hi = np.percentile(components, [self.q, 100 - self.q], axis=0)
        return self

    def transform(self, frame):
        components = self.pca.transform(self._scaled(frame))
        return np.clip((components - self.lo) / (self.hi - self.lo) * np.pi, 0, np.pi)


# ------------------------------------------------------------------ metrics (cell 92)
def choose_threshold(y_true, scores):
    """The threshold that maximises F1 on (y_true, scores)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ------------------------------------------------------------------ noise model (cell 106)
def noisy_qnode(circuit_fn, p, n_wires=4):
    """Native gates, then a depolarizing channel of probability p after every gate, on each of its qubits."""
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p,
                                    position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def fraud_readout(expval):
    return (1 - expval) / 2


def clean_predict(Z, weights, circuit_fn, readout=fraud_readout, n_wires=4, chunk=20000):
    """Cell 81 `predict`, for any circuit_fn."""
    circuit = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch")
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def noisy_predict(Z, weights, p, circuit_fn, readout=fraud_readout, n_wires=4, chunk=2000):
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def gate_stats(circuit_fn, *args, n_wires=4):
    """Native gate count, two-qubit gates, noise locations (cell 106 `noise_locations`) and depth."""
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)),
                                     gate_set=NATIVE_GATES)
    res = qml.specs(qnode)(*args)["resources"]
    return {"gates": res.num_gates, "two_qubit": res.gate_sizes.get(2, 0),
            "noise_locations": sum(size * count for size, count in res.gate_sizes.items()), "depth": res.depth}


# ------------------------------------------------------------------ protocol (cell 125, split in two steps)
def circuit_scorer(circuit_fn, weights, readout=fraud_readout, n_wires=4, chunk=2000):
    """score_fn(Z, p) for a PennyLane circuit: state vector at p = 0, density matrices with noise otherwise."""
    def score_fn(Z, p):
        if p == 0.0:
            return clean_predict(Z, weights, circuit_fn, readout, n_wires)
        return noisy_predict(Z, weights, p, circuit_fn, readout, n_wires, chunk)
    return score_fn


def scores_under_noise(score_fn, Z_val, Z_test, noise_levels=NOISE_LEVELS):
    """{p: (validation scores, test scores)} for p = 0 and every noise level."""
    return {p: (score_fn(Z_val, p), score_fn(Z_test, p)) for p in [0.0, *noise_levels]}


def protocol_table(scores, y_val, y_test):
    """Cell 125: threshold maximising F1 on validation scores at the same p, metrics on test, Spearman rho
    between noisy and noiseless test scores."""
    clean_test = scores[0.0][1]
    rows = []
    for p, (val_scores, test_scores) in scores.items():
        threshold = choose_threshold(y_val, val_scores)
        rows.append({"p": p, **evaluate(y_test, test_scores, threshold),
                     "Spearman rho": spearmanr(clean_test, test_scores).statistic, "threshold": threshold})
    return pd.DataFrame(rows).set_index("p")


def evaluate_under_noise(score_fn, Z_val, y_val, Z_test, y_test, noise_levels=NOISE_LEVELS):
    return protocol_table(scores_under_noise(score_fn, Z_val, Z_test, noise_levels), y_val, y_test)


def save_scores(path, scores, **arrays):
    """{p: (val, test)} -> npz with keys val_<p>, test_<p> (plain arrays, no pickle)."""
    flat = {f"{part}_{p}": s for p, pair in scores.items() for part, s in zip(("val", "test"), pair)}
    np.savez(path, **flat, **arrays)


def load_scores(path):
    data = np.load(path)
    levels = sorted(float(k[4:]) for k in data.files if k.startswith("val_"))
    return {p: (data[f"val_{p}"], data[f"test_{p}"]) for p in levels}


def mean_std(tables):
    """Per-seed protocol tables -> one table of mean and std over seeds for every p."""
    return pd.concat(tables).groupby(level="p")[METRICS].agg(["mean", "std"])


# ------------------------------------------------------------------ cost ledger (cells 86, 89)
def ledger(name, train_exec, val_exec, shots=SHOTS_PER_EXECUTION, **extra):
    """One ledger row: circuit executions and shots for training and validation on hardware."""
    return {"method": name, "train executions": train_exec, "validation executions": val_exec,
            "total executions": train_exec + val_exec, "total shots": (train_exec + val_exec) * shots, **extra}


def gradient_ledger(name, epochs, n_train, n_params, n_val_scored, **extra):
    """Cell 86: one forward pass plus two parameter-shift evaluations per parameter, per example and epoch;
    validation scores n_val_scored transactions after every epoch."""
    return ledger(name, epochs * n_train * (1 + 2 * n_params), epochs * n_val_scored,
                  executions_per_example=1 + 2 * n_params, **extra)


def tracked_executions_per_example(circuit_fn, weights, Z_batch, y_batch, n_wires=4, shots=SHOTS_PER_EXECUTION):
    """Cell 89: executions per training example of one parameter-shift gradient step, counted by qml.Tracker."""
    device = qml.device("default.qubit", wires=n_wires, seed=SEED)
    qnode = qml.set_shots(qml.QNode(circuit_fn, device, interface="torch", diff_method="parameter-shift"),
                          shots=shots)
    w = weights.detach().clone().requires_grad_(True)
    with qml.Tracker(device) as tracker:
        loss = torch.nn.BCELoss()(fraud_readout(qnode(torch.tensor(Z_batch), w)),
                                  torch.tensor(y_batch, dtype=torch.float64))
        loss.backward()
    return tracker.totals["executions"] / len(Z_batch)


def tracked_executions_per_example_readout(circuit_fn, weights, Z_batch, y_batch, readout, n_wires=4,
                                           shots=SHOTS_PER_EXECUTION):
    """Like tracked_executions_per_example (cell 89), for circuits whose fraud probability is
    readout(qnode_output, weights) rather than (1 - <Z_0>) / 2."""
    device = qml.device("default.qubit", wires=n_wires, seed=SEED)
    qnode = qml.set_shots(qml.QNode(circuit_fn, device, interface="torch", diff_method="parameter-shift"),
                          shots=shots)
    w = weights.detach().clone().requires_grad_(True)
    with qml.Tracker(device) as tracker:
        loss = torch.nn.BCELoss()(readout(qnode(torch.tensor(Z_batch), w), w),
                                  torch.tensor(y_batch, dtype=torch.float64))
        loss.backward()
    return tracker.totals["executions"] / len(Z_batch)


def with_tracker(row, tracker_per_example, n_examples_trained):
    """Add the qml.Tracker count next to the formula in a ledger row."""
    return {**row, "tracker_per_example": tracker_per_example,
            "train executions (Tracker)": int(round(tracker_per_example * n_examples_trained)),
            "total executions (Tracker)": int(round(tracker_per_example * n_examples_trained))
            + row["validation executions"]}


def training_examples(seed=0, neg_per_pos=1, n=5, n_components=4):
    """The first n encoded training examples of a seed's undersampled set (for the Tracker checks)."""
    split = common_split(load_df())
    X_b, y_b = undersample(split.X_train, split.y_train, seed, neg_per_pos)
    Z_b = AngleEncoder(n_components, seed).fit(X_b).transform(X_b)
    return Z_b[:n], y_b[:n]
