"""MOCO: Measure Once, Compile Once.

  1. Feature map on 4 qubits (default: RY(x_q), no CNOT).
  2. On hardware, measure ONCE the reduced Bloch vector of every qubit for every training transaction
     (X and Z bases; + Y if the feature map needs it): 2-3 executions per transaction, in total.
  3. Heisenberg picture: after a trainable Rot on qubit q, <Z_q> = m_q . r_q(x) exactly (also for entangling
     feature maps: a local Rot only rotates the measurement axis). So training the circuit is a logistic
     regression on the measured Bloch vectors: convex, solved in PyTorch (LBFGS), global optimum,
     no executions inside the loop, no barren plateaus.
  4. Compile the solution back: Rot(-phi, -theta, 0) per qubit + observable H = sum_q w_q Z_q,
     p_fraud = (1 - <H>) / 2. The circuit does the classification.

Noise: with no CNOT and the same number of gates on every qubit, <H>_noisy = lambda^k <H>_ideal with
lambda = 1 - 4p/3 per depolarizing channel, so the ranking (AUPRC, AUROC, Spearman = 1) is the same at every p.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import average_precision_score

N_QUBITS = 4
# Chosen on validation only (results/selection_on_validation.csv, 5 seeds, test untouched):
#   single 1:10 -> val AUPRC 0.675 +- 0.011 (best); pair 1:30 ties (0.675) at 4x the cost -> single 1:10.
NEG_PER_POS = 10
FEATURE_MAP = "single"     # "single" | "pair" | "linear_cnot" | "kit_zz"  (last two: entangling ablations)
SHOTS = 1024
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5


def feature_map(x, fm=FEATURE_MAP, n=N_QUBITS):
    if fm == "single":
        for q in range(n):
            qml.RY(x[..., q], wires=q)
    elif fm == "pair":
        for q in range(n):
            qml.RY(x[..., q], wires=q)
            qml.RZ(x[..., (q + 1) % n], wires=q)
    elif fm == "linear_cnot":
        for q in range(n):
            qml.RY(x[..., q], wires=q)
        for q in range(n - 1):
            qml.CNOT(wires=[q, q + 1])
    elif fm == "kit_zz":   # the baseline's ZZ feature map (2 reps, all pairs)
        for _ in range(2):
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[..., i], wires=i)
            for i in range(n):
                for j in range(i + 1, n):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                    qml.CNOT(wires=[i, j])
    else:
        raise ValueError(fm)


def measured_bases(fm=FEATURE_MAP):
    return ("X", "Z") if fm == "single" else ("X", "Y", "Z")   # for "single" the Y component is identically 0


def bloch_exact(Z, fm=FEATURE_MAP, n=N_QUBITS):
    """Reduced Bloch vectors (N, n, 3) after the feature map. Analytic for "single", exact simulation otherwise."""
    if fm == "single":
        return np.stack([np.sin(Z), np.zeros_like(Z), np.cos(Z)], -1)

    @qml.qnode(qml.device("default.qubit", wires=n))
    def c(x):
        feature_map(x, fm, n)
        return [qml.expval(P(q)) for q in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    out = np.stack([np.asarray(v) for v in c(np.asarray(Z))], -1)
    return out.reshape(len(Z), n, 3)


def bloch_from_shots(R, shots, rng):
    """What hardware would measure: every component estimated from `shots` +/-1 outcomes."""
    return 2 * rng.binomial(shots, np.clip((1 + R) / 2, 0, 1)) / shots - 1


def design(R, fm=FEATURE_MAP):
    cols = [0, 2] if fm == "single" else [0, 1, 2]
    return R[:, :, cols].reshape(len(R), -1)


def train_logistic_torch(F, y, C):
    """min mean BCE(Fw + b, y) + ||w||^2 / (2 C N), the same problem as sklearn's LogisticRegression.
    Strictly convex, so LBFGS finds the global optimum whatever the initialisation."""
    F_t, y_t = torch.as_tensor(F, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(F.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    lam = 1.0 / (2 * C * len(y))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(F_t @ w + b, y_t) + lam * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def fit_convex(R_train, y_train, seed, fm=FEATURE_MAP):
    """C chosen by cross-validation on the already measured Bloch vectors -> no validation executions."""
    F = design(R_train, fm)
    fold = stratified_folds(y_train, N_FOLDS, seed)
    best = None
    for C in CS:
        aps = []
        for k in range(N_FOLDS):
            tr, va = fold != k, fold == k
            w, b = train_logistic_torch(F[tr], y_train[tr], C)
            aps.append(average_precision_score(y_train[va], F[va] @ w + b))
        if best is None or np.mean(aps) > best[0]:
            best = (np.mean(aps), C)
    w, b = train_logistic_torch(F, y_train, best[1])
    return w, best[1]


def compile_to_circuit(coef, fm=FEATURE_MAP, n=N_QUBITS):
    """Coefficients -> Rot(-phi, -theta, 0) per qubit + weights w_q of the observable. p_fraud = (1 - <H>)/2."""
    coef = coef.reshape(n, -1)
    m = np.zeros((n, 3))
    if fm == "single":
        m[:, 0], m[:, 2] = coef[:, 0], coef[:, 1]
    else:
        m[:] = coef
    m = -m
    w = np.linalg.norm(m, axis=1)
    mh = m / np.clip(w[:, None], 1e-12, None)
    theta = np.arccos(np.clip(mh[:, 2], -1, 1))
    phi = np.arctan2(mh[:, 1], mh[:, 0])
    return dict(rot=np.stack([-phi, -theta, np.zeros(n)], 1), w=w / w.sum(), fm=fm)


def score_exact(params, R):
    th, ph = -params["rot"][:, 1], -params["rot"][:, 0]
    mh = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return (1 - (R * mh[None]).sum(-1) @ params["w"]) / 2


def make_circuit(params, n=N_QUBITS):
    """The compiled circuit as circuit_fn(x, weights) for the kit protocol (weights unused, params are fixed)."""
    def circuit(x, weights=None):
        feature_map(x, params["fm"], n)
        for q in range(n):
            qml.Rot(*params["rot"][q], wires=q)
        return qml.expval(qml.Hamiltonian([float(v) for v in params["w"]], [qml.PauliZ(q) for q in range(n)]))
    return circuit


def transfer_threshold(t0, p, k):
    """For CNOT-free circuits with k gates per qubit, the threshold at p follows analytically from the one at 0."""
    return 0.5 - (1 - 4 * p / 3) ** k * (0.5 - t0)


def fit(Z_train, y_train, seed, fm=FEATURE_MAP, shots=SHOTS):
    """Measure (simulated with `shots`, or exact if shots=0), fit convexly, compile. Returns params and C."""
    R = bloch_exact(Z_train, fm)
    R_meas = bloch_from_shots(R, shots, np.random.default_rng(seed)) if shots else R
    coef, C = fit_convex(R_meas, y_train, seed, fm)
    return compile_to_circuit(coef, fm), C


def train_executions(n_rows, fm=FEATURE_MAP):
    """One execution per measured basis per transaction (all qubits are read at once)."""
    return n_rows * len(measured_bases(fm))


def tracker_check(Z_rows, fm=FEATURE_MAP, shots=SHOTS, n=N_QUBITS):
    """qml.Tracker count of the acquisition on hardware: one circuit per measured basis per transaction, all qubits
    read at once (moved here from moco_task2.py). Returns executions and shots per transaction."""
    device = qml.device("default.qubit", wires=n, seed=0)

    def acquisition(x, basis):
        feature_map(x, fm, n)
        for q in range(n):
            if basis == "X":
                qml.Hadamard(wires=q)
            elif basis == "Y":
                qml.RX(np.pi / 2, wires=q)        # rotates the Y axis onto Z
        return qml.counts(wires=range(n))

    qnode = qml.set_shots(qml.QNode(acquisition, device), shots=shots)
    with qml.Tracker(device) as tracker:
        for x in Z_rows:
            for basis in measured_bases(fm):
                qnode(x, basis)
    return {"executions_per_row": tracker.totals["executions"] / len(Z_rows),
            "shots_per_row": tracker.totals["shots"] / len(Z_rows)}
