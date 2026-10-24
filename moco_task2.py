"""
BreaQ 2026 — Task 2
MOCO: Measure Once, Compile Once

  1. Feature map pe 4 qubiti (implicit: RY(x_q), fara CNOT).
  2. Pe hardware masuram O SINGURA DATA vectorul Bloch redus al fiecarui qubit pentru fiecare tranzactie de train
     (bazele X, Z; + Y daca feature map-ul o cere): 2-3 executii per tranzactie, TOTAL.
  3. Imaginea Heisenberg: dupa o rotatie antrenabila Rot pe qubitul q, <Z_q> = m_q . r_q(x) EXACT
     (si pentru feature map-uri cu entanglement: Rot local roteste doar axa de masurare).
     => antrenarea circuitului = regresie logistica pe vectorii Bloch masurati: CONVEXA, in PyTorch (LBFGS),
        optim global, zero executii in bucla, zero barren plateaus.
  4. Compilam solutia inapoi: Rot(-phi, -theta, 0) per qubit + observabil H = sum_q w_q Z_q.
     p_fraud = (1 - <H>)/2. Clasificarea o face circuitul.

Zgomot: fara CNOT si cu acelasi numar de porti pe fiecare qubit,
  <H>_noisy = lambda^k <H>_ideal, lambda = 1 - 4p/3  => ordonare, AUPRC, AUROC, Spearman = 1 identice la orice p.

Stack: PennyLane (cuantic) + PyTorch (partea clasica antrenabila). scikit-learn doar pentru preprocesare/metrici/split.
"""
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.set_default_dtype(torch.float64)

SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]
N_QUBITS = 4
# Configuratie aleasa DOAR pe validare (results/selection_on_validation.csv, 5 seed-uri, testul neatins):
#   single 1:10 -> VAL AUPRC 0.675 +- 0.011 (maxim); pair 1:30 egal (0.675) dar de 4x mai scump -> se alege single 1:10.
NEG_PER_POS = 10
FEATURE_MAP = "single"     # "single" | "pair" | "linear_cnot" | "kit_zz"  (ultimele doua = ablatii cu entanglement)
SHOTS = 1024
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5


# ------------------------------------------------------------------ split comun + encoder-ul kitului
def common_split(df, seed=42):
    X, y = df.drop(columns="Class"), df["Class"].values
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xv, ytr, yv = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=seed)
    return Xtr, ytr, Xv, yv, Xte, yte


def undersample(X, y, seed, neg_per_pos=NEG_PER_POS):
    rng = np.random.default_rng(seed)
    f = np.flatnonzero(y == 1)
    l = rng.choice(np.flatnonzero(y == 0), neg_per_pos * len(f), replace=False)
    k = rng.permutation(np.r_[f, l])
    return X.iloc[k], y[k]


class AngleEncoder:   # identic cu kitul: StandardScaler(Time, Amount) -> PCA(4) -> MinMax [0, pi], fit doar pe train
    def __init__(self, seed, n_components=N_QUBITS):
        self.sc, self.pca, self.mm = StandardScaler(), PCA(n_components, random_state=seed), MinMaxScaler((0, np.pi))

    def _s(self, f):
        f = f.copy()
        f[["Time", "Amount"]] = self.sc.transform(f[["Time", "Amount"]])
        return f.values

    def fit(self, f):
        self.sc.fit(f[["Time", "Amount"]])
        self.mm.fit(self.pca.fit_transform(self._s(f)))
        return self

    def transform(self, f):
        return np.clip(self.mm.transform(self.pca.transform(self._s(f))), 0, np.pi)


# ------------------------------------------------------------------ feature map-uri (o singura definitie, folosita peste tot)
def feature_map(x, fm=FEATURE_MAP, n=N_QUBITS, gate=None):
    """Aplica feature map-ul. `gate(op_class, params, wires)` permite inserarea zgomotului dupa fiecare poarta."""
    g = gate or (lambda op, prm, w: op(*prm, wires=w))
    if fm == "single":
        for q in range(n):
            g(qml.RY, [x[..., q]], q)
    elif fm == "pair":
        for q in range(n):
            g(qml.RY, [x[..., q]], q)
            g(qml.RZ, [x[..., (q + 1) % n]], q)
    elif fm == "linear_cnot":
        for q in range(n):
            g(qml.RY, [x[..., q]], q)
        for q in range(n - 1):
            g(qml.CNOT, [], [q, q + 1])
    elif fm == "kit_zz":   # ZZFeatureMap-ul baseline-ului (2 reps, full entanglement)
        for _ in range(2):
            for i in range(n):
                g(qml.Hadamard, [], i)
                g(qml.RZ, [2 * x[..., i]], i)
            for i in range(n):
                for j in range(i + 1, n):
                    g(qml.CNOT, [], [i, j])
                    g(qml.RZ, [2 * (np.pi - x[..., i]) * (np.pi - x[..., j])], j)
                    g(qml.CNOT, [], [i, j])
    else:
        raise ValueError(fm)


def measured_bases(fm=FEATURE_MAP):
    return ("X", "Z") if fm in ("single",) else ("X", "Y", "Z")   # la "single" componenta Y e identic 0


def bloch_exact(Z, fm=FEATURE_MAP, n=N_QUBITS):
    """Vectorii Bloch redusi (N, n, 3) dupa feature map. Analitic pentru "single"; altfel simulare PennyLane exacta."""
    if fm == "single":
        return np.stack([np.sin(Z), np.zeros_like(Z), np.cos(Z)], -1)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def c(x):
        feature_map(x, fm, n)
        return [qml.expval(P(q)) for q in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    out = np.stack([np.asarray(v) for v in c(np.asarray(Z))], -1)
    return out.reshape(len(Z), n, 3)


def bloch_from_shots(R, shots, rng):
    """Ce s-ar masura pe hardware: fiecare componenta estimata din `shots` rezultate +/-1."""
    return 2 * rng.binomial(shots, np.clip((1 + R) / 2, 0, 1)) / shots - 1


def design(R, fm=FEATURE_MAP):
    cols = [0, 2] if fm == "single" else [0, 1, 2]
    return R[:, :, cols].reshape(len(R), -1)


# ------------------------------------------------------------------ antrenare convexa in PyTorch (zero executii in bucla)
def train_logistic_torch(F, y, C):
    """min  mean BCE(Fw + b, y) + ||w||^2 / (2 C N)   (aceeasi problema ca LogisticRegression din sklearn).
    Strict convexa -> LBFGS gaseste optimul global; nu depinde de initializare."""
    F_t, y_t = torch.as_tensor(F), torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(F.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
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
    """C ales prin cross-validation pe vectorii Bloch DEJA masurati -> 0 executii de validare."""
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
    """Coeficienti -> Rot(-phi, -theta, 0) per qubit + ponderi w_q ale observabilului. p_fraud = (1 - <H>)/2."""
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


# ------------------------------------------------------------------ circuitul compilat (QNode) + protocolul de zgomot
def make_protocol_qnode(n=N_QUBITS, device_name="default.qubit"):
    """qnode(x, params) pentru evaluate_under_noise din kit. params = compile_to_circuit(...)."""
    dev = qml.device(device_name, wires=n)

    @qml.qnode(dev)
    def circuit(x, params):
        feature_map(x, params["fm"], n)
        for q in range(n):
            qml.Rot(*params["rot"][q], wires=q)
        return qml.expval(qml.Hamiltonian([float(v) for v in params["w"]], [qml.PauliZ(q) for q in range(n)]))

    return circuit


def noisy_scores(params, Z, p, n=N_QUBITS, batch=20000):
    """Protocolul oficial, vectorizat: fiecare poarta (deja nativa) urmata de DepolarizingChannel(p)
    pe fiecare qubit pe care actioneaza, simulat pe default.mixed. Intoarce p_fraud."""
    dev = qml.device("default.mixed", wires=n)

    def gate(op, prm, w):
        op(*prm, wires=w)
        for wi in ([w] if isinstance(w, int) else w):
            qml.DepolarizingChannel(p, wires=wi)

    @qml.qnode(dev)
    def c(x):
        feature_map(x, params["fm"], n, gate)
        for q in range(n):
            gate(qml.Rot, list(params["rot"][q]), q)
        return qml.expval(qml.Hamiltonian([float(v) for v in params["w"]], [qml.PauliZ(q) for q in range(n)]))

    return np.concatenate([(1 - np.asarray(c(Z[i:i + batch]))) / 2 for i in range(0, len(Z), batch)])


def max_f1_threshold(y, s):
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return th[np.argmax(f1[:-1])]


def evaluate(params, Z_val, y_val, Z_test, y_test, ideal_test=None):
    """Tabelul oficial pe seturile COMPLETE: prag max-F1 pe validarea zgomotoasa la acelasi p."""
    rows = []
    ideal_test = noisy_scores(params, Z_test, 0.0) if ideal_test is None else ideal_test
    for p in NOISE_LEVELS:
        sv, st = noisy_scores(params, Z_val, p), noisy_scores(params, Z_test, p)
        t = max_f1_threshold(y_val, sv)
        yp = (st >= t).astype(int)
        rows.append(dict(p=p, AUPRC=average_precision_score(y_test, st), AUROC=roc_auc_score(y_test, st),
                         Precision=precision_score(y_test, yp, zero_division=0), Recall=recall_score(y_test, yp),
                         F1=f1_score(y_test, yp), MCC=matthews_corrcoef(y_test, yp),
                         Spearman=spearmanr(ideal_test, st).statistic, threshold=t))
    return pd.DataFrame(rows)


def transfer_threshold(t0, p, k):
    """Bonus: pentru circuite fara CNOT cu k porti per qubit, pragul la p rezulta analitic din pragul la p=0."""
    return 0.5 - (1 - 4 * p / 3) ** k * (0.5 - t0)


def gate_count(fm=FEATURE_MAP, n=N_QUBITS):
    tape = qml.workflow.construct_tape(make_protocol_qnode())(np.zeros(n), dict(rot=np.zeros((n, 3)), w=np.ones(n) / n, fm=fm))
    ops = tape.operations
    return dict(gates=len(ops), cnot=sum(o.name == "CNOT" for o in ops),
                noise_locations=sum(len(o.wires) for o in ops), depth=tape.graph.get_depth())


# ------------------------------------------------------------------ cost ledger
BASELINE_LEDGER = dict(train_exec=1_150_500, val_exec=1_708_860, total_exec=2_859_360, total_shots=2_859_360 * 1024)


def cost_ledger(n_train_rows, fm=FEATURE_MAP, shots=SHOTS):
    """Definitia kitului: antrenare + validare pentru selectia modelului.
    MOCO: 1 executie per baza masurata per tranzactie (toti qubitii se citesc simultan); selectie prin CV -> 0.
    Trecerea de evaluare a protocolului (prag pe validare) e identica pentru toate metodele si se raporteaza separat."""
    per_row = len(measured_bases(fm))
    train_exec = n_train_rows * per_row
    return dict(exec_per_row=per_row, train_exec=train_exec, val_exec=0, total_exec=train_exec,
                total_shots=train_exec * shots,
                reduction_exec_pct=100 * (1 - train_exec / BASELINE_LEDGER["total_exec"]),
                reduction_shots_pct=100 * (1 - train_exec * shots / BASELINE_LEDGER["total_shots"]))


def tracker_check(R_rows, fm=FEATURE_MAP, shots=SHOTS, n=N_QUBITS):
    """Verificare cu qml.Tracker: un circuit de achizitie per baza, rulat pe R_rows tranzactii."""
    dev = qml.device("default.qubit", wires=n, shots=shots)

    @qml.qnode(dev)
    def acq(x, basis):
        feature_map(x, fm, n)
        for q in range(n):
            if basis == "X":
                qml.Hadamard(wires=q)
            elif basis == "Y":
                qml.RX(np.pi / 2, wires=q)
        return qml.counts(wires=range(n))

    with qml.Tracker(dev) as tr:
        for x in R_rows:
            for bs in measured_bases(fm):
                acq(x, bs)
    return dict(executions=tr.totals["executions"], shots=tr.totals["shots"])


# ------------------------------------------------------------------ un seed cap-coada
def run_seed(Xtr, ytr, Xv, yv, Xte, yte, seed, shots=SHOTS, fm=FEATURE_MAP, neg_per_pos=NEG_PER_POS, full_eval=True):
    Xb, yb = undersample(Xtr, ytr, seed, neg_per_pos)
    enc = AngleEncoder(seed).fit(Xb)
    Zb, Zv, Zte = enc.transform(Xb), enc.transform(Xv), enc.transform(Xte)
    R = bloch_exact(Zb, fm)
    R_meas = bloch_from_shots(R, shots, np.random.default_rng(seed)) if shots else R
    coef, C = fit_convex(R_meas, yb, seed, fm)
    params = compile_to_circuit(coef, fm)
    s_ideal = score_exact(params, bloch_exact(Zte, fm))
    out = dict(seed=seed, params=params, C=C, encoder=enc, ledger=cost_ledger(len(yb), fm, shots or SHOTS),
               test_auprc_ideal=average_precision_score(yte, s_ideal))
    if full_eval:
        out["table"] = evaluate(params, Zv, yv, Zte, yte, ideal_test=s_ideal)
    return out


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv")
    Xtr, ytr, Xv, yv, Xte, yte = common_split(df)
    res = [run_seed(Xtr, ytr, Xv, yv, Xte, yte, s) for s in SEEDS]
    tab = pd.concat([r["table"] for r in res]).groupby("p").agg(["mean", "std"])
    pd.set_option("display.width", 200)
    print(tab.drop(columns="threshold").round(4))
    print("gates:", gate_count(), "| ledger:", res[0]["ledger"])
