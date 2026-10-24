"""Adaptive shots: pay the noise tax only where it is needed.

On hardware every MOCO score is estimated from shots. One shot reads the 4 qubits at once, so it yields the per-shot
estimator h = sum_q w_q z_q in [-1, 1] with mean <H>_p = lambda^2 <H>_0 and score s = (1 - <H>) / 2. A fixed budget
(1024 shots) pays the same for every transaction. Triage: measure in batches of b shots and stop as soon as an
anytime-valid confidence interval for the mean of h excludes the threshold t_h(p) = 1 - 2 t_p, with a cap S_max.
t_p = transfer_threshold(t_0, p) comes analytically from the noiseless threshold t_0, which is computed classically
(product state) on validation: no noisy validation pass at all.

Guarantee (by construction, qai/adaptive.py, coverage tested in tests/test_adaptive.py): P(the interval misses the
mean at ANY batch) <= alpha. On the complement every early stop gives the same decision as the exact (infinite-shot)
score, so P(a transaction is stopped early with a decision that differs from the exact one) <= alpha. Transactions
that reach S_max are decided by the point estimate and carry no guarantee (reported as "capped").

Encodings. "MOCO kit" is the official MOCO (kit AngleEncoder, cached circuit). The smoke run on the kit encoding
showed that its legit transactions sit within ~0.08 of the threshold in units of <H> (the outlying frauds set the
MinMax range), so triage saves nothing there. "MOCO robust q" fits the angle scale on the [q, 100 - q] percentiles of
each PCA component of the train bag (protocol.RobustAngleEncoder, clipped to [0, pi]) and is trained exactly like
MOCO (moco.fit). q = 0 is the kit encoder. The official MOCO table is not changed.

Hypotheses (written before the code; also in docs/LOG.md, 16:45 and 17:20):
  H1  The anytime intervals miss the mean in <= alpha of 10,000 simulated sequences, while a fixed-n CLT interval
      re-checked at every batch exceeds alpha (tests/test_adaptive.py).
  H2  (kit encoding, REFUTED by the smoke run at 17:10) MOCO triage at p <= 0.05 cuts the mean shots >= 10x.
  H3  At the same mean budget, adaptive beats fixed S on F1.
  H4  The mean shots grow roughly like lambda^-4 with p (the margins |<H> - t_h| shrink like lambda^2).
  H5  On the baseline almost every transaction reaches S_max at p >= 0.05: triage cannot work there.
  H6  Noise meter: the error of p_hat falls like 1/sqrt(N_cal S); transfer_threshold(t_0, p_hat) keeps the F1 of the
      threshold calibrated on the full noisy validation set.
  H7  Robust scaling with q >= 0.5% cuts MOCO's mean shots >= 10x against the kit encoding at equal validation F1
      (+-0.01) at p = 0.05, with an exact AUPRC lower by <= 0.01.
  H8  The plug-in empirical Bernstein interval is tighter than Hoeffding / Maurer-Pontil: fewer shots at the same alpha.

Selection rule (written before the run; validation only, test untouched):
  Grid: q in {0 (kit), 0.5, 1, 2, 5}% x interval in {Hoeffding, Bernstein, plug-in Bernstein} x b in {16, 32, 64}
  x alpha in {0.001, 0.01, 0.05} x S_max in {1024, 4096}, each run as adaptive triage on the full validation set at
  p = 0.05, 5 seeds. Score: validation F1 (mean over seeds); configurations within 0.005 of the best F1 count as tied
  and the tie goes to the fewest mean shots. The overall winner fixes q* and the headline S_max. For the test tables
  (MOCO kit, MOCO robust q*, baseline) x S_max in {1024, 4096}, the (interval, b, alpha) of each pair is picked by the
  same rule inside that pair.
  Added at 17:40, after the full validation grid picked q* = 0 (the kit encoding): the best robust q by the same rule
  restricted to q > 0 (validation only) is also carried to test, as a comparator that was NOT selected. The baseline is triaged with its own pick; its threshold is the protocol one (max F1
  on its exact noisy validation scores at the same p), which is generous to it (that pass costs 56,962 executions).
  The fixed "same budget" comparator uses the picked configuration's mean shots on VALIDATION (same p, mean over
  seeds): no shot count comes from test.
  Noise meter: the estimator with the lower relative RMSE of p_hat (calibration rows drawn from validation, labels not
  used) and the smallest N_cal with relative RMSE <= 10% at every p for S = 1024 go into the ledger.

Simulation: the final state is a product state, so the shots of each qubit are independent with
P(z_q = +1) = (1 + lambda^2 <Z_q>_0) / 2; each shot draws the 4 bits and forms h, so the Bernstein variances are those
of the real per-shot estimator. The baseline reads Z_0 only: P(z_0 = +1) = 1 - s_p with its cached (exact, kit noise
model) scores, not recomputed. Checks: results/task2/adaptive_checks.csv.

    python3 -m qai.experiments.exp_adaptive_shots --smoke     # seeds 0-1, all frauds + 3000 legit rows, *_fast outputs
    python3 -m qai.experiments.exp_adaptive_shots             # 5 seeds, full validation and test sets
    python3 -m qai.experiments.exp_adaptive_shots --plot      # redraw the figures from the cached results
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qai import adaptive as ad
from qai import moco
from qai.protocol import (SEEDS, AngleEncoder, RobustAngleEncoder, choose_threshold, common_split, evaluate, load_df,
                          load_scores, noisy_predict, undersample)

OUT = Path("results/task2")
P_LEVELS = [0.0, 0.01, 0.05, 0.2]
SELECT_P = 0.05
CAPS = [1024, 4096]
Q_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
GRID = [(ci, b, a) for ci in ad.CI_KINDS for b in (16, 32, 64) for a in (0.001, 0.01, 0.05)]
F1_TIE = 0.005
S_FIXED = 1024
S_CURVE = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
K_GATES = 2                                        # gates per qubit of the MOCO circuit
METER_P, METER_NCAL, METER_S = [0.01, 0.05, 0.2], [50, 200, 1000], [256, 1024]
METER_REL_RMSE = 0.10
N_CHECK = 500
BASELINE = "baseline VQC"
W_BASELINE = np.array([1.0])
KEEP = ("exact", "fixed 1024", "fixed same budget", "adaptive", "adaptive + refine")   # methods whose rows go to npz


def lam2(p):
    return (1 - 4 * p / 3) ** 2


def suffix(smoke):
    return "_fast" if smoke else ""


def moco_name(q):
    return "MOCO kit" if q == 0 else f"MOCO robust q={q:g}%"


def q_of(name):
    return 0.0 if name == "MOCO kit" else float(name.split("q=")[1].rstrip("%"))


def slug(name):
    return name.replace(" ", "_").replace("=", "").replace("%", "pct")


def score_of(mean_h):
    return (1 - mean_h) / 2


# ------------------------------------------------------------------ data of one seed
@dataclass
class MocoModel:
    q: float
    Z_val: np.ndarray            # angles (for the Tracker run and the kit noisy_predict checks)
    zq_val: np.ndarray           # <Z_q>_0 per row and qubit, noiseless
    zq_test: np.ndarray
    rot: np.ndarray
    w: np.ndarray
    t0: float                    # max-F1 threshold on the exact noiseless validation scores (classical)
    protocol: dict | None        # cached protocol scores {p: (val, test)} (kit noise model), if they exist
    protocol_rot: np.ndarray | None


def _rows(y, smoke):
    if not smoke:
        return np.arange(len(y))
    return np.sort(np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)[:3000]])


class SeedData:
    def __init__(self, seed, smoke):
        split = common_split(load_df())
        self.seed = seed
        rv, rt = _rows(split.y_val, smoke), _rows(split.y_test, smoke)
        self.rv, self.rt = rv, rt
        self.y_val, self.y_test = split.y_val[rv], split.y_test[rt]
        self.X_val, self.X_test = split.X_val.iloc[rv], split.X_test.iloc[rt]
        self.X_b, self.y_b = undersample(split.X_train, split.y_train, seed, moco.NEG_PER_POS)
        b_protocol = load_scores(OUT / f"baseline_seed{seed}.npz")
        b_shots = load_scores(OUT / f"baseline_shots_seed{seed}.npz")    # read-only: cache written by exp_noise_shots
        self.base = {p: ((b_protocol if p in b_protocol else b_shots)[p][0][rv],
                         (b_protocol if p in b_protocol else b_shots)[p][1][rt]) for p in P_LEVELS}
        self.base_overlap_error = max(float(np.abs(b_protocol[p][i] - b_shots[p][i]).max())
                                      for p in b_protocol for i in (0, 1))
        self._moco = {}

    def _subset(self, scores):
        return {p: (v[self.rv], t[self.rt]) for p, (v, t) in scores.items()}

    def moco(self, q):
        if q in self._moco:
            return self._moco[q]
        protocol, protocol_rot = None, None
        if q == 0:
            enc = AngleEncoder(moco.N_QUBITS, self.seed).fit(self.X_b)
            path = OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{self.seed}.npz"
            data = np.load(path)
            rot, w = data["rot"], data["w"]
            protocol, protocol_rot = self._subset(load_scores(path)), rot
        else:
            enc = RobustAngleEncoder(moco.N_QUBITS, self.seed, q).fit(self.X_b)
            path = OUT / f"adaptive_moco_params_q{q:g}_seed{self.seed}.npz"
            if not path.exists():
                params, C = moco.fit(enc.transform(self.X_b), self.y_b, self.seed)
                np.savez(path, rot=params["rot"], w=params["w"], C=C)
            data = np.load(path)
            rot, w = data["rot"], data["w"]
            official = OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_q{q:g}_seed{self.seed}.npz"   # exp_moco --q
            if official.exists():
                protocol, protocol_rot = self._subset(load_scores(official)), np.load(official)["rot"]
        th, ph = -rot[:, 1], -rot[:, 0]
        axes = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        Z_val, Z_test = enc.transform(self.X_val), enc.transform(self.X_test)
        zq_val, zq_test = [(moco.bloch_exact(Z) * axes[None]).sum(-1) for Z in (Z_val, Z_test)]
        t0 = float(choose_threshold(self.y_val, (1 - zq_val @ w) / 2))
        self._moco[q] = MocoModel(q, Z_val, zq_val, zq_test, rot, w, t0, protocol, protocol_rot)
        return self._moco[q]


def problem(d, model, p, part):
    """(P(z_q = +1) (N, Q), weights, exact score, threshold t_p) of a model at p on 'val' or 'test'."""
    if model == BASELINE:
        s = d.base[p][0 if part == "val" else 1]
        return (1 - s)[:, None], W_BASELINE, s, float(choose_threshold(d.y_val, d.base[p][0]))
    m = d.moco(q_of(model))
    zq = m.zq_val if part == "val" else m.zq_test
    return (1 + lam2(p) * zq) / 2, m.w, (1 - lam2(p) * zq @ m.w) / 2, float(moco.transfer_threshold(m.t0, p, K_GATES))


# ------------------------------------------------------------------ metrics of one scored set
def summarise(y, score, t, exact, shots, executions, decided=None, fraud=None):
    """Ranking metrics from the scores, threshold metrics from the decisions (score >= t unless given)."""
    fraud = score >= t if fraud is None else fraud
    rank = evaluate(y, score, t)
    dec = evaluate(y, fraud.astype(float), 0.5)
    flip = fraud != (exact >= t)
    row = {"AUPRC": rank["AUPRC"], "AUROC": rank["AUROC"],
           **{k: dec[k] for k in ("precision", "recall", "F1", "MCC")},
           "threshold": t, "mean shots": float(np.mean(shots)), "shots legit": float(np.mean(shots[y == 0])),
           "shots fraud": float(np.mean(shots[y == 1])), "executions per transaction": float(np.mean(executions)),
           "decision flip rate": float(np.mean(flip)), "Spearman rho vs exact": float(spearmanr(exact, score).statistic)}
    if decided is not None:
        row.update({"capped": float(1 - decided.mean()), "wrong early decisions": float(np.mean(decided & flip))})
    return row


def run_triage(P, w, t, cfg, caps, rng, y, exact):
    ci, b, alpha = cfg
    res = ad.triage(P, w, 1 - 2 * t, b, alpha, caps, ci, rng)
    return {cap: (r, summarise(y, score_of(r["mean"]), t, exact, r["shots"], r["shots"] // b, r["decided"],
                               r["fraud"])) for cap, r in res.items()}


# ------------------------------------------------------------------ stage 1: grid on validation at p = SELECT_P
def grid_seed(seed, smoke):
    cache = OUT / f"adaptive_val_grid_seed{seed}{suffix(smoke)}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    d = SeedData(seed, smoke)
    rows = []
    for mi, model in enumerate([*map(moco_name, Q_GRID), BASELINE]):
        P, w, exact, t = problem(d, model, SELECT_P, "val")
        fixed = ad.fixed_shot_mean(P, w, S_FIXED, np.random.default_rng([seed, mi, 0]))
        rows.append(dict(model=model, method="fixed 1024", S=S_FIXED,
                         **summarise(d.y_val, score_of(fixed), t, exact, np.full(len(P), S_FIXED), np.ones(len(P)))))
        rows.append(dict(model=model, method="exact", **summarise(d.y_val, exact, t, exact, np.zeros(len(P)),
                                                                  np.zeros(len(P)))))
        for j, cfg in enumerate(GRID):
            for cap, (_, row) in run_triage(P, w, t, cfg, CAPS, np.random.default_rng([seed, mi, j, 1]), d.y_val,
                                            exact).items():
                rows.append(dict(model=model, method="adaptive", ci=cfg[0], b=cfg[1], alpha=cfg[2], cap=cap, **row))
    df = pd.DataFrame(rows).assign(seed=seed, p=SELECT_P)
    df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------------ stage 2: selection (validation only)
def _pick(g):
    tied = g[g.F1 >= g.F1.max() - F1_TIE]
    return tied["mean shots"].idxmin()


def select(grid):
    m = grid[grid.method == "adaptive"].groupby(["model", "ci", "b", "alpha", "cap"])[
        ["F1", "mean shots", "AUPRC", "capped", "wrong early decisions"]].agg(["mean", "std"])
    m.columns = [f"{a} {b_}" if b_ == "std" else a for a, b_ in m.columns]
    m = m.reset_index()
    head = m.loc[_pick(m[m.model != BASELINE])]
    robust = m.loc[_pick(m[m.model.str.startswith("MOCO robust")])]
    models = list(dict.fromkeys(["MOCO kit", head.model, robust.model, BASELINE]))
    chosen = {}
    m["selected"] = False
    for model in models:
        for cap in CAPS:
            idx = _pick(m[(m.model == model) & (m.cap == cap)])
            m.loc[idx, "selected"] = True
            chosen[(model, cap)] = (m.loc[idx, "ci"], int(m.loc[idx, "b"]), float(m.loc[idx, "alpha"]))
    m["headline"] = m.selected & (m.model == head.model) & (m.cap == head.cap)
    ref = grid[grid.method.isin(["fixed 1024", "exact"])].groupby(["model", "method"])[["F1", "AUPRC"]].mean()
    return chosen, models, head.model, int(head.cap), m, ref.reset_index()


def signature(*objs):
    return repr([sorted(o.items()) if isinstance(o, dict) else o for o in objs])


# ------------------------------------------------------------------ stage 2b: chosen configurations on validation at every p
def budget_seed(seed, chosen, smoke):
    sig = signature(chosen)
    frames, d = [], None
    for i, p in enumerate(P_LEVELS):
        cache = OUT / f"adaptive_val_chosen_seed{seed}_p{p}{suffix(smoke)}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            if (df.signature == sig).all():
                frames.append(df)
                continue
        d = d or SeedData(seed, smoke)
        rows = []
        for (model, cap), cfg in chosen.items():
            P, w, exact, t = problem(d, model, p, "val")
            _, row = run_triage(P, w, t, cfg, [cap], np.random.default_rng([seed, i, cap, 7]), d.y_val, exact)[cap]
            rows.append(dict(model=model, cap=cap, ci=cfg[0], b=cfg[1], alpha=cfg[2], **row))
        df = pd.DataFrame(rows).assign(seed=seed, p=p, signature=sig)
        df.to_csv(cache, index=False)
        frames.append(df)
    return pd.concat(frames)


# ------------------------------------------------------------------ stage 3: final evaluation on test
def test_seed(seed, chosen, budget, models, smoke):
    sig = signature(chosen, budget, models)
    d = None
    frames = []
    for i, p in enumerate(P_LEVELS):
        csv = OUT / f"adaptive_test_seed{seed}_p{p}{suffix(smoke)}.csv"
        npz = OUT / f"adaptive_test_seed{seed}_p{p}{suffix(smoke)}.npz"
        if csv.exists() and npz.exists() and str(np.load(npz)["signature"]) == sig:
            frames.append(pd.read_csv(csv))
            continue
        d = d or SeedData(seed, smoke)
        rows, arrays = [], []
        for mi, model in enumerate(models):
            P, w, exact, t = problem(d, model, p, "test")
            N, y = len(P), d.y_test

            def add(method, score, shots, executions, decided=None, fraud=None, **meta):
                rows.append(dict(model=model, method=method, **meta,
                                 **summarise(y, score, t, exact, shots, executions, decided, fraud)))
                if method in KEEP:
                    arrays.append((f"{model} | {method} | {meta.get('cap', '')}", score, shots, executions,
                                   np.ones(N, bool) if decided is None else decided))

            add("exact", exact, np.zeros(N), np.zeros(N))
            rng = np.random.default_rng([seed, i, 3, mi])
            for S in S_CURVE:
                add("fixed 1024" if S == S_FIXED else "fixed", score_of(ad.fixed_shot_mean(P, w, S, rng)),
                    np.full(N, S), np.ones(N), S=S)
            for cap in CAPS:
                ci, b, alpha = chosen[(model, cap)]
                meta = dict(cap=cap, ci=ci, b=b, alpha=alpha)
                if model != BASELINE:
                    S = budget[(model, cap, p)]
                    add("fixed same budget", score_of(ad.fixed_shot_mean(P, w, S, rng)), np.full(N, S), np.ones(N),
                        S=S, cap=cap)
                r = ad.triage(P, w, 1 - 2 * t, b, alpha, [cap], ci, np.random.default_rng([seed, i, 4, mi, cap]))[cap]
                add("adaptive", score_of(r["mean"]), r["shots"], r["shots"] // b, r["decided"], r["fraud"], **meta)
                # refine: flagged rows topped up to 1024 shots for a finer score; the decision stays the triage one
                shots2, mean2, extra = ad.refine(P, w, r["shots"], r["mean"], r["fraud"], S_FIXED,
                                                 np.random.default_rng([seed, i, 5, mi, cap]))
                add("adaptive + refine", score_of(mean2), shots2, r["shots"] // b + extra, r["decided"], r["fraud"],
                    **meta)
        df = pd.DataFrame(rows).assign(seed=seed, p=p)
        df.to_csv(csv, index=False)
        np.savez_compressed(npz, signature=np.array(sig), labels=np.array([a[0] for a in arrays]),
                            scores=np.stack([a[1] for a in arrays]).astype(np.float32),
                            shots=np.stack([a[2] for a in arrays]).astype(np.int16),
                            executions=np.stack([a[3] for a in arrays]).astype(np.int16),
                            decided=np.stack([a[4] for a in arrays]), y=d.y_test.astype(np.int8))
        frames.append(df)
    return pd.concat(frames)


# ------------------------------------------------------------------ Part B: noise meter
def _f1(y, score, t):
    pred = score >= t
    return 2 * np.sum(pred & (y == 1)) / max(pred.sum() + (y == 1).sum(), 1)


def noise_meter_seed(seed, model, smoke, reps):
    csv = OUT / f"noise_meter_seed{seed}_{slug(model)}{suffix(smoke)}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    d = SeedData(seed, smoke)
    m = d.moco(q_of(model))
    s0_val = (1 - m.zq_val @ m.w) / 2
    rows, phats = [], {}
    for i, p in enumerate(METER_P):
        P_val, w, exact_val, _ = problem(d, model, p, "val")
        exact_test = problem(d, model, p, "test")[2]
        t_proto = float(choose_threshold(d.y_val, exact_val))          # full noisy validation, infinite shots
        f1_proto = _f1(d.y_test, exact_test, t_proto)
        for n_cal in METER_NCAL:
            for S in METER_S:
                rng = np.random.default_rng([seed, i, n_cal, S])
                est = {"median ratio": [], "least squares": []}
                for _ in range(reps):
                    idx = rng.choice(len(P_val), n_cal, replace=False)
                    s_hat = score_of(ad.fixed_shot_mean(P_val[idx], w, S, rng))
                    est["median ratio"].append(ad.p_from_lambda2(ad.lambda2_median(s_hat, s0_val[idx]), K_GATES))
                    est["least squares"].append(ad.p_from_lambda2(ad.lambda2_least_squares(s_hat, s0_val[idx]),
                                                                  K_GATES))
                for name, ph in est.items():
                    ph = np.array(ph)
                    phats[f"{p}_{n_cal}_{S}_{name}"] = ph
                    f1 = np.array([_f1(d.y_test, exact_test, moco.transfer_threshold(m.t0, q, K_GATES)) for q in ph])
                    rmse = np.sqrt(np.mean((ph - p) ** 2))
                    rows.append({"model": model, "seed": seed, "p": p, "N_cal": n_cal, "S": S, "estimator": name,
                                 "p_hat mean": ph.mean(), "bias": ph.mean() - p, "RMSE": rmse, "relative RMSE": rmse / p,
                                 "F1 (p_hat)": f1.mean(), "F1 (p_hat) std": f1.std(),
                                 "F1 (full noisy validation)": f1_proto, "F1 drop": f1_proto - f1.mean(),
                                 "same F1 fraction": float(np.mean(np.isclose(f1, f1_proto))),
                                 "calibration executions": n_cal, "calibration shots": n_cal * S})
    np.savez(OUT / f"noise_meter_seed{seed}_{slug(model)}{suffix(smoke)}.npz", **phats)
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return df


def meter_choice(meter):
    """Estimator and N_cal for the ledger, from the p_hat errors only (calibration rows from validation, no test)."""
    m = meter.groupby(["estimator", "p", "N_cal", "S"])["relative RMSE"].mean().reset_index()
    estimator = m.groupby("estimator")["relative RMSE"].mean().idxmin()
    g = m[(m.estimator == estimator) & (m.S == 1024)].groupby("N_cal")["relative RMSE"].max()
    ok = g[g <= METER_REL_RMSE]
    return estimator, int(ok.index.min()) if len(ok) else int(g.index.max())


# ------------------------------------------------------------------ checks
def tracker_triage(m, p, cfg, cap, n_rows=5):
    """Triage on the kit's noisy device with real PennyLane sampling (b shots per execution); qml.Tracker vs formula."""
    import pennylane as qml
    ci, b, alpha = cfg
    noisy = qml.noise.insert(qml.device("default.mixed", wires=4, seed=0), qml.DepolarizingChannel, p, position="all")

    def circuit(x):
        moco.feature_map(x, moco.FEATURE_MAP)
        for q in range(4):
            qml.Rot(*m.rot[q], wires=q)
        return qml.sample(wires=range(4))

    qnode = qml.set_shots(qml.QNode(circuit, noisy), shots=b)
    t_h = 1 - 2 * float(moco.transfer_threshold(m.t0, p, K_GATES))
    batches = []
    with qml.Tracker(noisy) as tracker:
        for x in m.Z_val[:n_rows]:
            interval, k = ad.AnytimeCI(ci, alpha, 1), 0
            while k * b < cap:
                k += 1
                h = (1 - 2 * np.asarray(qnode(x), dtype=float)) @ m.w
                center, eps = interval.update(np.array([0]), h.sum(), (h * h).sum(), b)
                if abs(center[0] - t_h) > eps[0]:
                    break
            batches.append(k)
    return dict(batches=batches, formula_executions=sum(batches), formula_shots=b * sum(batches),
                tracker_executions=tracker.totals["executions"], tracker_shots=tracker.totals["shots"])


def checks_seed(seed, models, chosen, head_model, head_cap, smoke, reps):
    import pennylane as qml
    d = SeedData(seed, smoke)
    out = []

    def add(check, err, limit):
        out.append(dict(seed=seed, check=check, max_error=float(err), limit=limit,
                        status="PASS" if err <= limit else "FAIL"))

    for model in [mo for mo in models if mo != BASELINE]:
        m = d.moco(q_of(model))
        if m.protocol is not None:
            add(f"{model}: circuit used here == circuit of the protocol run", np.abs(m.rot - m.protocol_rot).max(),
                1e-9)
            for p, (v, t) in m.protocol.items():
                err = max(np.abs(problem(d, model, p, "val")[2] - v).max(), np.abs(problem(d, model, p, "test")[2] - t).max())
                add(f"{model}, all rows, p={p}: lambda^2 <H>_0 vs kit noisy_predict (cached protocol scores)", err, 1e-9)
                if p > 0:
                    add(f"{model}, p={p}: transfer_threshold vs max-F1 threshold on noisy validation",
                        abs(moco.transfer_threshold(m.t0, p, K_GATES) - choose_threshold(d.y_val, v)), 1e-9)
        params = dict(rot=m.rot, w=m.w, fm=moco.FEATURE_MAP)
        Z = m.Z_val[:N_CHECK]
        for p in (0.05, 0.2):
            add(f"{model}, {N_CHECK} rows, p={p}: lambda^2 <H>_0 vs kit noisy_predict",
                np.abs(problem(d, model, p, "val")[2][:N_CHECK] - noisy_predict(Z, None, p, moco.make_circuit(params)))
                .max(), 1e-9)

        def probs_circuit(x, weights=None):
            moco.feature_map(x, moco.FEATURE_MAP)
            for q in range(4):
                qml.Rot(*m.rot[q], wires=q)
            return qml.probs(wires=range(4))

        joint = np.asarray(noisy_predict(Z, None, 0.2, probs_circuit, lambda v: v))
        up = problem(d, model, 0.2, "val")[0][:N_CHECK]
        marg = np.stack([up, 1 - up], -1)
        product = np.einsum("ni,nj,nk,nl->nijkl", *[marg[:, q] for q in range(4)]).reshape(N_CHECK, 16)
        add(f"{model}, {N_CHECK} rows, p=0.2: joint probs vs product of marginals", np.abs(joint - product).max(), 1e-9)

    P, w, _, t = problem(d, "MOCO kit", 0.05, "val")
    P = P[:N_CHECK]
    tiled = np.tile(P, (reps, 1))
    rng = np.random.default_rng([seed, 99])
    seq = ad.triage(tiled, w, 1 - 2 * t, 16, 0.01, [S_FIXED], "plugin-bernstein", rng, stop=False)[S_FIXED]
    fixed = ad.fixed_shot_mean(tiled, w, S_FIXED, rng)
    mu = (2 * P - 1) @ w
    var = ((1 - (2 * P - 1) ** 2) * w ** 2).sum(1) / S_FIXED
    add(f"{N_CHECK} rows, stopping disabled: every row uses S_max shots", np.abs(seq["shots"] - S_FIXED).max(), 0)
    for name, est in [("triage without stopping", seq["mean"]), ("fixed S", fixed)]:
        mm = est.reshape(reps, -1)
        add(f"{N_CHECK} rows x {reps} reps, p=0.05, S={S_FIXED}: {name}, max |z| of the mean vs exact",
            np.abs((mm.mean(0) - mu) / np.sqrt(var / reps)).max(), 5.0)
        add(f"{N_CHECK} rows x {reps} reps: {name}, |variance / analytic variance - 1| (mean over rows)",
            abs((mm.var(0, ddof=1) / var).mean() - 1), 0.05)
    a, b_ = seq["mean"].reshape(reps, -1), fixed.reshape(reps, -1)
    add(f"{N_CHECK} rows: triage without stopping vs fixed S, max |z| of the difference of means",
        np.abs((a.mean(0) - b_.mean(0)) / np.sqrt(2 * var / reps)).max(), 5.0)
    add("baseline: cached protocol scores vs exp_noise_shots cache (common p)", d.base_overlap_error, 1e-9)

    cfg = chosen[(head_model, head_cap)]
    tr = tracker_triage(d.moco(q_of(head_model)), 0.05, cfg, head_cap)
    add(f"Tracker, {head_model}, 5 rows, p=0.05, b={cfg[1]}: executions vs formula sum_i K_i "
        f"({tr['formula_executions']})", abs(tr["tracker_executions"] - tr["formula_executions"]), 0)
    add(f"Tracker, 5 rows: shots vs formula b * sum_i K_i ({tr['formula_shots']})",
        abs(tr["tracker_shots"] - tr["formula_shots"]), 0)
    return pd.DataFrame(out), tr


# ------------------------------------------------------------------ tables
def mean_std_table(df, keys, cols):
    g = df.groupby(keys, dropna=False)[cols]
    return g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)


def ledger_table(test, head_model, head_cap, meter_rows, n_test, n_val, tr):
    rows = []
    sel = test[test.method.isin(["fixed 1024"])
               | (test.method.isin(["adaptive", "adaptive + refine", "fixed same budget"]) & (test.cap == head_cap))]
    formulas = {"fixed 1024": "1 execution x 1024 shots",
                "fixed same budget": "1 execution x S (S = adaptive mean shots on validation)",
                "adaptive": "sum_i K_i executions, b * sum_i K_i shots (K_i = batches of transaction i)",
                "adaptive + refine": "sum_i K_i + #flagged executions; flagged topped up to 1024 shots"}
    for (model, method, p), g in sel.groupby(["model", "method", "p"]):
        shots, ex = g["mean shots"].mean(), g["executions per transaction"].mean()
        b = f" (b = {int(g.b.iloc[0])})" if method.startswith("adaptive") else ""
        rows.append({"item": f"{model}: inference, {method}", "p": p, "formula": formulas[method] + b,
                     "inference executions per transaction": round(ex, 3),
                     "inference shots per transaction": round(shots, 1),
                     "test executions": int(round(ex * n_test)), "test shots": int(round(shots * n_test))})
    rows.append({"item": "threshold calibration: noisy validation pass (protocol)", "p": "any",
                 "formula": f"{n_val:,} executions x 1024 shots", "test executions": n_val, "test shots": n_val * S_FIXED})
    for model, (estimator, n_cal) in meter_rows.items():
        rows.append({"item": f"threshold calibration with the noise meter, {model} ({estimator})", "p": "any",
                     "formula": f"N_cal = {n_cal} executions x 1024 shots, then transfer_threshold(t0, p_hat)",
                     "test executions": n_cal, "test shots": n_cal * S_FIXED})
    led = pd.DataFrame(rows)
    led["Tracker check (5 rows, p=0.05)"] = None
    led.loc[led.item == f"{head_model}: inference, adaptive", "Tracker check (5 rows, p=0.05)"] = (
        f"executions {tr['tracker_executions']} = formula {tr['formula_executions']}, "
        f"shots {tr['tracker_shots']} = formula {tr['formula_shots']}")
    return led


# ------------------------------------------------------------------ figures
INK, MUTED, GRID_C, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#1baf7a", "#eb6834"]           # validated: MOCO kit, MOCO robust, baseline (fixed order)
CLASS_COLOR = {"legit": "#008300", "fraud": "#eda100"}


def _style(plt):
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax):
    ax.grid(axis="y", color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(test, models, chosen, head_cap, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    sfx = suffix(smoke)
    seeds = sorted(test.seed.unique())
    colors = dict(zip(models, SERIES if len(models) == 3 else [SERIES[0], SERIES[2]]))

    # (a) shots per transaction at p = 0.05, legit vs fraud, pooled over seeds
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=True)
    edges = 16 * 2.0 ** (np.arange(0, np.log2(head_cap / 16) + 2) - 0.5)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for ax, model in zip(axes, models):
        shots = {"legit": [], "fraud": []}
        for s in seeds:
            data = np.load(OUT / f"adaptive_test_seed{s}_p0.05{sfx}.npz")
            k = list(data["labels"]).index(f"{model} | adaptive | {head_cap}")
            shots["legit"].append(data["shots"][k][data["y"] == 0])
            shots["fraud"].append(data["shots"][k][data["y"] == 1])
        for j, (cls, vals) in enumerate(shots.items()):
            v = np.concatenate(vals)
            frac = np.histogram(v, edges)[0] / len(v)
            ax.bar(np.arange(len(centers)) + (j - 0.5) * 0.38, frac, width=0.35, color=CLASS_COLOR[cls],
                   label=f"{cls} (mean {v.mean():.0f} shots)")
        ci, b, alpha = chosen[(model, head_cap)]
        ax.set_xticks(np.arange(len(centers)), [f"{int(round(c))}" for c in centers], fontsize=8)
        ax.set_xlabel("shots used by the transaction (log2 bins)")
        ax.set_title(f"{model}\n{ci}, b = {b}, alpha = {alpha:g}", fontsize=9.5, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        _clean(ax)
    axes[0].set_ylabel("fraction of the class")
    fig.suptitle(f"Adaptive triage at p = 0.05, S_max = {head_cap}: shots per test transaction "
                 f"({len(seeds)} seeds pooled)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"adaptive_shots_histogram{sfx}.png", dpi=160)
    plt.close(fig)

    # (b) F1 vs mean shots per transaction, one panel per p
    fig, axes = plt.subplots(1, len(P_LEVELS), figsize=(4.2 * len(P_LEVELS), 4.6), sharey=True)
    for ax, p in zip(axes, P_LEVELS):
        t = test[test.p == p]
        for model, color in colors.items():
            g = t[(t.model == model) & t.method.isin(["fixed", "fixed 1024"])].groupby("S")
            ax.errorbar(g["mean shots"].mean(), g.F1.mean(), yerr=g.F1.std(), color=color, linewidth=2, marker="o",
                        markersize=4, label=f"{model}, fixed S")
            for cap, marker in zip(CAPS, ("D", "s")):
                a = t[(t.model == model) & (t.method == "adaptive") & (t.cap == cap)]
                ax.errorbar(a["mean shots"].mean(), a.F1.mean(), yerr=a.F1.std(), xerr=a["mean shots"].std(),
                            color=color, marker=marker, markersize=9, markerfacecolor=SURFACE, markeredgewidth=2,
                            linestyle="none", label=f"{model}, adaptive S_max = {cap}")
            ax.axhline(t[(t.model == model) & (t.method == "exact")].F1.mean(), color=color, linewidth=1,
                       linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_title(f"p = {p:g}", fontsize=10, color=INK)
        ax.set_xlabel("mean shots per test transaction")
        _clean(ax)
    axes[0].set_ylabel("test F1 (threshold fixed before test)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(colors))
    fig.suptitle(f"F1 vs shot budget: adaptive triage vs fixed shots (mean ± std over {len(seeds)} seeds; "
                 "dotted = exact expectation values)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(OUT / f"adaptive_shots_f1_vs_shots{sfx}.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ main
def _pool_map(fn, args, jobs):
    if jobs == 1:
        return [fn(*a) for a in args]
    with Pool(jobs) as pool:
        return pool.starmap(fn, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--jobs", type=int, default=len(SEEDS))
    args = ap.parse_args()
    smoke, sfx = args.smoke, suffix(args.smoke)
    seeds = [0, 1] if smoke else SEEDS
    reps = 50 if smoke else 200
    jobs = 1 if smoke else args.jobs
    OUT.mkdir(parents=True, exist_ok=True)

    grid = pd.concat(_pool_map(grid_seed, [(s, smoke) for s in seeds], jobs))
    chosen, models, head_model, head_cap, sel, ref = select(grid)
    sel.to_csv(OUT / f"adaptive_selection_on_validation{sfx}.csv", index=False)
    val = pd.concat(_pool_map(budget_seed, [(s, chosen, smoke) for s in seeds], jobs))
    budget = {(mo, int(c), float(p)): max(1, int(round(s)))
              for (mo, c, p), s in val.groupby(["model", "cap", "p"])["mean shots"].mean().items()}
    test = pd.concat(_pool_map(test_seed, [(s, chosen, budget, models, smoke) for s in seeds], jobs))
    test.to_csv(OUT / f"adaptive_shots{sfx}.csv", index=False)
    if args.plot:
        plot(test, models, chosen, head_cap, smoke)
        return
    moco_models = [mo for mo in models if mo != BASELINE]
    meter = pd.concat(_pool_map(noise_meter_seed, [(s, mo, smoke, reps) for mo in moco_models for s in seeds], jobs))
    meter.to_csv(OUT / f"noise_meter{sfx}.csv", index=False)
    meter_rows = {mo: meter_choice(meter[meter.model == mo]) for mo in moco_models}
    checks, tr = checks_seed(seeds[0], models, chosen, head_model, head_cap, smoke, reps)
    for _, r in test[test.method == "adaptive"].iterrows():
        checks.loc[len(checks)] = dict(seed=r.seed, check=f"{r.model}, p={r.p}, S_max={int(r.cap)}: wrong early "
                                       "decisions per transaction <= alpha", max_error=r["wrong early decisions"],
                                       limit=r.alpha, status="PASS" if r["wrong early decisions"] <= r.alpha else "FAIL")
    checks.to_csv(OUT / f"adaptive_checks{sfx}.csv", index=False)
    n_test = len(np.load(OUT / f"adaptive_test_seed{seeds[0]}_p0.0{sfx}.npz")["y"])
    led = ledger_table(test, head_model, head_cap, meter_rows, n_test, 56_962, tr)
    led.to_csv(OUT / f"ledger_adaptive{sfx}.csv", index=False)
    plot(test, models, chosen, head_cap, smoke)

    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_rows", 400,
                  "display.max_colwidth", 100)
    print(f"=== selection on validation, p = {SELECT_P}: best (interval, b, alpha) per (model, S_max)")
    show = ["model", "cap", "ci", "b", "alpha", "F1", "F1 std", "mean shots", "AUPRC", "capped", "selected", "headline"]
    print(sel[sel.selected][show].round(4).to_string(index=False))
    print("\nvalidation references at p = 0.05 (mean over seeds):\n", ref.round(4).to_string(index=False))
    print("\nheadline:", head_model, "S_max", head_cap, "\nsame-budget S (validation):",
          {k: v for k, v in budget.items() if k[0] != BASELINE})
    cols = ["mean shots", "shots legit", "shots fraud", "executions per transaction", "F1", "MCC", "recall", "AUPRC",
            "capped", "wrong early decisions"]
    view = test[test.method.isin(["exact", "fixed 1024", "fixed same budget", "adaptive", "adaptive + refine"])]
    print(f"\n=== test, mean ± std over {len(seeds)} seeds")
    print(mean_std_table(view, ["model", "p", "method", "cap"], cols).to_string())
    print("\n=== noise meter (mean over seeds)")
    print(meter.groupby(["model", "estimator", "p", "N_cal", "S"])[
        ["p_hat mean", "bias", "relative RMSE", "F1 (p_hat)", "F1 (full noisy validation)"]].mean().round(4)
        .to_string())
    print("ledger choice:", meter_rows)
    print("\n=== ledger\n", led.drop(columns="formula").to_string(index=False))
    print(f"\n=== checks: {checks.status.value_counts().to_dict()}")
    print(checks[~checks.check.str.contains("wrong early")].to_string(index=False))


if __name__ == "__main__":
    main()
