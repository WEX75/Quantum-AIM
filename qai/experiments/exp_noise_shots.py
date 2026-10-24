"""The mentor's question: what happens when the noise grows? The protocol scores circuits with exact expectation
values (infinite shots). There, depolarizing noise on MOCO only compresses the scores towards 1/2 and cannot change
the ranking. On hardware every score is estimated from S shots, so the compression has to be paid for in shots.

Hypotheses (written before the run):
  H1  Exact expectation values: MOCO's AUPRC and Spearman rho are flat in p up to 0.5 and its effective compression
      is exactly lambda^2 = (1 - 4p/3)^2. The baseline's compression falls much faster (126 noise locations) and its
      rho decays; R2 (18 noise locations) sits in between, with rho slightly below 1.
  H2  Finite shots: the AUPRC loss follows the compression. MOCO needs about lambda^-4 times more shots to keep its
      noiseless AUPRC (1.3x at p = 0.05, 81x at p = 0.5), so from S = 1024 on it stays within 0.01 of its noiseless
      AUPRC up to p ~ 0.2. The baseline leaves that 0.01 band at a much smaller p, for every S on the grid.
  H3  The shots-needed table is the same on validation and on test for MOCO (same compression on both sets).

Shot simulation, exact per model:
  baseline VQC  readout <Z_0>: shots on qubit 0 are Binomial(S, P(1)), with P(1) = (1 - <Z_0>_p) / 2 the score
  MOCO          the final state is a product state, so each Z_q is an independent Binomial with mean lambda^2 <Z_q>
                (checked on 200 rows against the kit's noisy_predict, and the joint distribution against the
                product of the marginals)
  R2            qml.probs on default.mixed with the kit's noise insertion (noisy_qnode), then multinomial sampling
                of the joint 16-outcome distribution
Grid: p in P_LEVELS x S in SHOTS (+ exact), full validation and test sets, 5 seeds. F1 and MCC use the threshold
that maximises F1 on validation scores at the SAME (p, S), validation also sampled with S shots.

shots_needed.csv is a DIAGNOSTIC: measured on test (and on validation, to check that they agree), not used for any
choice. No model, threshold or parameter is selected with it.

    python3 -m qai.experiments.exp_noise_shots            # full run (background, ~40 min)
    python3 -m qai.experiments.exp_noise_shots --smoke    # seed 0, all frauds + 3000 legit rows, outputs *_fast
    python3 -m qai.experiments.exp_noise_shots --plot     # redraw the figures from noise_shots.csv
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from qai import kit_vqc as kit
from qai import moco
from qai.experiments import exp_entangled as r2
from qai.protocol import (SEEDS, AngleEncoder, choose_threshold, clean_predict, common_split, evaluate, load_df,
                          load_scores, noisy_predict, save_scores, undersample)

OUT = Path("results/task2")
P_LEVELS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
SHOTS = [64, 256, 1024, 4096, 16384]
R2_VARIANTS = ["random", "moco", "moco-noisy"]
MODELS = ["baseline VQC", "MOCO", *[f"R2 ({v})" for v in R2_VARIANTS]]
DIAGNOSTIC = "diagnostic, measured on test, not used for any choice"
# Z_q eigenvalue (+1 / -1) of each of the 16 outcomes of qml.probs(wires=[0, 1, 2, 3]); wire 0 is the leading bit
SIGNS = 1 - 2 * ((np.arange(16)[:, None] >> (3 - np.arange(4))[None, :]) & 1)


def lam(p):
    return 1 - 4 * p / 3


def identity(x):
    return x


def effective_compression(s_p, s_0):
    """Median over transactions of |s_p - 1/2| / |s_0 - 1/2| (1 = no compression)."""
    d0 = np.abs(s_0 - 0.5)
    keep = d0 > 1e-9
    return float(np.median(np.abs(s_p - 0.5)[keep] / d0[keep]))


# ------------------------------------------------------------------ the three ways a readout is sampled
class QubitZeroBinomial:
    """Baseline: the score is P(qubit 0 reads 1)."""

    def __init__(self, scores):
        self.scores = scores                       # {p: (val, test)}

    def exact(self, p, part):
        return self.scores[p][part]

    def sample(self, p, part, S, rng):
        return rng.binomial(S, self.scores[p][part]) / S


class ProductBinomials:
    """MOCO: independent qubits, <Z_q>_p = lambda^2 <Z_q>_0, readout (1 - sum_q w_q Z_q) / 2."""

    def __init__(self, zq, w):
        self.zq, self.w = zq, w                    # zq = (val (N, 4), test (N, 4)), noiseless

    def exact(self, p, part):
        return (1 - lam(p) ** 2 * self.zq[part] @ self.w) / 2

    def sample(self, p, part, S, rng):
        z = 2 * rng.binomial(S, (1 + lam(p) ** 2 * self.zq[part]) / 2) / S - 1
        return (1 - z @ self.w) / 2


class JointMultinomial:
    """R2: the full 16-outcome distribution under the kit's noise model, readout (1 - sum_q w_q Z_q) / 2."""

    def __init__(self, probs, w):
        self.probs, self.w = probs, w              # {p: (val (N, 16), test (N, 16))}

    def exact(self, p, part):
        return (1 - (self.probs[p][part] @ SIGNS) @ self.w) / 2

    def sample(self, p, part, S, rng):
        counts = rng.multinomial(S, self.probs[p][part])
        return (1 - (counts @ SIGNS / S) @ self.w) / 2


# ------------------------------------------------------------------ loading the trained circuits of each model
def _rows(split, smoke):
    """Row indices of (validation, test): all of them, or all frauds + 3000 legit rows for a smoke run."""
    def pick(y):
        if not smoke:
            return np.arange(len(y))
        return np.sort(np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)[:3000]])
    return pick(split.y_val), pick(split.y_test)


def _encode(split, seed, neg_per_pos, rows):
    X_b, _ = undersample(split.X_train, split.y_train, seed, neg_per_pos)
    enc = AngleEncoder(4, seed).fit(X_b)
    return enc.transform(split.X_val.iloc[rows[0]]), enc.transform(split.X_test.iloc[rows[1]])


def _normalised(probs):
    probs = np.clip(np.asarray(probs, dtype=float), 0, None)
    return probs / probs.sum(1, keepdims=True)


def baseline_source(seed, split, rows, smoke):
    cache = OUT / f"baseline_shots_seed{seed}.npz"
    if not smoke and cache.exists():
        return QubitZeroBinomial(load_scores(cache)), []
    Z_val, Z_test = _encode(split, seed, 1, rows)
    w = torch.tensor(np.load(OUT / f"baseline_seed{seed}.npz")["weights"])
    protocol = {} if smoke else load_scores(OUT / f"baseline_seed{seed}.npz")
    scores = {}
    for p in P_LEVELS:
        if p in protocol:
            scores[p] = protocol[p]
        elif p == 0:
            scores[p] = tuple(clean_predict(Z, w, kit.vqc) for Z in (Z_val, Z_test))
        else:
            scores[p] = tuple(noisy_predict(Z, w, p, kit.vqc, chunk=10000) for Z in (Z_val, Z_test))
    if not smoke:
        save_scores(cache, scores)
    return QubitZeroBinomial(scores), []


def moco_source(seed, split, rows, smoke):
    Z_val, Z_test = _encode(split, seed, moco.NEG_PER_POS, rows)
    data = np.load(OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{seed}.npz")
    params = dict(rot=data["rot"], w=data["w"], fm=moco.FEATURE_MAP)
    th, ph = -params["rot"][:, 1], -params["rot"][:, 0]
    axes = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    zq = tuple((moco.bloch_exact(Z) * axes[None]).sum(-1) for Z in (Z_val, Z_test))
    source = ProductBinomials(zq, params["w"])

    checks, Z200 = [], Z_test[:200]

    def probs_circuit(x, weights=None):
        moco.feature_map(x, params["fm"])
        for q in range(4):
            qml.Rot(*params["rot"][q], wires=q)
        return qml.probs(wires=range(4))

    for p in (0.05, 0.5):
        kit_scores = noisy_predict(Z200, None, p, moco.make_circuit(params))
        joint = np.asarray(noisy_predict(Z200, None, p, probs_circuit, identity))
        up = (1 + lam(p) ** 2 * zq[1][:200]) / 2                     # P(Z_q = +1), independent qubits
        marg = np.stack([up, 1 - up], -1)                             # (200, 4, 2)
        product = np.einsum("ni,nj,nk,nl->nijkl", *[marg[:, q] for q in range(4)]).reshape(200, 16)
        checks.append(dict(model="MOCO", seed=seed, check=f"200 rows, p={p}: binomial means vs kit noisy_predict",
                           max_error=float(np.abs(source.exact(p, 1)[:200] - kit_scores).max())))
        checks.append(dict(model="MOCO", seed=seed, check=f"200 rows, p={p}: joint probs vs product of marginals",
                           max_error=float(np.abs(joint - product).max())))
    return source, checks


def r2_source(variant, seed, split, rows, smoke):
    Z_val, Z_test = _encode(split, seed, moco.NEG_PER_POS, rows)
    tag = f"entangled_{variant}_seed{seed}"
    theta = torch.tensor(np.load(OUT / f"{tag}.npz")["theta"])
    w = torch.softmax(theta[r2.N_ROT:], 0).numpy()
    probs = {}
    for p in P_LEVELS:
        if p == 0:
            probs[p] = tuple(_normalised(clean_predict(Z, theta, r2.probs_circuit, identity)) for Z in (Z_val, Z_test))
        else:
            probs[p] = tuple(_normalised(noisy_predict(Z, theta, p, r2.probs_circuit, identity, chunk=20000))
                             for Z in (Z_val, Z_test))
    source = JointMultinomial(probs, w)
    checks = []
    if not smoke:                                  # same circuit, same noise: must match the protocol scores
        protocol = load_scores(OUT / f"{tag}.npz")
        err = max(float(np.abs(source.exact(p, part) - protocol[p][part]).max())
                  for p in protocol for part in (0, 1))
        checks.append(dict(model=f"R2 ({variant})", seed=seed, check="probs readout vs protocol scores (all p)",
                           max_error=err))
    return source, checks


# ------------------------------------------------------------------ one (model, seed) job
def run_job(model, seed, smoke=False):
    torch.set_num_threads(2)
    split = common_split(load_df())
    rows = _rows(split, smoke)
    y_val, y_test = split.y_val[rows[0]], split.y_test[rows[1]]
    if model == "baseline VQC":
        source, checks = baseline_source(seed, split, rows, smoke)
    elif model == "MOCO":
        source, checks = moco_source(seed, split, rows, smoke)
    else:
        source, checks = r2_source(model[4:-1], seed, split, rows, smoke)

    clean_test = source.exact(0.0, 1)
    out = []
    for i, p in enumerate(P_LEVELS):
        for S in [np.inf, *SHOTS]:
            if np.isinf(S):
                s_val, s_test = source.exact(p, 0), source.exact(p, 1)
            else:
                rng = np.random.default_rng([seed, MODELS.index(model), i, int(S)])
                s_val, s_test = source.sample(p, 0, int(S), rng), source.sample(p, 1, int(S), rng)
            threshold = choose_threshold(y_val, s_val)
            row = dict(model=model, seed=seed, p=p, shots=S, **evaluate(y_test, s_test, threshold),
                       **{"Spearman rho": spearmanr(clean_test, s_test).statistic, "threshold": threshold,
                          "val_AUPRC": average_precision_score(y_val, s_val)})
            if np.isinf(S):
                row["compression"] = effective_compression(s_test, clean_test)
            out.append(row)
    if model == "MOCO":
        err = max(abs(r["compression"] - lam(r["p"]) ** 2) for r in out if np.isinf(r["shots"]))
        checks.append(dict(model=model, seed=seed, check="compression == lambda^2 (all p)", max_error=err))
    return pd.DataFrame(out), checks


def _run_job(args):
    return run_job(*args)


# ------------------------------------------------------------------ tables
def shots_needed(df, tol=0.01):
    """Smallest S whose mean AUPRC stays within tol of the model's exact noiseless AUPRC, on validation and on test.
    DIAGNOSTIC: not used for any choice."""
    out = {}
    for part, col in [("val", "val_AUPRC"), ("test", "AUPRC")]:
        exact = df[np.isinf(df.shots)]
        ref = exact[exact.p == 0].groupby("model")[col].mean()
        m = df[np.isfinite(df.shots)].groupby(["model", "p", "shots"])[col].mean().reset_index()
        m = m[m[col] >= m.model.map(ref) - tol]
        out[f"S_needed_{part}"] = m.groupby(["model", "p"]).shots.min()
    need = pd.DataFrame(out).reindex(pd.MultiIndex.from_product([MODELS, P_LEVELS], names=["model", "p"]))
    need = need.reset_index()
    need["agree"] = need.S_needed_val.fillna(-1) == need.S_needed_test.fillna(-1)
    need["note"] = "diagnostic: within 0.01 of noiseless AUPRC; NaN = not reached by S=16384; not used for any choice"
    return need


def sanity_checks(df, checks):
    rows = []
    for model, g in df[df.p == 0].groupby("model"):
        exact = g[np.isinf(g.shots)].set_index("seed").AUPRC
        top = g[g.shots == SHOTS[-1]].set_index("seed").AUPRC
        gap = float((top - exact).abs().max())
        rows.append(dict(model=model, seed="all", check=f"p=0, S={SHOTS[-1]}: |AUPRC - exact| < 0.01 (worst seed)",
                         max_error=gap))
    out = pd.DataFrame(rows + checks)
    limit = out.check.map(lambda c: 0.01 if "S=" in c else 1e-6)
    out["status"] = np.where(out.max_error < limit, "PASS", "FAIL")
    return out


# ------------------------------------------------------------------ figures (palette: dataviz reference instance)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"MOCO": "#2a78d6", "baseline VQC": "#eb6834", "R2 (moco)": "#1baf7a"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot(df, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    mean = df.groupby(["model", "p", "shots"]).AUPRC.mean()
    cols = [*SHOTS, np.inf]

    # 1. heatmaps: one panel per model, shared scale
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmin, vmax = mean.min(), mean.max()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        grid = np.array([[mean.get((model, p, S), np.nan) for S in cols] for p in P_LEVELS])
        ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(P_LEVELS)):
            for j in range(len(cols)):
                v = grid[i, j]
                dark = (v - vmin) / max(vmax - vmin, 1e-9) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK)
        ax.set_xticks(range(len(cols)), [f"{int(S)}" if np.isfinite(S) else "exact" for S in cols], fontsize=8)
        ax.set_yticks(range(len(P_LEVELS)), [f"{p:g}" for p in P_LEVELS], fontsize=8)
        ax.set_title(model, fontsize=10, color=INK)
        ax.set_xlabel("shots per transaction S", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("depolarizing p per gate", fontsize=9)
    fig.suptitle(f"Test AUPRC (mean over seeds) by noise and shots  |  {DIAGNOSTIC}", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"noise_shots_heatmaps{suffix}.png", dpi=160)
    plt.close(fig)

    # 2. AUPRC vs p at S = 1024 (solid, mean +- std band) and exact (dashed), with the classical reference
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(P_LEVELS))
    ends = {}
    for model, color in SERIES.items():
        g = df[(df.model == model) & (df.shots == 1024)].groupby("p").AUPRC
        m, s = g.mean().reindex(P_LEVELS).to_numpy(), g.std().reindex(P_LEVELS).to_numpy()
        ex = df[(df.model == model) & np.isinf(df.shots)].groupby("p").AUPRC.mean().reindex(P_LEVELS).to_numpy()
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=5, label=f"{model}, S = 1024")
        ax.plot(x, ex, color=color, linewidth=1.2, linestyle="--", label=f"{model}, exact (protocol)")
        ends[model] = m[-1]
    label_y = {}
    for model, y in sorted(ends.items(), key=lambda kv: kv[1]):     # direct labels, pushed apart when they collide
        label_y[model] = max(y, max(label_y.values(), default=-1) + 0.05)
    for model, y in label_y.items():
        ax.annotate(model, (x[-1] + 0.12, y), va="center", fontsize=9, color=INK)
    lr_file = OUT / "moco_per_seed_single_10.csv"
    if lr_file.exists():
        lr = pd.read_csv(lr_file)
        lr = lr[lr.model.str.startswith("logistic")].AUPRC.mean()
        ax.axhline(lr, color=MUTED, linewidth=1.2, linestyle=":")
        ax.annotate(f"logistic regression 1:10 (classical) {lr:.3f}", (0, lr), xytext=(0, 5),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xticks(x, [f"{p:g}" for p in P_LEVELS])
    ax.set_xlabel("depolarizing p per gate (ordinal axis, not to scale)")
    ax.set_ylabel("test AUPRC")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.3, len(P_LEVELS) + 0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    ax.set_title(f"Test AUPRC vs noise at S = 1024 shots (band: ±1 std over 5 seeds)\n{DIAGNOSTIC}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"noise_shots_auprc_vs_p{suffix}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    suffix = "_fast" if args.smoke else ""
    if args.plot:
        plot(pd.read_csv(OUT / f"noise_shots{suffix}.csv"), suffix)
        return
    seeds = [0] if args.smoke else SEEDS
    jobs = [(m, s, args.smoke) for m in MODELS for s in seeds]          # slowest (baseline) first
    with Pool(min(len(jobs), 8)) as pool:
        results = list(pool.imap_unordered(_run_job, jobs))
    df = pd.concat([r[0] for r in results]).sort_values(["model", "seed", "p", "shots"])
    df.to_csv(OUT / f"noise_shots{suffix}.csv", index=False)
    checks = sanity_checks(df, [c for r in results for c in r[1]])
    checks.to_csv(OUT / f"noise_shots_checks{suffix}.csv", index=False)
    need = shots_needed(df)
    need.to_csv(OUT / f"shots_needed{suffix}.csv", index=False)
    plot(df, suffix)

    pd.set_option("display.width", 240, "display.max_columns", None, "display.max_rows", 200)
    exact = df[np.isinf(df.shots)].groupby(["model", "p"])[["AUPRC", "Spearman rho", "compression", "F1", "MCC"]]
    print("=== exact expectation values (shots = inf), extended noise range: mean over seeds\n",
          exact.mean().round(4))
    print(f"\n=== test AUPRC with finite shots, mean over seeds ({DIAGNOSTIC})\n",
          df[np.isfinite(df.shots)].pivot_table(index=["model", "p"], columns="shots", values="AUPRC").round(3))
    print("\n=== F1 / MCC at S = 1024 (threshold chosen on validation sampled at the same p and S)\n",
          df[df.shots == 1024].groupby(["model", "p"])[["F1", "MCC"]].mean().round(3))
    print(f"\n=== shots needed to stay within 0.01 of the noiseless AUPRC ({DIAGNOSTIC})\n",
          need.drop(columns="note").to_string(index=False))
    print("\n=== checks\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()
