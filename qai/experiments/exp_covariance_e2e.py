"""Diagnostics (seed 0, no circuit simulation, new files only):
  1. correlation of V1..V28 on the full training split vs on MOCO's 1:10 undersampled set
     -> results/task2/figures/m_corr_heatmap.png, results/task2/corr_stats.csv
  2. the PCA inside MOCO's encoder: explained_variance_ratio_ and the top loadings
  3. one VALIDATION fraud and one VALIDATION legit transaction followed end to end through MOCO (raw values, PCA
     scores, angles, Bloch vector, Rot angles, weights, <Z_q>, <H>, score, threshold, verdict at p = 0 and at
     p = 0.05 with the analytically transferred threshold) -> results/task2/e2e_example.md
The two transactions are the validation fraud and the validation legit transaction whose noiseless MOCO scores are
the medians of their class (a fixed rule, not chosen by hand). The test set is not used.

    python3 -m qai.experiments.exp_covariance_e2e
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qai import moco
from qai.protocol import AngleEncoder, choose_threshold, common_split, load_df, load_scores, undersample

OUT = Path("results/task2")
SEED = 0
P = 0.05
V = [f"V{i}" for i in range(1, 29)]


def corr_stats(name, X, y):
    c = X[V].corr().to_numpy()
    off = np.abs(c[~np.eye(28, dtype=bool)])
    i, j = np.unravel_index(np.argmax(np.abs(c) - np.eye(28) * 2), c.shape)
    return c, {"set": name, "rows": len(y), "frauds": int(y.sum()), "mean |r| off-diagonal": off.mean(),
               "max |r| off-diagonal": np.abs(c[i, j]), "max pair": f"{V[i]}-{V[j]}", "r of max pair": c[i, j],
               "pairs with |r| > 0.5": int((off > 0.5).sum() // 2)}


def heatmap(c_full, c_moco, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])   # blue <-> grey <-> red
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "figure.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), constrained_layout=True)
    for ax, c, s in zip(axes, [c_full, c_moco], stats):
        im = ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(0, 28, 3), [V[k] for k in range(0, 28, 3)], fontsize=11)
        ax.set_yticks(range(0, 28, 3), [V[k] for k in range(0, 28, 3)], fontsize=11)
        ax.set_title(f"{s['set']}: {s['rows']:,} rows, {s['frauds']} frauds\nmean |r| {s['mean |r| off-diagonal']:.3f}, "
                     f"max |r| {s['max |r| off-diagonal']:.2f} ({s['max pair']})", fontsize=14, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
    fig.suptitle("V1..V28 are uncorrelated on the full data but correlated on the fraud-enriched training set",
                 x=0.01, ha="left", fontsize=16)
    fig.savefig(OUT / "figures" / "m_corr_heatmap.png", dpi=200)
    plt.close(fig)


def fmt(v, d=4):
    return f"{v:.{d}f}"


def main():
    split = common_split(load_df())
    X_b, y_b = undersample(split.X_train, split.y_train, SEED, moco.NEG_PER_POS)
    c_full, s_full = corr_stats("full training split", split.X_train, split.y_train)
    c_moco, s_moco = corr_stats("MOCO 1:10 training set (seed 0)", X_b, y_b)
    pd.DataFrame([s_full, s_moco]).to_csv(OUT / "corr_stats.csv", index=False)
    heatmap(c_full, c_moco, [s_full, s_moco])

    enc = AngleEncoder(moco.N_QUBITS, SEED).fit(X_b)
    cols = list(X_b.columns)
    evr = enc.pca.explained_variance_ratio_
    load = pd.DataFrame(enc.pca.components_, columns=cols, index=[f"PC{k + 1}" for k in range(4)])
    top = {pc: load.loc[pc].abs().sort_values(ascending=False).head(5).index.tolist() for pc in load.index}

    data = np.load(OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{SEED}.npz")
    params = dict(rot=data["rot"], w=data["w"], fm=moco.FEATURE_MAP)
    Z_val = enc.transform(split.X_val)
    s0 = moco.score_exact(params, moco.bloch_exact(Z_val))
    cached = load_scores(OUT / f"moco_{moco.FEATURE_MAP}_{moco.NEG_PER_POS}_seed{SEED}.npz")
    consistency = np.abs(s0 - cached[0.0][0]).max()
    t0 = choose_threshold(split.y_val, s0)
    lam2 = (1 - 4 * P / 3) ** 2
    tp = moco.transfer_threshold(t0, P, 2)
    tp_protocol = choose_threshold(split.y_val, cached[P][0])

    pick = {}
    for label, cls in [("fraud", 1), ("legit", 0)]:
        idx = np.flatnonzero(split.y_val == cls)
        pick[label] = idx[np.argmin(np.abs(s0[idx] - np.median(s0[idx])))]

    th, ph = -params["rot"][:, 1], -params["rot"][:, 0]
    axes_ = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    show_raw = ["Time", "Amount"] + [c for c in load.abs().sum().sort_values(ascending=False).index
                                     if c not in ("Time", "Amount")][:6]

    md = [f"# MOCO end to end: one validation fraud and one validation legit transaction (seed {SEED})", "",
          "Rule for choosing them: the validation transaction of each class whose noiseless MOCO score is the median "
          "of its class. The test set is not used. Generated by `qai/experiments/exp_covariance_e2e.py`.", "",
          f"Encoder fitted on MOCO's 1:10 training set ({len(y_b):,} rows, {int(y_b.sum())} frauds): StandardScaler on "
          "Time and Amount, PCA to 4 components, MinMax to [0, pi], clipped.", "",
          "## PCA inside the encoder", "",
          "| component | explained variance ratio | top 5 loadings (feature: loading) |", "|---|---|---|"]
    for k, pc in enumerate(load.index):
        md.append(f"| {pc} | {fmt(evr[k])} | " + ", ".join(f"{f}: {load.loc[pc, f]:+.3f}" for f in top[pc]) + " |")
    md += ["", f"Total explained variance of the 4 components: {fmt(evr.sum())}.", "",
           "## The two transactions", ""]
    for label in ["fraud", "legit"]:
        i = pick[label]
        row = split.X_val.iloc[[i]]
        comp = enc.pca.transform(enc._scaled(row))[0]
        z = Z_val[i]
        r = moco.bloch_exact(z[None])[0]
        zq = (r * axes_).sum(-1)
        H = zq @ params["w"]
        s = (1 - H) / 2
        sp = 0.5 - lam2 * (0.5 - s)
        md += [f"### Validation {label} (validation row {i}, dataset index {split.X_val.index[i]}, "
               f"true class {int(split.y_val[i])})", "",
               "Raw values (Time, Amount and the 6 features with the largest PCA loadings): "
               + ", ".join(f"{c} = {row[c].iloc[0]:.4f}" for c in show_raw), "",
               "| qubit q | PCA score | angle z_q | <X> = sin z | <Z> = cos z | Rot phi_q | Rot theta_q | weight w_q | <Z_q> after Rot |",
               "|---|---|---|---|---|---|---|---|---|"]
        for q in range(4):
            md.append(f"| {q} | {fmt(comp[q])} | {fmt(z[q])} | {fmt(r[q, 0])} | {fmt(r[q, 2])} | "
                      f"{fmt(params['rot'][q, 0])} | {fmt(params['rot'][q, 1])} | {fmt(params['w'][q])} | {fmt(zq[q])} |")
        md += ["", f"- <H> = sum_q w_q <Z_q> = {fmt(H)}; score p_fraud = (1 - <H>)/2 = **{fmt(s)}**",
               f"- p = 0: threshold t0 = {fmt(t0)} (max-F1 on the noiseless validation scores) -> "
               f"verdict **{'FRAUD' if s >= t0 else 'legit'}**",
               f"- p = 0.05: lambda^2 = {fmt(lam2)}, score = 1/2 - lambda^2 (1/2 - s) = **{fmt(sp)}**, transferred "
               f"threshold t_p = {fmt(tp)} -> verdict **{'FRAUD' if sp >= tp else 'legit'}** (same as at p = 0)", ""]
    md += ["## Checks", "",
           f"- Scores recomputed here vs the cached protocol scores (kit noise model, p = 0), all validation rows: max "
           f"difference {consistency:.1e}.",
           f"- Transferred threshold at p = 0.05: {fmt(tp)}; threshold the protocol chose on the noisy validation "
           f"scores at p = 0.05: {fmt(tp_protocol)} (difference {abs(tp - tp_protocol):.1e}).", ""]
    (OUT / "e2e_example.md").write_text("\n".join(md))

    print(pd.DataFrame([s_full, s_moco]).to_string(index=False))
    print("PCA evr:", np.round(evr, 4), "total", round(evr.sum(), 4))
    for pc in load.index:
        print(pc, [(f, round(load.loc[pc, f], 3)) for f in top[pc]])
    print(f"consistency {consistency:.1e}; t0 {t0:.4f}; t_p {tp:.4f} vs protocol {tp_protocol:.4f}")
    print("picked validation rows:", pick)


if __name__ == "__main__":
    main()
