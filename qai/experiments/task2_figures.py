"""Slide figures for Task 2, in one style, from CSVs the experiments already wrote in results/task2/ (no simulation).
The only non-CSV input is the gate count of figure f, read from the circuit definitions with qml.specs (no data,
no simulation; the same numbers are asserted in tests/test_protocol.py).

Fixed colours per model: baseline orange, R2 aqua, MOCO blue, logistic regression grey (dataviz reference palette).
Every figure is written as PNG (200 dpi) and SVG to results/task2/figures/. Figures whose CSV does not exist yet
(all-frauds CV, adaptive shots) are reported as missing, so the script can simply be rerun later.

    python3 -m qai.experiments.task2_figures
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/task2")
FIG = OUT / "figures"
COLOR = {"baseline": "#eb6834", "R2": "#1baf7a", "MOCO": "#2a78d6", "LogReg": "#52514e"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
PROTOCOL_MODELS = {   # label -> (csv, model name in csv, colour key)
    "baseline VQC": ("baseline_per_seed.csv", "baseline VQC", "baseline"),
    "R2 warm start": ("entangled_per_seed.csv", "R2 entangled (moco)", "R2"),
    "MOCO": ("moco_per_seed_single_10.csv", "MOCO single 1:10", "MOCO"),
    "LogReg 1:10": ("moco_per_seed_single_10.csv", "logistic regression 1:10", "LogReg"),
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "svg.fonttype": "none"})
    return plt


def _clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 200}), ("svg", {})):
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    _plt().close(fig)
    return name


def _direct_labels(ax, ends, x, min_gap):
    placed = {}                                    # top label stays, the next ones are pushed down
    for label, (y, color) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
        placed[label] = (min(y, min((v[0] for v in placed.values()), default=1e9) - min_gap), color)
    for label, (y, _) in placed.items():
        ax.annotate(label, (x, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=14, color=INK)


def protocol_curves(metric, name, title, ylabel, skip_logreg=False):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    levels = [0.0, 0.001, 0.01, 0.05]
    x = np.arange(len(levels))
    ends = {}
    for label, (csv, model, key) in PROTOCOL_MODELS.items():
        if skip_logreg and key == "LogReg":
            continue
        d = pd.read_csv(OUT / csv)
        g = d[d.model == model].groupby("p")[metric]
        m, s = g.mean().reindex(levels).to_numpy(), g.std().reindex(levels).to_numpy()
        style = dict(linestyle=":", linewidth=2) if key == "LogReg" else dict(linewidth=2.5, marker="o", markersize=7)
        ax.fill_between(x, m - s, m + s, color=COLOR[key], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=COLOR[key], **style)
        ends[label] = (m[-1], COLOR[key])
    _direct_labels(ax, ends, x[-1], 0.035 if metric == "AUPRC" else 0.03)
    ax.set_xticks(x, [f"{p:g}" for p in levels])
    ax.set_xlim(-0.2, len(levels) - 0.2 + 1.1)
    ax.set_xlabel("depolarizing p per gate (official protocol, exact expectation values)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    _clean(ax)
    return _save(fig, name)


def fig_a():
    return protocol_curves("AUPRC", "a_auprc_protocol",
                           "MOCO keeps 0.716 at every noise level; the baseline falls to 0.405 at p = 0.05",
                           "test AUPRC (mean ± std over 5 seeds)")


def fig_b():
    return protocol_curves("Spearman rho", "b_spearman_protocol",
                           "MOCO's ranking is untouched by noise (ρ = 1); the baseline's decays to 0.69",
                           "Spearman ρ, noisy vs noiseless test scores", skip_logreg=True)


def fig_c():
    plt = _plt()
    led = pd.read_csv(OUT / "ledger_final.csv")
    led = led[~led.method.str.startswith("footnote")]
    short = {"baseline VQC (kit formula, 65 per example)": ("baseline (kit formula)", "baseline"),
             "baseline VQC (qml.Tracker, 41 per example)": ("baseline (Tracker)", "baseline"),
             "MOCO single 1:10": ("MOCO", "MOCO"), "R2 entangled (warm start)": ("R2 warm start", "R2"),
             "R2 entangled (random init)": ("R2 random init", "R2")}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(led))[::-1]
    for yi, (_, r) in zip(y, led.iterrows()):
        label, key = short[r.method]
        ax.barh(yi, r["total executions"], color=COLOR[key], height=0.6)
        red = r["reduction vs baseline (formula)"]
        text = f"{r['total executions']:,.0f}" + ("" if red <= 0 else f"   (−{100 * red:.1f}%)")
        ax.annotate(text, (r["total executions"], yi), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=14, color=INK)
    ax.set_yticks(y, [short[m][0] for m in led.method])
    ax.set_xscale("log")
    ax.set_xlim(1e4, 4e7)
    ax.set_xlabel("circuit executions: training + model selection + threshold calibration (log scale)")
    ax.set_title("MOCO needs 46x fewer circuit executions than the baseline (−97.8%)", loc="left")
    _clean(ax, "x")
    return _save(fig, "c_ledger")


def fig_d():
    plt = _plt()
    d = pd.read_csv(OUT / "noise_shots.csv")
    levels = sorted(d.p.unique())
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ends = {}
    for label, model, key in [("baseline VQC", "baseline VQC", "baseline"), ("R2 warm start", "R2 (moco)", "R2"),
                              ("MOCO", "MOCO", "MOCO")]:
        g = d[(d.model == model) & (d.shots == 1024)].groupby("p").AUPRC
        m, s = g.mean().reindex(levels).to_numpy(), g.std().reindex(levels).to_numpy()
        ax.fill_between(x, m - s, m + s, color=COLOR[key], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=COLOR[key], linewidth=2.5, marker="o", markersize=7)
        ends[label] = (m[-1], COLOR[key])
    lr = pd.read_csv(OUT / "moco_per_seed_single_10.csv")
    lr = lr[lr.model == "logistic regression 1:10"].AUPRC.mean()
    ax.axhline(lr, color=COLOR["LogReg"], linestyle=":", linewidth=2)
    ends[f"LogReg 1:10 ({lr:.3f})"] = (lr, COLOR["LogReg"])
    _direct_labels(ax, ends, x[-1], 0.05)
    ax.set_xticks(x, [f"{p:g}" for p in levels])
    ax.set_xlim(-0.3, len(levels) - 0.3 + 1.8)
    ax.set_xlabel("depolarizing p per gate (ordinal axis, not to scale)")
    ax.set_ylabel("test AUPRC with 1024 shots per transaction")
    ax.set_title("With 1024 real shots at p = 0.05: MOCO 0.705, R2 0.643, baseline 0.007", loc="left")
    _clean(ax)
    return _save(fig, "d_auprc_1024_shots")


def fig_e():
    plt = _plt()
    t = pd.read_csv(OUT / "shot_tax.csv")
    levels = sorted(t.p.unique())
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ends = {}
    for label, model, key in [("baseline VQC (lower bound)", "baseline VQC", "baseline"),
                              ("R2 warm start", "R2 (moco)", "R2"), ("MOCO = λ⁻⁴", "MOCO", "MOCO")]:
        y = t[t.model == model].set_index("p").shot_tax.reindex(levels).to_numpy()
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=COLOR[key], linewidth=2.5, marker="o", markersize=7)
        ends[label] = (np.log10(y[ok][-1]), x[ok][-1])
        if not ok.all():
            ax.annotate(f"infinite at p = {levels[np.flatnonzero(~ok)[0]]:g}", (x[ok][-1], y[ok][-1]),
                        xytext=(10, -22), textcoords="offset points", fontsize=14, color=MUTED)
    for label, (ly, xe) in ends.items():
        ax.annotate(label, (xe, 10 ** ly), xytext=(10, 0), textcoords="offset points", va="center", fontsize=14,
                    color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{p:g}" for p in levels])
    ax.set_xlim(-0.3, len(levels) - 0.3 + 2.2)
    ax.set_xlabel("depolarizing p per gate (ordinal axis, not to scale)")
    ax.set_ylabel("shot tax 1/c² (x shots for the same S/N)")
    ax.set_title("At p = 0.05 MOCO pays 1.32x more shots; the baseline at least 504x", loc="left")
    _clean(ax)
    return _save(fig, "e_shot_tax")


def fig_f():
    from qai import kit_vqc as kit
    from qai import moco
    from qai.protocol import gate_stats
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    params = dict(rot=np.zeros((4, 3)), w=np.ones(4) / 4, fm="single")
    sm = gate_stats(moco.make_circuit(params), np.zeros(4), None)
    sb = gate_stats(kit.vqc, np.zeros(4), kit.init_weights(0).detach())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})

    def box(ax, x, y, w, h, text, color, fs=14):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor="none"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#ffffff")

    for ax in (a1, a2):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.8, 4.2)
        ax.axis("off")
        for q in range(4):
            ax.plot([0.9, 8.9], [3 - q, 3 - q], color=MUTED, linewidth=1.5, zorder=0)
            ax.text(0.45, 3 - q, f"q{q}", ha="center", va="center", fontsize=14, color=INK)
    for q in range(4):
        box(a1, 2.6, 3 - q, 2.0, 0.6, f"RY(x{q})", COLOR["MOCO"])
        box(a1, 6.0, 3 - q, 3.6, 0.6, f"Rot(φ{q}, θ{q}, 0)", COLOR["MOCO"])
        a1.text(9.3, 3 - q, "Z", ha="center", va="center", fontsize=14, color=INK)
    a1.set_title(f"MOCO: {sm['gates']} gates, {sm['two_qubit']} CNOT, {sm['noise_locations']} noise locations, "
                 f"depth {sm['depth']}", loc="left", fontsize=15)
    a1.text(0.4, -0.55, "readout H = Σ w_q Z_q, p_fraud = (1 − ⟨H⟩) / 2", fontsize=14, color=INK)

    box(a2, 2.9, 1.5, 3.6, 3.7, "ZZ feature map\nx 2 reps\nH, RZ on each qubit\n+ 6 pairs CNOT-RZ-CNOT", COLOR["baseline"])
    box(a2, 6.9, 1.5, 3.6, 3.7, "EfficientSU2 ansatz\n4 rotation layers\nRY, RZ on each qubit\n+ 3 CNOT chains",
        COLOR["baseline"])
    a2.text(9.3, 3, "Z", ha="center", va="center", fontsize=14, color=INK)
    for q in range(1, 4):
        a2.text(9.3, 3 - q, "-", ha="center", va="center", fontsize=14, color=MUTED)
    a2.set_title(f"Baseline: {sb['gates']} gates, {sb['two_qubit']} CNOT, {sb['noise_locations']} noise locations, "
                 f"depth {sb['depth']}", loc="left", fontsize=15)
    a2.text(0.4, -0.55, "readout ⟨Z0⟩ only; 12 of 32 parameters never reach q0", fontsize=14, color=INK)
    fig.suptitle("Every gate is followed by a depolarizing channel on each qubit it touches: 8 vs 126 noise locations",
                 x=0.01, ha="left", fontsize=16)
    return _save(fig, "f_circuits")


def fig_g():
    plt = _plt()
    rows = []
    for fm in ["single", "pair", "linear_cnot", "kit_zz"]:
        f = OUT / f"moco_per_seed_{fm}_10.csv"
        if not f.exists():
            rows.append((fm, np.nan, np.nan, np.nan, np.nan))
            continue
        d = pd.read_csv(f)
        d = d[(d.model == f"MOCO {fm} 1:10") & (d.p == 0.05)]
        rows.append((fm, d.AUPRC.mean(), d.AUPRC.std(), d["Spearman rho"].mean(), d["Spearman rho"].std()))
    t = pd.DataFrame(rows, columns=["fm", "auprc", "auprc_sd", "rho", "rho_sd"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    colors = [COLOR["MOCO"] if fm in ("single", "pair") else COLOR["baseline"] for fm in t.fm]
    for ax, col, sd, label in [(a1, "auprc", "auprc_sd", "test AUPRC at p = 0.05"),
                               (a2, "rho", "rho_sd", "Spearman ρ at p = 0.05")]:
        x = np.arange(len(t))
        ax.bar(x, t[col].fillna(0), yerr=t[sd].fillna(0), color=colors, width=0.6, capsize=5)
        for xi, v in zip(x, t[col]):
            ax.annotate("not run\nlocally" if np.isnan(v) else f"{v:.3f}", (xi, 0 if np.isnan(v) else v),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=14,
                        color=MUTED if np.isnan(v) else INK)
        ax.set_xticks(x, ["single\n(8 gates)", "pair\n(12, 0 CNOT)", "linear_cnot\n(3 CNOT)", "kit_zz\n(24 CNOT)"])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.1)
        _clean(ax)
    fig.suptitle("Same MOCO training, only the feature map changes: every CNOT costs AUPRC and ranking",
                 x=0.01, ha="left", fontsize=16)
    return _save(fig, "g_ablations")


def fig_h():
    plt = _plt()
    sel = pd.read_csv(Path("results/selection_on_validation.csv"))
    g = sel.groupby(["fm", "ratio"]).agg(m=("val_auprc", "mean"), s=("val_auprc", "std"),
                                         ex=("train_exec", "first")).reset_index()
    g = g.sort_values(["fm", "ratio"], ascending=[False, True])
    cells = [[fm, f"1:{r}", f"{m:.3f} ± {s:.3f}", f"{int(ex):,}"] for fm, r, m, s, ex in g.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["feature map", "legit : fraud", "validation AUPRC (5 seeds)",
                                                "training executions"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.6)
    best = g.m.max()
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=INK, weight="bold")
        elif cells[r - 1][0] == "single" and cells[r - 1][1] == "1:10":
            cell.set_facecolor("#cde2fb")
    ax.set_title("Configuration chosen on validation only: single 1:10 (best mean, cheapest among ties)",
                 loc="left", fontsize=16)
    return _save(fig, "h_selection")


MISSING = {"i_cv_all_frauds": OUT / "cv" / "cv_per_fold.csv", "j_adaptive_shots": OUT / "adaptive_shots.csv"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    made = [f() for f in (fig_a, fig_b, fig_c, fig_d, fig_e, fig_f, fig_g, fig_h)]
    print("written:", ", ".join(made))
    for name, src in MISSING.items():
        print(f"missing: {name} (needs {src})" if not src.exists() else f"available, not drawn yet: {name} ({src})")


if __name__ == "__main__":
    main()
