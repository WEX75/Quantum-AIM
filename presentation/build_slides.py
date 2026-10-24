"""Build the Task 2 presentation as a vector PDF (presentation/moco_slides.pdf), straight from the CSVs in results/.

No LaTeX or LibreOffice is installed, so the deck is drawn with matplotlib. Every number on a slide is read from a CSV
when the deck is built (see NUM below; the source of each number is in presentation/SPEAKER_NOTES.md). Numbers that
the adaptive-shots / robust-encoding session has not produced yet live in ONE place, PLACEHOLDERS: fill a value
there (or point it to its CSV) and rebuild. Unfilled ones are drawn as a red [TBD: name].

    python3 presentation/build_slides.py                 # -> presentation/moco_slides.pdf
    pdftoppm -png -r 80 presentation/moco_slides.pdf presentation/pages/slide
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "task2"
FIG, CFIG = R / "figures", R / "figures" / "compact"
OUT = ROOT / "presentation" / "moco_slides.pdf"

# ---------------------------------------------------------------- the ONE place for numbers that do not exist yet
PLACEHOLDERS = {
    "team_member_2": None,            # teammate's name (slides 1, 2-4 speaker)
    "N_cal": None,                    # noise-meter calibration executions      <- results/task2/noise_meter.csv
    "noise_meter_p_error": None,      # error of the estimated p                 <- results/task2/noise_meter.csv
    "kit_margin": None,               # score margin fraud vs legit, kit enc.    <- other session's CSV
    "per_shot_std": None,             # per-shot std of the score, kit encoding  <- other session's CSV
    "robust_auprc_p005": None,        # MOCO-robust test AUPRC at p = 0.05       <- other session's CSV
    "triage_shots_per_tx": None,      # mean shots per transaction with triage   <- results/task2/adaptive_shots.csv
    "triage_f1": None,                # F1 with triage vs fixed S = 1024         <- results/task2/adaptive_shots.csv
}


def ph(key):
    v = PLACEHOLDERS[key]
    return f"[TBD: {key}]" if v is None else str(v)


# ---------------------------------------------------------------- numbers, all read from CSVs
def _ms(d, col):
    return d[col].mean(), d[col].std()


def load_numbers():
    n = {}
    split = pd.read_csv(R / "dataset_split.csv").set_index("part")
    n["rows"], n["frauds"] = int(split.loc["full dataset", "rows"]), int(split.loc["full dataset", "frauds"])
    n["rate"] = 100 * split.loc["full dataset", "fraud_rate"]
    n["split"] = {k: (int(split.loc[k, "rows"]), int(split.loc[k, "frauds"])) for k in ["train", "validation", "test"]}
    circ = pd.read_csv(R / "circuits.csv").set_index("circuit")
    n["circ"] = circ
    n["base_exec"] = int(pd.read_csv(R / "ledger_baseline.csv").iloc[0]["total executions"])
    lm = pd.read_csv(R / "ledger_moco_single_10.csv").iloc[0]
    n["moco_exec"], n["moco_per_row"] = int(lm["train executions"]), int(lm["executions_per_example"])

    tables = {"baseline VQC": ("baseline_per_seed.csv", "baseline VQC"),
              "R2 random init": ("entangled_per_seed.csv", "R2 entangled (random)"),
              "R2 warm start": ("entangled_per_seed.csv", "R2 entangled (moco)"),
              "MOCO": ("moco_per_seed_single_10.csv", "MOCO single 1:10"),
              "LogReg 1:10": ("moco_per_seed_single_10.csv", "logistic regression 1:10"),
              "LogReg 1:1": ("baseline_per_seed.csv", "logistic regression")}
    proto = {}
    for label, (f, m) in tables.items():
        d = pd.read_csv(R / f)
        d = d[d.model == m]
        proto[label] = {p: {c: _ms(g, c) for c in ["AUPRC", "Spearman rho", "F1", "MCC"]} for p, g in d.groupby("p")}
    n["proto"] = proto

    ns = pd.read_csv(R / "noise_shots.csv")
    s1024 = ns[(ns.shots == 1024) & (ns.p == 0.05)].groupby("model").AUPRC.mean()
    n["s1024"] = {k: s1024[k] for k in ["MOCO", "R2 (moco)", "baseline VQC"]}
    tax = pd.read_csv(R / "shot_tax.csv").query("p == 0.05").set_index("model").shot_tax
    n["tax"] = {k: tax[k] for k in ["MOCO", "R2 (moco)", "baseline VQC"]}
    chk = pd.read_csv(R / "noise_shots_checks.csv")
    moco_chk = chk[chk.model == "MOCO"]
    n["chk_kit"] = moco_chk[moco_chk.check.str.contains("vs kit noisy_predict")].max_error.max()
    n["chk_comp"] = moco_chk[moco_chk.check.str.contains("compression")].max_error.max()
    n["chk_pass"], n["chk_total"] = int((chk.status == "PASS").sum()), len(chk)

    cv = pd.read_csv(R / "cv" / "cv_per_fold.csv")
    cv0 = cv[cv.p == 0].set_index(["model", "fold"]).AUPRC
    n["cv"] = {m: _ms(cv[(cv.model == m) & (cv.p == 0)], "AUPRC") for m in ["MOCO", "logistic regression 1:10", "baseline VQC"]}
    diff = (cv0["MOCO"] - cv0["logistic regression 1:10"])
    n["cv_diff"] = (diff.mean(), diff.std())
    b = cv[cv.model == "baseline VQC"].pivot(index="fold", columns="p", values="AUPRC")
    n["cv_rise"] = int((b[0.01] > b[0.0]).sum())
    n["cv_rho05"] = cv[(cv.model == "baseline VQC") & (cv.p == 0.05)]["Spearman rho"].mean()

    led = pd.read_csv(R / "ledger_final.csv").set_index("method")
    n["led"] = led
    sel = pd.read_csv(ROOT / "results" / "selection_on_validation.csv")
    s = sel[(sel.fm == "single") & (sel.ratio == 10)].val_auprc
    n["sel"] = (s.mean(), s.std())
    zz = pd.read_csv(R / "moco_per_seed_kit_zz_10.csv")
    n["zz"] = zz[(zz.model == "MOCO kit_zz 1:10") & (zz.p == 0)].AUPRC.mean()
    need = pd.read_csv(R / "shots_needed.csv")
    n["agree"] = (int(need.agree.sum()), len(need))
    n["cv_drop05"] = int((b[0.05] < b[0.0]).sum())
    n["kit_red"] = 1 - n["moco_exec"] / n["base_exec"]
    return n


# ---------------------------------------------------------------- drawing helpers (inches, origin bottom-left)
W, H = 13.333, 7.5
INK, MUTED, BG, DARK, ACCENT, TBD = "#0b0b0b", "#52514e", "#ffffff", "#0d366b", "#2a78d6", "#c62828"
COL = {"MOCO": "#2a78d6", "baseline": "#eb6834", "R2": "#1baf7a", "LogReg": "#52514e"}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def slide(title, dark=False, number=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(DARK if dark else BG)
    if title:
        txt(fig, 0.55, 6.95, title, 24, bold=True, color="#ffffff" if dark else INK)
    if number is not None:
        fig.text(12.85 / W, 0.25 / H, str(number), fontsize=11, color=MUTED, ha="right")
    return fig


def wrap(s, width_in, size, bold=False):
    chars = max(10, int(width_in / (size / 72 * (0.62 if bold else 0.55))))
    return "\n".join(textwrap.fill(par, chars) for par in s.split("\n"))


def txt(fig, x, y, s, size=16, bold=False, color=INK, width=None, va="top", ha="left", style="normal"):
    s = wrap(s, width, size, bold) if width else s
    return fig.text(x / W, y / H, s, fontsize=size, fontweight="bold" if bold else "normal", color=color,
                    va=va, ha=ha, linespacing=1.35, style=style)


def bullets(fig, x, y, items, size=16, width=6.0, gap=0.16, color=INK):
    """Draw bullet paragraphs, return the y below the last one."""
    for it in items:
        lines = wrap(it, width - 0.3, size).split("\n")
        fig.text(x / W, y / H, "–", fontsize=size, color=ACCENT, va="top")
        fig.text((x + 0.3) / W, y / H, "\n".join(lines), fontsize=size, color=TBD if "[TBD" in it else color,
                 va="top", linespacing=1.35)
        y -= len(lines) * size / 72 * 1.35 + gap
    return y


def image(fig, path, x, y_top, w=None, h=None):
    img = mpimg.imread(path)
    ar = img.shape[0] / img.shape[1]
    if w is None:
        w = h / ar
    h = w * ar
    ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
    ax.imshow(img)
    ax.axis("off")
    return y_top - h


def card(fig, x, y_top, w, h, color, alpha=0.08):
    ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=color, alpha=alpha, edgecolor="none"))


def stat(fig, x, y, value, label, color, size=40, width=3.6):
    txt(fig, x, y, value, size, bold=True, color=color)
    txt(fig, x, y - size / 72 * 1.25, label, 14, color=MUTED, width=width)


def table(fig, x, y_top, header, rows, widths, size=13, row_h=0.36, colors=None):
    ax = fig.add_axes([0, 0, 1, 1], zorder=1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    xs = np.cumsum([x] + widths)
    y = y_top
    for j, h in enumerate(header):
        ax.text(xs[j] + 0.08, y - row_h / 2, h, fontsize=size, fontweight="bold", va="center", color=INK)
    ax.plot([x, xs[-1]], [y - row_h, y - row_h], color="#c3c2b7", linewidth=1)
    for i, r in enumerate(rows):
        y -= row_h
        for j, v in enumerate(r):
            c = (colors or {}).get(r[0], INK) if j == 0 else INK
            ax.text(xs[j] + 0.08, y - row_h / 2, v, fontsize=size, va="center", color=c,
                    fontweight="bold" if r[0] == "MOCO" else "normal")
        ax.plot([x, xs[-1]], [y - row_h, y - row_h], color="#e1e0d9", linewidth=0.8)
    return y - row_h


def f3(v):
    return f"{v:.3f}"


def pm(t):
    return f"{t[0]:.3f} ± {t[1]:.3f}"


# ---------------------------------------------------------------- slides
def build(n):
    pages = []
    P, C = n["proto"], n["circ"]
    bc, mc, rc = C.loc["baseline VQC (kit)"], C.loc["MOCO single"], C.loc["R2 entangled"]

    # 1 title
    f = slide(None, dark=True)
    txt(f, 0.8, 5.9, "MOCO: Measure Once, Compile Once", 40, bold=True, color="#ffffff")
    txt(f, 0.8, 5.05, "Fraud detection on a quantum budget", 26, color="#cde2fb")
    for i, (a, b) in enumerate([
            ("Measure once", f"{n['moco_per_row']} circuit executions per training transaction, in total"),
            ("Train convexly", "global optimum in PyTorch, no circuit inside the training loop"),
            ("Noise cannot change the ranking", "Spearman ρ = 1 at every noise level, by construction")]):
        y = 3.75 - 0.75 * i
        txt(f, 0.8, y, a, 20, bold=True, color="#ffffff")
        txt(f, 6.2, y, b, 18, color="#e1e0d9", width=6.5)
    txt(f, 0.8, 0.95, f"Darius Tatomir  ·  {ph('team_member_2')}   |   BreaQ 2026, Quantum AI track, Task 2", 15,
        color="#ff8a80" if PLACEHOLDERS["team_member_2"] is None else "#cde2fb")
    pages.append(f)

    # 2 problem and baseline audit
    f = slide(f"The baseline breaks under noise: AUPRC {P['baseline VQC'][0.0]['AUPRC'][0]:.3f} to {P["baseline VQC"][0.05]["AUPRC"][0]:.3f}", number=2)
    stat(f, 0.6, 6.0, f"{n['frauds']}", f"frauds in {n['rows']:,} transactions ({n['rate']:.2f}%)", COL["baseline"])
    bullets(f, 0.6, 4.55, [
        f"A model that flags nothing is {100 - n['rate']:.2f}% accurate: accuracy says nothing. We rank by AUPRC "
        "(precision-recall).",
        f"Common split, stratified 60/20/20, seed 42: test {n['split']['test'][0]:,} rows with "
        f"{n['split']['test'][1]} frauds; threshold chosen on validation at the same noise level."], 16, width=5.6)
    card(f, 6.75, 6.15, 6.0, 5.4, COL["baseline"])
    txt(f, 7.05, 5.95, "Baseline audit (kit VQC: ZZ map + EfficientSU2)", 17, bold=True, width=5.6)
    for i, (v, l) in enumerate([(int(bc.gates), "gates"), (int(bc.two_qubit), "CNOT"), (int(bc.depth), "depth"),
                                (int(bc.noise_locations), "noise locations")]):
        stat(f, 7.05 + 2.85 * (i % 2), 5.2 - 1.45 * (i // 2), str(v), l, COL["baseline"], size=34, width=2.6)
    b0, b5 = P["baseline VQC"][0.0]["AUPRC"], P["baseline VQC"][0.05]["AUPRC"]
    txt(f, 7.05, 2.25, f"{n['base_exec']:,} circuit executions to train and validate", 15, width=5.5)
    txt(f, 7.05, 1.6, f"Test AUPRC {pm(b0)} at p = 0  →  {pm(b5)} at p = 0.05 (5 seeds)", 15, bold=True, width=5.6)
    pages.append(f)

    # 3 Task 1 -> idea
    f = slide("Two facts from Task 1 make noise harmless for the ranking", number=3)
    for i, (head, body, formula) in enumerate([
            ("Exercise 3.2: squashing keeps AUPRC",
             "AUPRC depends only on the order of the scores. Compressing them towards 1/2 keeps the order, "
             "so AUPRC and AUROC do not change; only the threshold has to move.",
             "s  →  1/2 + c (s − 1/2),  c > 0"),
            ("Exercise 10.1: depolarizing scales ⟨Z⟩",
             "A depolarizing channel after a one-qubit gate shrinks the Bloch vector by the same factor for "
             "every input.",
             "⟨Z⟩  →  (1 − 4p/3) ⟨Z⟩")]):
        x = 0.6 + 6.25 * i
        card(f, x, 6.1, 5.95, 4.3, ACCENT)
        txt(f, x + 0.3, 5.85, head, 19, bold=True, width=5.4)
        txt(f, x + 0.3, 5.05, body, 16, width=5.3)
        txt(f, x + 0.3, 2.75, formula, 22, bold=True, color=ACCENT)
    txt(f, 0.6, 1.25, "MOCO = a circuit where every qubit sees the same number of one-qubit gates and no CNOT, so the "
        "whole score is squashed by one factor λ² with λ = 1 − 4p/3.", 17, bold=True, width=12.2)
    pages.append(f)

    # 4 pipeline
    f = slide("Every choice is made on train or validation, never on test", number=4)
    bullets(f, 0.6, 6.0, [
        "Encoder fitted on the training subset only: scale Time and Amount, PCA to 4 components, angles in [0, π].",
        "Undersample the training split, keeping all 295 training frauds; ratio and feature map chosen on validation "
        f"(single 1:10: validation AUPRC {pm(n['sel'])}, 5 seeds).",
        "C of the convex fit chosen by 5-fold CV inside the training subset: no validation executions.",
        "Test set touched once, at the end; threshold maximises F1 on validation at the same noise level."],
        16, width=5.3)
    sel = pd.read_csv(ROOT / "results" / "selection_on_validation.csv")
    g = sel.groupby(["fm", "ratio"]).agg(m=("val_auprc", "mean"), s=("val_auprc", "std"), ex=("train_exec", "first"))
    g = g.reset_index().sort_values(["fm", "ratio"], ascending=[False, True])
    rows = [[fm, f"1:{r}", f"{m:.3f} ± {sd:.3f}", f"{int(ex):,}"] for fm, r, m, sd, ex in g.itertuples(index=False)]
    end = table(f, 6.3, 6.2, ["feature map", "ratio", "val AUPRC (5 seeds)", "train exec."], rows,
                [1.55, 0.95, 2.5, 1.45], size=14, row_h=0.4, colors={"single": INK})
    txt(f, 6.38, end - 0.15, "Chosen: single 1:10, best mean; pair 1:30 ties at 4x the executions.", 14,
        color=ACCENT, bold=True, width=6.3)
    pages.append(f)

    # 5 circuit
    f = slide(f"MOCO has {int(mc.noise_locations)} noise locations, the baseline {int(bc.noise_locations)}", number=5)
    image(f, FIG / "f_circuits.png", 0.45, 6.25, w=12.4)
    pages.append(f)

    # 6 method
    f = slide("Training the circuit is a convex fit on data measured once", number=6)
    steps = [("1  Measure once", f"RY(x)|0⟩ has Bloch vector (sin x, 0, cos x). Read every training transaction in the "
                                 f"X and Z bases: {n['moco_per_row']} executions per row, {n['moco_exec']:,} in total."),
             ("2  Heisenberg picture", "After a local Rot, ⟨Z_q⟩ = m_q · r_q(x) exactly. The readout H = Σ w_q Z_q is "
                                       "linear in the measured Bloch vectors."),
             ("3  Convex fit", "Logistic regression on the measured vectors in PyTorch (LBFGS), C by CV on train: "
                               "global optimum, no barren plateau, no circuit in the loop."),
             ("4  Compile back", "Coefficients become Rot(−φ, −θ, 0) angles and observable weights; p_fraud = "
                                 "(1 − ⟨H⟩)/2. The circuit does the classification.")]
    for i, (h, b) in enumerate(steps):
        x, y = 0.6 + 6.25 * (i % 2), 6.1 - 2.35 * (i // 2)
        card(f, x, y, 5.95, 2.1, ACCENT)
        txt(f, x + 0.3, y - 0.25, h, 19, bold=True, color=ACCENT)
        txt(f, x + 0.3, y - 0.8, b, 15, width=5.4)
    txt(f, 0.6, 1.15, "Same idea as classical surrogates of quantum models (Schreiber, Eisert, Meyer, PRL 2023) and "
        "data acquisition followed by classical processing (Cerezo et al., 2025): here the result is compiled back "
        "into the circuit.", 13, color=MUTED, width=12.2, style="italic")
    pages.append(f)

    # 7 theorem
    f = slide("Noise only squashes MOCO's scores: the ranking cannot change", number=7)
    for i, s in enumerate([
            "1  Each depolarizing channel on a qubit shrinks its Bloch vector by λ = 1 − 4p/3 (exercise 10.1).",
            "2  No CNOT and k = 2 gates on every qubit: the state stays a product, every ⟨Z_q⟩ is multiplied by λ².",
            "3  The readout is linear, so the whole score is squashed by λ² around 1/2 (exercise 3.2)."]):
        txt(f, 0.6, 6.0 - 0.75 * i, s, 18, width=12.2)
    card(f, 0.6, 3.55, 12.15, 1.25, ACCENT, alpha=0.12)
    txt(f, 6.67, 3.18, "s_noisy(x)  =  1/2  −  λ² (1/2 − s_ideal(x))", 28, bold=True, color=ACCENT, ha="center")
    bullets(f, 0.6, 1.95, [
        "AUPRC, AUROC and ρ = 1 identical at every p < 3/4; the max-F1 threshold moves analytically: "
        "t_p = 1/2 − λ² (1/2 − t_0).",
        f"Checked against the kit's noise model (noisy_predict): max error {n['chk_kit']:.1e} on 200 rows at p = 0.05 "
        f"and 0.5; effective compression equals λ² to {n['chk_comp']:.0e}."], 16, width=12.2)
    pages.append(f)

    # 8 official results
    f = slide("MOCO matches logistic regression and stays flat under noise", number=8)
    order = ["baseline VQC", "R2 random init", "R2 warm start", "MOCO", "LogReg 1:10", "LogReg 1:1"]
    rows = []
    for m in order:
        r = P[m]
        rows.append([m] + [pm(r[p]["AUPRC"]) for p in [0.0, 0.001, 0.01, 0.05]]
                    + [f3(r[0.05]["Spearman rho"][0]) if not m.startswith("LogReg") else "-",
                       f3(r[0.05]["F1"][0]), f3(r[0.05]["MCC"][0])])
    colors = {"baseline VQC": COL["baseline"], "R2 random init": COL["R2"], "R2 warm start": COL["R2"],
              "MOCO": COL["MOCO"], "LogReg 1:10": COL["LogReg"], "LogReg 1:1": COL["LogReg"]}
    table(f, 0.5, 6.3, ["model (5 seeds)", "AUPRC p=0", "p=0.001", "p=0.01", "p=0.05", "ρ .05", "F1 .05", "MCC .05"],
          rows, [2.15, 1.63, 1.63, 1.63, 1.63, 1.0, 1.0, 1.0], size=13, row_h=0.36, colors=colors)
    image(f, CFIG / "a_auprc_protocol.png", 0.5, 3.6, h=3.25)
    image(f, CFIG / "b_spearman_protocol.png", 6.75, 3.6, h=3.25)
    pages.append(f)

    # 9 answer to the hint
    f = slide(f"With real shots noise is a tax: MOCO {n['tax']['MOCO']:.2f}x, baseline at least {n['tax']['baseline VQC']:.0f}x", number=9)
    s = n["s1024"]
    stat(f, 0.6, 6.15, f3(s["MOCO"]), "MOCO, test AUPRC with 1024 shots at p = 0.05", COL["MOCO"], size=34, width=3.6)
    stat(f, 4.75, 6.15, f3(s["R2 (moco)"]), "R2 warm start, same setting", COL["R2"], size=34, width=3.6)
    stat(f, 8.9, 6.15, f3(s["baseline VQC"]), "baseline: chance level", COL["baseline"], size=34, width=3.6)
    image(f, CFIG / "d_auprc_1024_shots.png", 0.45, 4.55, h=3.55)
    image(f, CFIG / "e_shot_tax.png", 6.75, 4.55, h=3.55)
    t = n["tax"]
    txt(f, 0.6, 0.62, f"Shot tax 1/c² at p = 0.05 (c = measured score compression): MOCO {t['MOCO']:.2f}x (= λ⁻⁴), "
        f"R2 {t['R2 (moco)']:.2f}x, baseline ≥ {t['baseline VQC']:.0f}x (lower bound: its ranking is also lost).",
        13, color=MUTED, width=12.0)
    pages.append(f)

    # 10 CV
    cvm, cvl, cvb = n["cv"]["MOCO"], n["cv"]["logistic regression 1:10"], n["cv"]["baseline VQC"]
    f = slide(f"All {n['frauds']} frauds: MOCO matches LogReg and is {cvb[1] / cvm[1]:.0f}x more stable", number=10)
    for i, (v, l, c) in enumerate([(pm(cvm), "MOCO, 5-fold CV, every p", COL["MOCO"]),
                                   (pm(cvl), "LogReg 1:10, same inputs", COL["LogReg"]),
                                   (pm(cvb), "baseline VQC, p = 0", COL["baseline"])]):
        stat(f, 0.6 + 4.15 * i, 6.15, v, l, c, size=28, width=3.8)
    image(f, FIG / "i_cv_all_frauds.png", 0.5, 4.75, w=12.3)
    pages.append(f)

    # 11 paradox
    f = slide("Paradox: the baseline's AUPRC rises while its ranking decays", number=11)
    image(f, FIG / "k_paradox.png", 0.5, 6.3, w=12.3)
    txt(f, 0.6, 1.15, f"AUPRC rises at p = 0.01 in {n['cv_rise']} of 5 folds while ρ falls (ρ = "
        f"{n['cv_rho05']:.2f} at p = 0.05): one metric can report an improvement where the model gets worse, as in the "
        "starting article. MOCO cannot show this: its ranking is invariant.", 15, width=12.2)
    pages.append(f)

    # 12 ledger
    led = n["led"]
    bf = led.loc["baseline VQC (kit formula, 65 per example)"]
    mo = led.loc["MOCO single 1:10"]
    r2w = led.loc["R2 entangled (warm start)"]
    f = slide(f"MOCO needs {round(bf['total executions'] / mo['total executions'])}x fewer circuit executions", number=12)
    base_kit = int(bf["training executions"] + bf["model-selection executions"])
    bullets(f, 0.6, 6.1, [
        f"Kit definition (training + model selection): baseline {base_kit:,} vs MOCO {int(mo['training executions']):,} "
        f"({100 * (int(mo['training executions']) / base_kit - 1):.2f}%).",
        f"With threshold calibration by the noise meter: baseline {int(bf['total executions']):,} vs MOCO "
        f"{int(mo['training executions']):,} + {ph('N_cal')}.",
        f"Most conservative, threshold calibrated on the full noisy validation set: MOCO "
        f"{int(mo['total executions']):,} ({-100 * mo['reduction vs baseline (formula)']:.1f}%).",
        f"R2 warm start: {int(r2w['total executions']):,} ({-100 * r2w['reduction vs baseline (formula)']:.1f}%)."],
        15, width=5.4)
    image(f, FIG / "c_ledger.png", 6.1, 6.2, w=6.8)
    pages.append(f)

    # 13 shot-aware encoding and adaptive triage (placeholders until the other session's CSVs exist)
    f = slide("Adaptive triage needs a shot-aware encoding", number=13)
    for i, (h, b) in enumerate([
            ("Why triage fails on the kit encoding",
             f"Score margin between frauds and legit transactions {ph('kit_margin')} vs per-shot standard deviation "
             f"{ph('per_shot_std')}: a few shots cannot separate them."),
            ("Robust percentile encoding",
             f"Angles from training-set percentiles keep the classes apart on the Bloch sphere. MOCO-robust test "
             f"AUPRC at p = 0.05: {ph('robust_auprc_p005')}."),
            ("Noise meter and triage",
             f"Estimate p from {ph('N_cal')} calibration executions (error {ph('noise_meter_p_error')}), move the "
             f"threshold analytically, stop sampling early: {ph('triage_shots_per_tx')} shots per transaction, "
             f"F1 {ph('triage_f1')}.")]):
        x = 0.6 + 4.15 * i
        card(f, x, 6.1, 3.9, 4.6, ACCENT)
        txt(f, x + 0.25, 5.85, h, 16, bold=True, width=3.4)
        txt(f, x + 0.25, 4.85, b, 15, width=3.45, color=TBD if "[TBD" in b else INK)
    txt(f, 0.6, 1.1, "All numbers on this slide come from the adaptive-shots session (results/task2/adaptive_shots.csv, "
        "noise_meter.csv); red values are still being computed.", 13, color=MUTED, width=12.2, style="italic")
    pages.append(f)

    # 14 what did not work
    f = slide("What did not work, measured honestly", number=14)
    r1 = C.loc["R1 light-cone baseline"]
    bullets(f, 0.6, 6.1, [
        f"Light-cone pruning (R1): {int(bc.gates - r1.gates)} gates removed, scores identical to 1e-10: zero gain.",
        "Entanglement: 3 CNOTs already lower AUPRC and break the invariance (figure).",
        f"The kit's ZZ encoding caps at {n['zz']:.2f} even with the globally optimal readout.",
        "Noise-aware fine-tuning in R2: no gain over noiseless training.",
        f"The shots-needed table was too noisy to use (validation and test agree in {n['agree'][0]} of "
        f"{n['agree'][1]} cells).",
        f"The baseline's drop under exact noise is fold-dependent ({n['cv_drop05']} of 5 folds)."], 15, width=5.6,
        gap=0.12)
    image(f, CFIG / "g_ablations.png", 6.35, 6.2, w=6.45)
    pages.append(f)

    # 15 claims, next steps, references
    f = slide("What we claim, what we do not, what comes next", number=15)
    card(f, 0.6, 6.1, 5.95, 2.6, COL["MOCO"])
    txt(f, 0.85, 5.85, "We claim", 19, bold=True, color=COL["MOCO"])
    bullets(f, 0.85, 5.3, ["Robustness: ρ = 1 at every p, shot tax λ⁻⁴.",
                           f"Stability: ±{cvm[1]:.3f} across folds (baseline ±{cvb[1]:.3f}).",
                           f"Cost: {n['moco_exec']:,} training executions ({-100 * n['kit_red']:.2f}%)."],
            15, width=5.5, gap=0.08)
    card(f, 6.8, 6.1, 5.95, 2.6, COL["baseline"])
    txt(f, 7.05, 5.85, "We do not claim", 19, bold=True, color=COL["baseline"])
    bullets(f, 7.05, 5.3, ["Quantum advantage: the circuit is a product state, classically simulable.",
                           f"Beating logistic regression: we match it ({n['cv_diff'][0]:+.3f} ± {n['cv_diff'][1]:.3f} "
                           "on all frauds)."],
            15, width=5.5, gap=0.08)
    txt(f, 0.6, 3.2, "Next: real hardware noise (amplitude damping, readout error); more qubits at zero robustness cost.",
        15, bold=True, width=12.2)
    txt(f, 0.6, 2.55, "[1] Dinuț, Constantinescu, Alexandrescu, Electronics 15(11) 2489 (2026)  ·  Schreiber, Eisert, "
        "Meyer, PRL 131, 100803 (2023)  ·  Cerezo et al., Does provable absence of barren plateaus imply classical "
        "simulability?, arXiv:2312.09121  ·  Bowles, Ahmed, Schuld, arXiv:2403.07059 (2024)  ·  Schuld et al., PRA 99, "
        "032331 (2019)  ·  Havlíček et al., Nature 567, 209 (2019)  ·  Dal Pozzolo et al., IEEE SSCI (2015)",
        12, color=MUTED, width=12.2)
    pages.append(f)
    return pages


def main():
    n = load_numbers()
    pages = build(n)
    with PdfPages(OUT) as pdf:
        for f in pages:
            pdf.savefig(f, facecolor=f.get_facecolor())
            plt.close(f)
    print(f"wrote {OUT} ({len(pages)} slides); unfilled placeholders:",
          [k for k, v in PLACEHOLDERS.items() if v is None])


if __name__ == "__main__":
    main()
