"""Final cost ledger, one row per method, built only from CSVs that the experiments already wrote.

Columns: training, model selection, threshold calibration, total executions, total shots (1024 per execution),
reduction vs the baseline (both its kit-formula total and its Tracker total), shot tax at p = 0.05.
Threshold calibration = one scoring pass over the validation set at the deployment noise level, needed to pick the
max-F1 threshold; it is the protocol's evaluation pass and identical for every method. MOCO can skip it: its
noiseless validation scores are a closed-form function of the classical angles (no circuit), and the threshold at
p follows from moco.transfer_threshold(t0, p, k=2). The one-off validation selection study is a footnote row, not
part of MOCO's total.

    python3 -m qai.experiments.ledger_final
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("results/task2")
SHOTS = 1024
EPOCHS_BASELINE = 30


def _row(df, method):
    return df[df.method == method].iloc[0]


def build():
    base = pd.read_csv(OUT / "ledger_baseline.csv")
    moco = pd.read_csv(OUT / "ledger_moco_single_10.csv")
    r2 = pd.read_csv(OUT / "ledger_entangled.csv")
    tax = pd.read_csv(OUT / "shot_tax.csv").query("p == 0.05").set_index("model").shot_tax

    b = _row(base, "baseline VQC")
    n_val = int(b["validation executions"]) // EPOCHS_BASELINE      # 56,962 validation rows
    m = _row(moco, "MOCO single 1:10")
    sel = _row(moco, "one-off selection study (fm x ratio on validation)")
    rows = [
        ("baseline VQC (kit formula, 65 per example)", b["train executions"], b["validation executions"],
         "baseline VQC", "ledger_baseline.csv"),
        ("baseline VQC (qml.Tracker, 41 per example)", b["train executions (Tracker)"], b["validation executions"],
         "baseline VQC", "ledger_baseline.csv"),
        ("MOCO single 1:10", m["train executions"], m["validation executions"], "MOCO",
         "ledger_moco_single_10.csv"),
    ]
    for variant, model in [("moco", "R2 (moco)"), ("random", "R2 (random)")]:
        r = _row(r2, f"R2 entangled ({variant})")
        rows.append((f"R2 entangled ({'warm start' if variant == 'moco' else 'random init'})",
                     r["train executions"], r["validation executions"], model, "ledger_entangled.csv"))

    out = []
    for name, train, select, model, src in rows:
        total = int(train) + int(select) + n_val
        out.append({"method": name, "training executions": int(train), "model-selection executions": int(select),
                    "threshold-calibration executions": n_val, "total executions": total,
                    "total shots": total * SHOTS, "shot tax at p=0.05": float(tax[model]),
                    "source": f"results/task2/{src}; shot tax: results/task2/shot_tax.csv"})
    led = pd.DataFrame(out)
    formula, tracker = led.loc[0, "total executions"], led.loc[1, "total executions"]
    led["reduction vs baseline (formula)"] = 1 - led["total executions"] / formula
    led["reduction vs baseline (Tracker)"] = 1 - led["total executions"] / tracker
    foot = {"method": "footnote: one-off selection study (fm x ratio on validation, not in MOCO's total)",
            "training executions": int(sel["train executions"]), "model-selection executions": int(sel["validation executions"]),
            "threshold-calibration executions": 0, "total executions": int(sel["total executions"]),
            "total shots": int(sel["total executions"]) * SHOTS, "source": "results/task2/ledger_moco_single_10.csv"}
    return pd.concat([led, pd.DataFrame([foot])], ignore_index=True), n_val


def markdown(led, n_val):
    def n(v):
        return "" if pd.isna(v) else f"{int(v):,}".replace(",", ".")

    def pct(v):
        return "" if pd.isna(v) else f"−{100 * v:.2f}%".replace(".", ",") if v > 0 else "—"

    lines = ["| Metodă | Antrenare | Selecția modelului | Calibrarea pragului | Total execuții | Total shots | "
             "Reducere vs baseline (formulă / Tracker) | Taxa în shots la p = 0.05 | Sursa |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in led.iterrows():
        tax = "" if pd.isna(r["shot tax at p=0.05"]) else f"{r['shot tax at p=0.05']:.3g}×".replace(".", ",")
        red = "" if pd.isna(r["reduction vs baseline (formula)"]) else \
            f"{pct(r['reduction vs baseline (formula)'])} / {pct(r['reduction vs baseline (Tracker)'])}"
        lines.append(f"| {r['method']} | {n(r['training executions'])} | {n(r['model-selection executions'])} | "
                     f"{n(r['threshold-calibration executions'])} | {n(r['total executions'])} | {n(r['total shots'])} | "
                     f"{red} | {tax} | `{r['source'].split(';')[0].replace('results/task2/', '')}` |")
    lines.append("")
    lines.append(f"Calibrarea pragului = o trecere de scorare a validării ({n(n_val)} de rânduri) la nivelul de zgomot "
                 "al dispozitivului, pentru pragul max-F1. E pasul de evaluare din protocol, identic pentru toate "
                 "metodele. **MOCO îl poate înlocui cu `moco.transfer_threshold`:** scorurile lui fără zgomot pe "
                 "validare sunt o formulă închisă a unghiurilor clasice (nicio execuție), iar pragul la p rezultă "
                 "analitic, t_p = ½ − λ²(½ − t₀). Așa MOCO coboară la 6.490 de execuții în total. Taxa în shots "
                 "înmulțește costul de inferență pe hardware la p = 0.05, pentru aceeași calitate a ordonării ca fără "
                 "zgomot; la baseline e o limită inferioară.")
    return "\n".join(lines)


def main():
    led, n_val = build()
    led.to_csv(OUT / "ledger_final.csv", index=False)
    md = markdown(led, n_val)
    (OUT / "ledger_final.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
