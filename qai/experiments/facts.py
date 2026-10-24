"""Facts the slides need that no experiment CSV holds yet, written to CSVs so every number on a slide has a source:
  results/task2/circuits.csv       native gates, two-qubit gates, noise locations and depth of every circuit
                                   (qml.specs on the circuit definitions, no simulation)
  results/task2/dataset_split.csv  rows and frauds of the full dataset and of the common split

    python3 -m qai.experiments.facts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qai import kit_vqc as kit
from qai import moco
from qai.experiments import exp_entangled as r2
from qai.protocol import common_split, gate_stats, load_df

OUT = Path("results/task2")


def circuits():
    rows = [("baseline VQC (kit)", gate_stats(kit.vqc, np.zeros(4), kit.init_weights(0).detach())),
            ("R1 light-cone baseline", gate_stats(kit.vqc_lightcone, np.zeros(4), kit.init_weights(0).detach())),
            ("R2 entangled", gate_stats(r2.circuit, np.zeros(4), torch.zeros(r2.N_ROT + r2.N, dtype=torch.float64)))]
    for fm in ["single", "pair", "linear_cnot", "kit_zz"]:
        params = dict(rot=np.zeros((4, 3)), w=np.ones(4) / 4, fm=fm)
        rows.append((f"MOCO {fm}", gate_stats(moco.make_circuit(params), np.zeros(4), None)))
    return pd.DataFrame([{"circuit": n, **s} for n, s in rows])


def dataset_split():
    df = load_df()
    s = common_split(df)
    parts = [("full dataset", df["Class"].to_numpy()), ("train", s.y_train), ("validation", s.y_val), ("test", s.y_test)]
    return pd.DataFrame([{"part": n, "rows": len(y), "frauds": int(y.sum()), "fraud_rate": y.mean()} for n, y in parts])


def main():
    c, d = circuits(), dataset_split()
    c.to_csv(OUT / "circuits.csv", index=False)
    d.to_csv(OUT / "dataset_split.csv", index=False)
    print(c.to_string(index=False))
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
