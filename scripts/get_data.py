"""Download the credit-card fraud dataset once, at home, so the venue Wi-Fi doesn't matter.

    python3 scripts/get_data.py            # OpenML 42175 -> data/creditcard.csv
    python3 scripts/get_data.py --check    # only verify an existing data/creditcard.csv

Alternative: download creditcard.csv from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and put it in data/creditcard.csv.
Both sources have the columns Time, V1..V28, Amount, Class (284,807 rows, 492 frauds).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUT = Path("data/creditcard.csv")


def check(path: Path = OUT) -> bool:
    import pandas as pd
    if not path.exists():
        print(f"missing: {path}")
        return False
    df = pd.read_csv(path)
    cls = df["Class"].astype(str).str.strip("'\" ").astype(float).astype(int)
    ok = {"Time", "Amount", "Class"} <= set(df.columns) and all(f"V{i}" in df for i in range(1, 29))
    print(f"{path}: {df.shape[0]:,} rows x {df.shape[1]} cols, frauds={int(cls.sum())} "
          f"({cls.mean():.4%}) -> {'OK' if ok else 'UNEXPECTED COLUMNS'}")
    return ok


def download(path: Path = OUT) -> None:
    from sklearn.datasets import fetch_openml
    print("downloading OpenML 42175 (~150 MB, a minute or two)...")
    d = fetch_openml(data_id=42175, as_frame=True, parser="auto")
    df = d.frame
    target = d.target_names[0] if getattr(d, "target_names", None) else df.columns[-1]
    if target != "Class":
        df = df.rename(columns={target: "Class"})
    df["Class"] = df["Class"].astype(str).str.strip("'\" ").astype(float).astype(int)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check and not OUT.exists():
        download()
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
