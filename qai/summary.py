"""Compact text summary of results/*.json, so nobody (human or Claude) has to read raw JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(res: dict) -> str:
    c = res["config"]
    lines = [f"[{c['dataset']}] {c['qubits']}q x {c['layers']}L {c['encoding']}, "
             f"VQC params={c['vqc_params']}, train/test={c['n_train']}/{c['n_test']}, "
             f"classes={c['n_classes']}, p_train={c['p_train']}"]
    lines.append("classical: " + ", ".join(
        f"{k}={v['test_acc']:.3f} ({v['n_params']}p)" for k, v in res["classical"].items()))
    keys = ["clean_trained", "noise_aware", "clean_trained+zne"]
    lines.append("p      " + "  ".join(f"{k:>18}" for k in keys))
    for r in res["sweep"]:
        lines.append(f"{r['p']:<6} " + "  ".join(
            f"{r[k]:>18.3f}" if k in r else f"{'-':>18}" for k in keys))
    if "train_seconds" in res:
        lines.append(f"train time (s): {res['train_seconds']}")
    return "\n".join(lines)


def main():
    files = [Path(p) for p in sys.argv[1:]] or sorted(Path("results").glob("qai_*.json"))
    if not files:
        print("no results yet: run `make qai-fast`")
    for f in files:
        print(summarize(json.loads(f.read_text())), "\n")


if __name__ == "__main__":
    main()
