"""Compact text summary of results/*.json, so nobody has to read raw JSON.

    python3 -m qai.summary                     # all results, primary metric
    python3 -m qai.summary --metric f1         # another metric
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _val(v, metric):
    return v if isinstance(v, (int, float)) else v.get(metric, float("nan"))


def summarize(res: dict, metric: str | None = None) -> str:
    c = res["config"]
    metric = metric or c.get("metric", "acc")
    head = (f"[{c['dataset']}] {c['qubits']}q x {c['layers']}L {c['encoding']}, "
            f"VQC params={c['vqc_params']}, train/test={c['n_train']}/{c['n_test']}, "
            f"classes={c['n_classes']}, p_train={c['p_train']}, metric={metric}")
    lines = [head]
    if "test_fraud_rate" in c:
        lines.append(f"test fraud rate={c['test_fraud_rate']:.3f} (chance PR-AUC ~ this), "
                     f"features={c.get('features')}")
    lines.append("classical: " + ", ".join(
        f"{k}={_val(v, metric):.3f} ({v['n_params']}p)" for k, v in res["classical"].items()))
    keys = ["clean_trained", "noise_aware", "clean_trained+zne"]
    lines.append("p      " + "  ".join(f"{k:>18}" for k in keys))
    for r in res["sweep"]:
        lines.append(f"{r['p']:<6} " + "  ".join(
            f"{_val(r[k], metric):>18.3f}" if k in r else f"{'-':>18}" for k in keys))
    if "train_seconds" in res:
        lines.append(f"train time (s): {res['train_seconds']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--metric", default=None)
    a = ap.parse_args(argv)
    files = [Path(p) for p in a.files] or sorted(Path("results").glob("qai_*.json"))
    if not files:
        print("no results yet: run `make qai-fast`")
    for f in files:
        print(summarize(json.loads(f.read_text()), a.metric), "\n")


if __name__ == "__main__":
    main()
