"""Template: copy to exp_<topic>.py, change run(), keep the output format.

    python3 -m qai.experiments.exp_template --fast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qai.bench import run
from qai.summary import summarize


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(layers=2, epochs=8, max_samples=120, noise_levels=(0.0, 0.05, 0.1)) if a.fast else {}
    res = run(**kw)  # <- change the experiment here
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"{Path(__file__).stem}.json"
    out.write_text(json.dumps(res, indent=1))
    print(summarize(res))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
