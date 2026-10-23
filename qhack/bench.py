"""Quantum Hacking sweeps -> results/qhack.json + results/qhack.png

    python -m qhack.bench            # default sweeps
    python -m qhack.bench --n 50000  # more photons, smoother curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .attacks import (breidbart, detection_probability, intercept_resend,
                      min_sample_for_detection, trojan_horse, trojan_leak_probability)
from .bb84 import QBER_ABORT, run_bb84

RESULTS = Path("results")


def sweeps(n=20_000, p_noise=0.02, seed=0) -> dict:
    fr = np.linspace(0, 1, 11)
    ir = [run_bb84(n, p_noise, intercept_resend(f), seed=seed).as_dict() | {"fraction": float(f)}
          for f in fr]
    bb = [run_bb84(n, p_noise, breidbart(f), seed=seed).as_dict() | {"fraction": float(f)}
          for f in fr]
    iso = np.arange(0, 61, 5)
    th = [run_bb84(n, p_noise, trojan_horse(mu_in=1e3, isolation_db=float(a)), seed=seed).as_dict()
          | {"isolation_db": float(a), "p_leak": trojan_leak_probability(1e3, float(a))}
          for a in iso]
    det = [{"n_sample": int(k), "p_detect": detection_probability(int(k))} for k in
           [1, 2, 5, 10, 20, 50, 100]]
    return {"config": {"n": n, "p_noise": p_noise, "abort_threshold": QBER_ABORT},
            "baseline": run_bb84(n, p_noise, None, seed=seed).as_dict(),
            "intercept_resend": ir, "breidbart": bb, "trojan_horse": th, "detection": det,
            "min_sample_99.9%": min_sample_for_detection(0.999)}


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in ["intercept_resend", "breidbart"]:
        f = [r["fraction"] for r in res[name]]
        ax[0].plot(f, [r["qber_estimate"] for r in res[name]], "o-", label=f"QBER {name}")
        ax[0].plot(f, [r["eve_info"] for r in res[name]], "x--", label=f"Eve info {name}")
    ax[0].axhline(QBER_ABORT, color="red", ls=":", label="abort 11%")
    ax[0].set(xlabel="fraction attacked", title="Intercept-resend family")
    ax[0].legend(fontsize=7)
    a = [r["isolation_db"] for r in res["trojan_horse"]]
    ax[1].plot(a, [r["p_leak"] for r in res["trojan_horse"]], "o-", label="P(leak)")
    ax[1].plot(a, [r["qber_estimate"] for r in res["trojan_horse"]], "s-", label="QBER (unchanged!)")
    ax[1].plot(a, [r["key_rate"] for r in res["trojan_horse"]], "^-", label="secret key rate")
    ax[1].set(xlabel="optical isolation (dB)", title="Trojan horse vs isolator (mu_in=1e3)")
    ax[1].legend(fontsize=8)
    d = res["detection"]
    ax[2].semilogx([r["n_sample"] for r in d], [r["p_detect"] for r in d], "o-")
    ax[2].set(xlabel="public sample size", ylabel="P(detect)", title="Detecting full intercept-resend")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--p-noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    res = sweeps(a.n, a.p_noise, a.seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "qhack.json").write_text(json.dumps(res, indent=1))
    plot(res, RESULTS / "qhack.png")
    b = res["baseline"]
    print(f"no attack: QBER={b['qber_estimate']:.3f} rate={b['key_rate']:.3f}")
    for r in res["intercept_resend"][::5]:
        print(f"IR f={r['fraction']:.1f}: QBER={r['qber_estimate']:.3f} eve={r['eve_info']:.3f} "
              f"abort={r['aborted']}")
    for r in res["trojan_horse"][::3]:
        print(f"TH {r['isolation_db']:>4.0f}dB: p_leak={r['p_leak']:.3f} QBER={r['qber_estimate']:.3f} "
              f"rate={r['key_rate']:.3f}")
    print(f"sample size for 99.9% detection of full IR: {res['min_sample_99.9%']}")
    print("saved results/qhack.json and results/qhack.png")


if __name__ == "__main__":
    main()
