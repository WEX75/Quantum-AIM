# BreaQ 2026 — Quantum AI & Quantum Hacking toolkit

Team repo for the [BreaQ Hackathon 2026](https://roqteam.ro/breaq-hackathon.html)
(24–25 Oct, CAMPUS Research Institute, Bucharest).

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # torch CPU + pennylane + the kit's stack
make data           # downloads the fraud dataset (OpenML 42175) to data/creditcard.csv
make check          # prints READY when libraries + data are in place
make test           # 33 tests, ~20s
make fraud-fast     # quantum classifier on fraud data under noise, ~20s
make hybrid-fast    # torch hybrid vs same-size classical twin
make qai-fast       # toy dataset, ~15s
make qhack          # BB84 attacks, ~1s
make summary
```
(Without `make`, e.g. on Windows: `python run.py <same target>`.)

## Quantum AI — fraud detection with quantum models under realistic noise (PennyLane + torch)
Data: credit-card fraud (284,807 transactions, 0.17% fraud). Balanced training subsample, test
with all held-out frauds + 10x normals, primary metric PR-AUC (accuracy is meaningless here),
strong classical reference trained on the full imbalanced train split. See `qai/fraud.py`.

### Toolkit
Hybrid VQC with data re-uploading, a configurable noise model (depolarizing, two-qubit
depolarizing, amplitude/phase damping, readout), noise-aware training, noise-scaling
extrapolation (ZNE) and classical baselines with parameter counts.

Pre-hackathon result (moons, 4 qubits × 3 layers, 40 params; `make qai`):

| noise p | trained noiseless | trained under noise (p=0.05) | noiseless + ZNE |
|---|---|---|---|
| 0.00 | 1.000 | 0.967 | – |
| 0.02 | 0.989 | 0.967 | 1.000 |
| 0.05 | 0.933 | 0.967 | 0.978 |
| 0.10 | 0.511 | **0.967** | 0.622 |

Classical: logistic regression 0.911 (5 params), MLP[8] 0.867 (49), RBF-SVM 0.967.

![qai](docs/qai_moons.png)

Use your own data: `python -m qai.bench --dataset file:data/kit.csv --qubits 6`.

## Quantum Hacking — attacks on BB84 QKD
Exact numpy simulation of BB84 with sifting, public-sample QBER estimation, 11% abort
threshold and Shor-Preskill key rate; intercept-resend, Breidbart and Trojan-horse attacks
with an optical-isolator defence and detection statistics.

![qhack](docs/qhack.png)

## Layout
`qai/` · `qhack/` · `tests/` · `docs/` (plan, pitch, log, kit digest) · `kit/` (starter kit)
· `.claude/skills/` (`/kickoff`, `/exp`, `/ship` for Claude Code) · `CLAUDE.md`.

## Limitations
Simulator only; noise channels are an idealised model, not a device calibration. ZNE here
scales channel strengths directly (on hardware: unitary folding).
