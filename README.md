# MOCO: Measure Once, Compile Once

BreaQ 2026, Quantum AI track, Task 2 "Fraud on a Quantum Budget". Team: Darius Tatomir, Deliu Georgiana.

A variational quantum classifier for credit-card fraud (ULB dataset, 284,807 transactions, 0.17% fraud) whose ranking
cannot be changed by depolarizing noise, trained with 2 circuit executions per training transaction:

1. **Measure once.** The feature map is RY(x_q) on 4 qubits. The reduced Bloch vector of every training transaction
   is measured once (X and Z bases).
2. **Train convexly.** After a trainable local Rot, <Z_q> is linear in the measured Bloch vector (Heisenberg
   picture), so training is a logistic regression on measured data: convex, solved in PyTorch, global optimum, no
   circuit in the loop.
3. **Compile back.** The coefficients become Rot angles and observable weights; the circuit classifies,
   p_fraud = (1 - <H>) / 2.
4. **Noise cannot reorder.** No CNOT and 2 gates per qubit: under the official noise model every score is squashed
   by the same factor, s_noisy = 1/2 - lambda^2 (1/2 - s_ideal), lambda = 1 - 4p/3. AUPRC, AUROC and Spearman rho are
   identical at every p, and the threshold moves analytically.

We do not claim quantum advantage: the circuit is a product state, and MOCO matches (does not beat) a logistic
regression on the same inputs. The contribution is robustness, stability and cost.

Everything is evaluated with the organizers' protocol (`qai/protocol.py`, copied cell by cell from the kit notebook):
common split (stratified 60/20/20, seed 42), depolarizing noise after every native gate, max-F1 threshold on
validation at the same noise level, 5 seeds.

## Results (test set, mean ± std over 5 seeds; from `results/task2/TABLES.md`)

| Model | AUPRC p = 0 | p = 0.01 | p = 0.05 | Spearman rho, p = 0.05 | F1, p = 0.05 |
|---|---|---|---|---|---|
| Baseline VQC (kit: ZZ map + EfficientSU2, 93 gates) | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 | 0.686 ± 0.145 | 0.537 ± 0.184 |
| R2: shallow entangled VQC, random init | 0.635 ± 0.018 | 0.636 ± 0.017 | 0.638 ± 0.015 | 0.999 ± 0.001 | 0.760 ± 0.021 |
| R2: shallow entangled VQC, MOCO warm start | 0.667 ± 0.010 | 0.667 ± 0.010 | 0.668 ± 0.009 | 0.999 ± 0.001 | 0.798 ± 0.009 |
| **MOCO (8 gates, 0 CNOT)** | **0.716 ± 0.007** | **0.716 ± 0.007** | **0.716 ± 0.007** | **1.000 ± 0.000** | **0.800 ± 0.019** |
| Logistic regression, same inputs as MOCO (1:10) | 0.710 ± 0.007 | 0.710 ± 0.007 | 0.710 ± 0.007 | — | 0.794 ± 0.023 |
| Logistic regression, same inputs as the baseline (1:1) | 0.703 ± 0.010 | 0.703 ± 0.010 | 0.703 ± 0.010 | — | 0.791 ± 0.019 |
| MOCO, robust encoding q = 0.5% (comparator, not selected) | 0.738 ± 0.006 | 0.738 ± 0.006 | 0.738 ± 0.006 | 1.000 ± 0.000 | 0.783 ± 0.024 |
| Logistic regression, same robust inputs | 0.731 ± 0.004 | 0.731 ± 0.004 | 0.731 ± 0.004 | — | 0.785 ± 0.020 |

Ablations (same MOCO training, only the feature map changes), test AUPRC at p = 0 / p = 0.05: pair (0 CNOT)
0.707 ± 0.024 / 0.707 ± 0.024; linear CNOT chain 0.675 ± 0.034 / 0.667 ± 0.032; the kit's ZZ map 0.440 ± 0.104 /
0.450 ± 0.114.

**Shot tax** (diagnostic, measured on test). With finite shots, the measured score compression c(p) has to be paid
in shots: 1/c^2 more shots for the same signal-to-noise. At p = 0.05: MOCO 1.32x (exactly lambda^-4), R2 warm start
1.58x, baseline at least 504x (a lower bound: it also loses ranking with infinite shots).

**Adaptive triage** (inference on hardware, p = 0.05): MOCO with batched shots and an anytime-valid plug-in Bernstein
interval needs 208.5 shots per transaction instead of 1024 (`ledger_adaptive.csv`); the baseline hits the cap
(4088.8 of 4096).

## Cost ledger (circuit executions on hardware, 1024 shots each; `results/task2/ledger_final.csv`)

| Method | Training | Model selection | Threshold calibration | Total | Reduction vs baseline |
|---|---|---|---|---|---|
| Baseline VQC, kit formula (65 per example) | 1,150,500 | 1,708,860 | 56,962 | 2,916,322 | — |
| Baseline VQC, qml.Tracker count (41 per example) | 725,700 | 1,708,860 | 56,962 | 2,491,522 | -14.6% |
| **MOCO** | **6,490** | **0** | 56,962 | **63,452** | **-97.8%** |
| R2 warm start | 298,835 | 113,920 | 56,962 | 469,717 | -83.9% |

Kit definition (training + model selection): baseline 2,859,360 vs MOCO 6,490 (-99.8%). MOCO's threshold can be
transferred analytically, so threshold calibration can be replaced by a noise meter (see `docs/PROJECT.md`).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
make install                 # torch CPU + PennyLane 0.45 + the kit's stack
make data                    # data/creditcard.csv (OpenML 42175)
make task2-test              # protocol, physics and adaptive-interval tests (~2 min)
make task2-experiments       # all experiments, 5 seeds, cached per seed in results/task2/ (hours)
make task2-light             # tables, figures and slides from the CSVs (minutes)
# or everything at once:
make task2
```

`make task2-light` runs, in order: `shot_tax`, `ledger_final`, `facts`, `task2_report` (writes
`results/task2/TABLES.md`), `task2_figures` (full and `--compact`), then `presentation/build_slides.py`
(writes `presentation/moco_slides.pdf`). Smoke run on a 10% sample: `make task2-fast`.

## Layout

| Path | Content |
|---|---|
| `qai/protocol.py` | The kit's evaluation protocol, each function citing its notebook cell |
| `qai/kit_vqc.py` | Kit baseline VQC (R0) and its light-cone pruned variant (R1) |
| `qai/moco.py` | MOCO: Bloch vectors, convex fit, compilation, threshold transfer, Tracker check |
| `qai/adaptive.py` | Anytime-valid intervals, batched triage, noise meter |
| `qai/experiments/` | One script per experiment (`exp_*.py`) and the light steps (`shot_tax`, `ledger_final`, `facts`, `task2_report`, `task2_figures`) |
| `tests/` | `test_protocol.py`, `test_adaptive.py` and the older toolkit tests |
| `results/task2/` | Per-seed CSVs, ledgers, `TABLES.md`, `figures/` |
| `presentation/` | `moco_slides.pdf`, its generator and `SPEAKER_NOTES.md` |
| `task1/` | The completed Task 1 notebook and its README |
| `docs/PROJECT.md` | Full project document: method, every result with its source, honest negative results |
| `kit/` | The organizers' starter kit |
