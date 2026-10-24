# Starter kit digest — "Fraud on a Quantum Budget"
Sources: `kit/Fraud on a Quantum Budget - Task.pdf`, `kit/task1_guided_vqc_fraud_detection.ipynb` (cells 81–127).

## Task
Build a PennyLane VQC (classical parts in PyTorch) that detects card fraud on ULB creditcard data and keeps
working under simulated depolarizing noise, with a ledger of what training/validation would cost on hardware.
Task 1 = guided notebook (not judged). **Task 2 = judged**: baseline VQC + >=1 noise-aimed improvement +
classical reference + cost ledger, all with the official protocol over >=5 seeds.
- **Deliverables**: (1) code runnable end to end with fixed seeds, dataset → common split → final tables;
  (2) slides (.pdf/.pptx): pipeline, circuit + gate count + noise locations, improvement + why, results table
  (baseline / ours / classical at every p, mean ± std), cost ledger vs baseline, what did not work / next.
- **Deadline**: slides uploaded within the 24 competition hours (confirm cutoff; event ends Sun 14:00); talk uses that version.

## Data
- `data/creditcard.csv` (OpenML 42175 == Kaggle; NOT 1597): 284,807 × 31, `Class` 492 frauds (0.17%).
- **Common split (mandatory)**: stratified 60/20/20, `random_state=42`, on the full data *before* balancing:
  `train_test_split(test_size=0.2)` then `(test_size=0.25)` on the rest. Test = 56,962 rows / 98 frauds,
  val = 99 frauds. Saved as `split_indices.npz` (keys train/val/test = df index).
- Any resampling/scaling/PCA fitted on train only, documented. Results on the untouched test set.

## Protocol (`evaluate_under_noise`, nb cell 125)
- Noise: `qml.transforms.decompose(gate_set={RX,RY,RZ,Rot,Hadamard,CNOT,CZ})`, then
  `qml.noise.insert(default.mixed, DepolarizingChannel, p, position="all")` → one channel per qubit of every
  gate. p ∈ {0, 0.001, 0.01, 0.05}; exact density matrices.
- Threshold = max-F1 on **validation scores at the same p** (not an improvement by itself).
- Metrics: test **AUPRC** (headline at p=0.01, 0.05), AUROC, Spearman ρ(noisy, clean test scores),
  precision/recall/F1/MCC at threshold. Mean ± std over seeds [0..4] (seed drives undersample, PCA, init).

## Tools / constraints
PennyLane 0.45 (we have 0.45.1), torch (2.14 here), sklearn only for preprocessing/baselines/metrics.
Classification must come from the circuit (classical layers only pre-/post-scale). Mitigation allowed if
its executions go in the ledger. Shots for ledger: 1024/execution. No qubit cap stated (baseline: 4).

## Kit baseline
Undersample 1:1 (295+295) → StandardScaler(Time, Amount) → PCA(4) → MinMax to [0, π] →
ZZ feature map (2 reps, full entanglement, CNOT-RZ-CNOT) → EfficientSU2-style ansatz (3 reps RY/RZ + linear
CNOT, 32 params) → ⟨Z0⟩, p_fraud=(1-⟨Z0⟩)/2. Adam lr 0.05, BCE, batch 32, 30 epochs, keep best val AUPRC.
93 native gates, depth 45, **126 noise locations**; only 19/32 params can reach qubit 0.
- Noiseless test AUPRC **0.58 ± 0.05** (5 seeds) vs logreg on same 4 angles **0.70 ± 0.01**.
- p=0.01: ranking kept, threshold stops catching frauds; p=0.05: Spearman ≈ 0.65.
- Cost: 30 ep × 590 × 65 = **1,150,500** executions (1.18 B shots); full-val every epoch +1,708,860.
- Paper [1] (Dinuț 2026): VQC AUPRC 0.494 ± 0.139 vs tuned classical 0.60–0.74.

## Judging
AUPRC at p=0.01/0.05 vs baseline & classical; robustness (ρ, threshold metrics vs p); cost (same perf for
less counts); circuit rationale (fewer qubits/depth); seed stability; sound design (no leakage, val
thresholds, no cherry-picking); originality (own method > known one); code clarity; ablations + honest
negative results; understanding of the solution.

## Repo mapping
| Need | Have | Status |
|---|---|---|
| common split | `qai/fraud.py` | lipsește: own split, test = frauds+10× normals (9% fraud) → not comparable |
| encoder | `qai/fraud.py` ANOVA/PCA | atenție: no Time/Amount scaler → PCA4 → [0,π] variant |
| noise | `qai/noise.py` manual channels | lipsește: not decompose+insert-after-every-gate |
| VQC | `qai/model.py` reupload, `torch_hybrid.py` | atenție: no ZZ+EfficientSU2 baseline; not BCE/Adam per kit |
| metrics | `qai/metrics.py` | atenție: missing MCC, Spearman, val-max-F1 threshold |
| ZNE | `qai/mitigation.py` | atenție: bound to our NoiseModel; kit uses `qml.noise.mitigate_with_zne` |
| baselines | `qai/baseline.py` | gata: logreg etc. (must be fed the same 4 angles) |
| cost ledger, multi-seed table | — | lipsește: missing |
