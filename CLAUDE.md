# BreaQ 2026 hackathon repo

24h quantum hackathon (Sat 24 Oct 09:00 → Sun 25 Oct 14:00, Bucharest). Team of 2–3.
Tracks: **Quantum AI** (PennyLane + torch, QML under realistic noise) and **Quantum Hacking**
(attacks on quantum devices/protocols). The organizers' starter kit goes in `kit/`;
its digest lives in `docs/KIT.md` — read that, not the kit, unless told otherwise.

**Quantum AI data: credit-card fraud** (OpenML 42175 / Kaggle mlg-ulb, `data/creditcard.csv`,
284,807 rows, 0.17% fraud, columns Time, V1..V28, Amount, Class). Accuracy is meaningless here:
the primary metric is PR-AUC (also report recall, precision, F1, ROC-AUC, bal_acc) and the test
fraud rate (= chance PR-AUC). Never claim a result from accuracy alone.

## Layout
- `qai/` Quantum AI: `noise.py` (NoiseModel, channels), `data.py` (toy sets + `file:` loader),
  `fraud.py` (creditcard: stratified split on full data, balanced train subsample, test = all test
  frauds + 10x normals, ANOVA/PCA feature selection, `make_synthetic` for offline tests),
  `metrics.py`, `model.py` (VQC, re-uploading, `evaluate`), `train.py`, `mitigation.py` (ZNE),
  `baseline.py` (logreg/MLP/SVM/RF + logreg on the full imbalanced train set),
  `torch_hybrid.py` (HybridNet = Linear → quantum TorchLayer → Linear, `ClassicalTwin` of the same
  size, `set_noise`), `bench.py` (end-to-end → `results/qai_*.json/png`), `summary.py`,
  `experiments/` (one file per experiment, `exp_template.py`, `exp_hybrid_torch.py`).
- `qhack/` Quantum Hacking: `bb84.py`, `attacks.py` (intercept-resend, Breidbart, Trojan horse),
  `bench.py`.
- `scripts/get_data.py` downloads/validates the fraud CSV. `tests/` ~20s. `docs/` plan, ideas,
  pitch, git workflow, log.

## Commands (Ubuntu; `make <target>`, or `python3 run.py <target>`)
- `make test-fast` — after every change (skips slow tests, stops at first failure)
- `make fraud-fast` (~20s) / `make fraud` (minutes, background) — VQC on fraud data
- `make hybrid-fast` — torch hybrid vs classical twin on fraud data
- `make qai-fast` — toy dataset smoke run; `make qhack` — ~1s
- `make summary` (`--metric f1` etc.) — one-screen results. **Never read results/*.json directly.**

## Rules (keep tokens low)
- Use `python3` (inside the activated .venv, `python` works too).
- Read only the files the task needs; use Grep before Read. Do not scan the whole repo or `data/`.
- Prefer editing existing modules over new files. Keep functions small and tested.
- Long runs (> 30s) go in the background; read only their last ~20 lines.
- Pytest: `python3 -m pytest -q -x <file>`; show failures only.
- No notebooks unless asked; scripts + plots in `results/` are the source of truth.
- Every quantum result is reported next to the best classical baseline (incl. `logreg_fulltrain`)
  and its param count. Physics claims need a test or a cited formula.

## Git (team of 2–3, see docs/GIT.md)
- Never commit on `main`. Branch `<owner>/<topic>`, start with `make sync`, finish with `make merge`.
- New experiments = new file `qai/experiments/exp_<topic>.py`. Shared modules: small edits only.
- File owners: story owner edits `docs/PITCH.md`, `README.md`, slides; code owner edits code.
  `docs/LOG.md` is append-only (union merge), everyone appends.
- On a merge/rebase conflict: keep both sides' intent, run `make test-fast`, then continue.

## Compact instructions
When compacting, keep: current track and goal, last `make summary` table, failing tests,
decisions made, files changed. Drop: file dumps, full logs, exploration.
