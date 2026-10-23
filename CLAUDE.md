# BreaQ 2026 hackathon repo

24h quantum hackathon (Sat 24 Oct 09:00 → Sun 25 Oct 14:00, Bucharest). Team of 2–3.
Tracks: **Quantum AI** (PennyLane, QML under realistic noise) and **Quantum Hacking**
(attacks on quantum devices/protocols). The organizers' starter kit goes in `kit/`;
its digest lives in `docs/KIT.md` — read that, not the kit, unless told otherwise.

## Layout
- `qai/` Quantum AI: `noise.py` (NoiseModel, channels), `data.py` (load + PCA + scale to [0,π],
  `file:` loader for kit CSV/NPZ), `model.py` (VQC, data re-uploading), `train.py` (Adam, CE),
  `mitigation.py` (noise-scaling ZNE), `baseline.py` (logreg/MLP/SVM + param counts),
  `bench.py` (end-to-end experiment → `results/qai_*.json/png`), `summary.py`.
- `qhack/` Quantum Hacking: `bb84.py` (BB84, sifting, QBER, Shor-Preskill rate),
  `attacks.py` (intercept-resend, Breidbart, Trojan horse + isolator, detection stats),
  `bench.py` (sweeps → `results/qhack.json/png`).
- `tests/` pytest, ~15s total. `docs/` plan, kit digest, pitch.

## Commands
- `make test-fast` — run after every change (skips slow tests, stops at first failure)
- `make qai-fast` — 15s smoke experiment; `make qai` — full (~2.5 min, run in background)
- `make qhack` — ~1s
- `make summary` — one-screen view of results. **Never read results/*.json directly.**

## Rules (keep tokens low)
- Read only the files the task needs; use Grep before Read. Do not scan the whole repo.
- Prefer editing existing modules over new files. Keep functions small and tested.
- Long runs (> 30s) go in the background; read only their last ~20 lines.
- Pytest: `python3 -m pytest -q -x <file>`; show failures only.
- After a working change: one-line commit (`git commit -am "qai: ..."`).
- No notebooks unless asked; scripts + plots in `results/` are the source of truth.
- Physics claims must have a test or a cited formula (e.g. IR QBER = f/4, BB84 threshold 11%).
- Every quantum result is reported next to the best classical baseline and its param count.

## Compact instructions
When compacting, keep: current track and goal, last `make summary` table, failing tests,
decisions made, files changed. Drop: file dumps, full logs, exploration.
