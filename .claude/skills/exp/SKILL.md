---
name: exp
description: Run one experiment change end-to-end (branch, implement, test, run, compare to baseline, log, commit). Use for each idea in the TODO.
---

Argument: the idea to try (e.g. "noise-aware training with p_train=0.1 on the kit dataset").

0. `make sync`. If on `main`, create a branch: `git checkout -b <owner>/<short-topic>`
   (owner = the branch prefix already used in this clone, ask once if unknown).
1. State the hypothesis and the metric in one line.
2. Put new experiment logic in `qai/experiments/exp_<topic>.py` (copy `exp_template.py`).
   Touch shared modules (`model.py`, `noise.py`, `data.py`, `train.py`, `bench.py`) only for a
   small, reusable change (new arg with a default), never a rewrite.
3. `make test-fast`. Fix until green. Add one test if you added logic.
4. Run the fast variant first; only if it is promising, run the full variant in the background.
5. Report: before vs after, best classical baseline, param counts.
6. Append one line to `docs/LOG.md`:
   `HH:MM | <owner> | idea | result | keep/drop | PITCH: <claim with the number>`
   Do NOT edit `docs/PITCH.md` (the story owner copies PITCH lines from the log).
7. If kept: commit on the branch and `make merge`. If dropped: commit only the LOG line
   (`git checkout -- . ':!docs/LOG.md'` first), then `make merge`. Then stop.
