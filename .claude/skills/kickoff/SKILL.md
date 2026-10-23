---
name: kickoff
description: Digest the organizers' starter kit in kit/ into docs/KIT.md and map it onto this repo. Use once, right after the challenge is announced.
---

1. List `kit/` (names and sizes only). Read the challenge README/PDF and at most one notebook.
   Do not open data files beyond their first 5 lines.
2. Write `docs/KIT.md` (max 60 lines):
   - Task in 3 sentences, exact deliverables, evaluation metric, deadline, submission format
   - Dataset: path, shape, labels, train/test split given or not
   - Required tools / constraints (PennyLane version, devices, noise model, shot budget, qubit limit)
   - Baseline provided by the kit and its score
   - Judging criteria if stated
3. Map it onto this repo: which existing module covers what (`qai/data.py` file: loader,
   `qai/noise.py`, `qai/bench.py` ...), what is missing. List the gaps as a numbered TODO
   with an estimate in hours, ordered by impact on the judged metric.
4. Get the kit baseline running through our pipeline if possible (`python3 run.py qai-fast`-sized run),
   and report the numbers with `python3 run.py summary`.
5. Commit: `git add -A && git commit -m "kit: digest + baseline"`.
Stop and show the TODO; do not start implementing.
