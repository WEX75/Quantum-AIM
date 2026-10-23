---
name: exp
description: Run one experiment change end-to-end (implement, test, run, compare to baseline, commit). Use for each idea in the TODO.
---

Argument: the idea to try (e.g. "noise-aware training with p_train=0.1 on the kit dataset").

1. State the hypothesis and the metric in one line.
2. Make the smallest code change that tests it (CLI flag or function arg, not a new script).
3. `python run.py test-fast`. Fix until green. Add one test if you added logic.
4. Run the fast variant first; only if it is promising, run the full variant in the background.
5. `python run.py summary` and report: before vs after, best classical baseline, param counts.
6. Append one line to `docs/LOG.md`: `HH:MM | idea | result | keep/drop`.
   If kept, put the number in the matching row of the "Cifre" table in `docs/PITCH.md`
   (add a row if the result supports a new claim). The pitch is half the grade.
7. Commit if kept, `git checkout -- .` if dropped. Then stop.
