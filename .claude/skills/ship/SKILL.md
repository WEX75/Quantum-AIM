---
name: ship
description: Prepare the final submission - clean README with results, plots, reproduction commands and the pitch outline. Use in the last 2-3 hours.
---

1. `make test` must pass. Then run the final full experiments (background) and `make summary`.
2. Copy the final plots from `results/` into `docs/` (they are gitignored in results/).
3. Rewrite `README.md` results section: problem, approach, a results table (quantum vs best
   classical, param counts, noise levels), the plots, limitations (simulator vs hardware,
   noise model assumptions), exact reproduction commands.
4. Fill `docs/PITCH.md` (5 slides, 3 minutes): problem, idea, result plot, why it matters,
   what's next. One claim per slide, each backed by a number from the summary.
5. `git add -A && git commit -m "final submission" && git push`.
