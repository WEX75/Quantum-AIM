# Sweep de încredere pentru triajul adaptiv (doar validare)

**Configurația:** MOCO cu encodarea kitului, intervalul plug-in Bernstein, b = 16, S_max = 4.096. Am variat α de la 0,001 la 0,49, adică încrederea de la 0,999 la 0,51, pe setul complet de validare, cu 5 seed-uri și p ∈ {0.01, 0.05}. Valoarea aleasă rămâne α = 0,05.

**Shots:** costul scade de 4,3 ori de la încrederea 0,999 la 0,51: la p = 0.05 de la 405 ± 93 la 95 ± 20 de shots per tranzacție, la p = 0.01 de la 327 la 77. Ipoteza C1, scrisă înainte de rulare, anticipa doar ~2×.

**F1:** rămâne în 0,01 de valoarea de la α = 0,05 (0,784 la p = 0.05) până la încrederea 0,7, deci C2 se confirmă. La 0,51 scade cu 0,02 (0,762 la p = 0.05).

**Garanția:** rata empirică de decizii greșite oprite devreme e cel mult 0,00018 pe orice seed (media pe seed-uri e ≤ 0,00012), mult sub α la orice nivel (C3 confirmată).

**Concluzie:** încrederea e un buton de cost, nu de calitate. Între 0,95 și 0,7, costul scade cu ~43% (209 -> 119 shots la p = 0.05; 169 -> 97 la p = 0.01) fără pierdere de F1. Totuși, păstrăm α = 0,05, ales înainte după regula de selecție, iar sweep-ul nu se folosește pentru nicio alegere.

**Verificare:** rândurile cu α = 0,05 reproduc exact validarea din `exp_adaptive_shots` (diferență 0).

Sursa: `qai/experiments/exp_confidence_sweep.py`, `results/task2/confidence_sweep.csv`, figura `results/task2/figures/l_confidence_sweep.png`.
