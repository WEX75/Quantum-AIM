# Figurile pentru slide-uri

Generate de `python3 -m qai.experiments.task2_figures`, doar din CSV-urile din `results/task2/`. Fiecare figură există în PNG (200 dpi) și SVG. Numerele de slide sunt cele din `docs/PROJECT.md`, secțiunea 9.

| Figură | Slide | Ce spunem |
|---|---|---|
| `h_selection` | 3 (Pipeline-ul) | „Configurația am ales-o doar pe validare, pe 5 seed-uri: single 1:10 are cea mai bună medie, iar dintre variantele la egalitate e cea mai ieftină; testul l-am atins o singură dată.” |
| `f_circuits` | 4 (Circuitul) | „După fiecare poartă vine un canal de zgomot pe fiecare qubit atins: MOCO are 8 astfel de locuri, baseline-ul 126.” |
| `a_auprc_protocol` | 7 (Rezultate, protocolul oficial) | „Sub protocolul oficial, MOCO rămâne la 0.716 la orice nivel de zgomot, la nivelul regresiei logistice pe aceleași intrări, în timp ce baseline-ul cade la 0.405 la p = 0.05.” |
| `b_spearman_protocol` | 7 (Rezultate, protocolul oficial) | „Ordonarea tranzacțiilor la MOCO nu se schimbă deloc sub zgomot, ρ = 1, pentru că zgomotul doar comprimă scorurile spre ½; la baseline ρ scade la 0.69.” |
| `d_auprc_1024_shots` | 8 (Răspunsul la hint) | „Cu 1024 de shots reale la p = 0.05, MOCO păstrează 0.705, R2 0.643, iar baseline-ul ajunge la 0.007, adică la nivelul șansei.” |
| `e_shot_tax` | 8 (Răspunsul la hint) | „Zgomotul e o taxă în shots: ca să păstreze același raport semnal/zgomot, MOCO plătește 1,32x la p = 0.05, exact λ⁻⁴, iar baseline-ul cel puțin 504x.” |
| `c_ledger` | 9 (Cost ledger) | „Antrenarea, selecția și calibrarea pragului costă la MOCO 63.452 de execuții față de 2,9 milioane la baseline, adică de 46 de ori mai puțin.” |
| `g_ablations` | 10 (Ce n-a mers) | „Cu aceeași antrenare MOCO, fiecare CNOT adăugat costă AUPRC și ordonare: encodarea ZZ a kitului rămâne la 0.45 chiar cu citirea optimă.” |

Lipsesc, se generează automat când apar CSV-urile (scriptul le raportează ca „missing”):
- `i_cv_all_frauds`, din `results/task2/cv/cv_per_fold.csv` (CV pe toate cele 492 de fraude);
- `j_adaptive_shots`, din `results/task2/adaptive_shots.csv` (experimentul cu shots adaptive).
