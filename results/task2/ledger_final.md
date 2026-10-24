| Metodă | Antrenare | Selecția modelului | Calibrarea pragului | Total execuții | Total shots | Reducere vs baseline (formulă / Tracker) | Taxa în shots la p = 0.05 | Sursa |
|---|---|---|---|---|---|---|---|---|
| baseline VQC (kit formula, 65 per example) | 1.150.500 | 1.708.860 | 56.962 | 2.916.322 | 2.986.313.728 | — / — | 504× | `ledger_baseline.csv` |
| baseline VQC (qml.Tracker, 41 per example) | 725.700 | 1.708.860 | 56.962 | 2.491.522 | 2.551.318.528 | −14,57% / — | 504× | `ledger_baseline.csv` |
| MOCO single 1:10 | 6.490 | 0 | 56.962 | 63.452 | 64.974.848 | −97,82% / −97,45% | 1,32× | `ledger_moco_single_10.csv` |
| R2 entangled (warm start) | 298.835 | 113.920 | 56.962 | 469.717 | 480.990.208 | −83,89% / −81,15% | 1,58× | `ledger_entangled.csv` |
| R2 entangled (random init) | 289.100 | 113.920 | 56.962 | 459.982 | 471.021.568 | −84,23% / −81,54% | 1,97× | `ledger_entangled.csv` |
| footnote: one-off selection study (fm x ratio on validation, not in MOCO's total) | 0 | 390.661 | 0 | 390.661 | 400.036.864 |  |  | `ledger_moco_single_10.csv` |

Calibrarea pragului = o trecere de scorare a validării (56.962 de rânduri) la nivelul de zgomot al dispozitivului, pentru pragul max-F1. E pasul de evaluare din protocol, identic pentru toate metodele. **MOCO îl poate înlocui cu `moco.transfer_threshold`:** scorurile lui fără zgomot pe validare sunt o formulă închisă a unghiurilor clasice (nicio execuție), iar pragul la p rezultă analitic, t_p = ½ − λ²(½ − t₀). Așa MOCO coboară la 6.490 de execuții în total. Taxa în shots înmulțește costul de inferență pe hardware la p = 0.05, pentru aceeași calitate a ordonării ca fără zgomot; la baseline e o limită inferioară.
