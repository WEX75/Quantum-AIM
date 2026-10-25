# Unde am implementat X? (pentru întrebările juriului)

Fiecare răspuns indică fișierul și linia (linkurile merg pe GitHub și în editor), apoi ce produce și unde găsești rezultatul. Toate căile sunt relative la rădăcina repo-ului (`breaq-2026/`).

Regula generală: **`qai/protocol.py` e protocolul organizatorilor, copiat celulă cu celulă din notebook-ul kitului; `qai/moco.py` e metoda noastră; `qai/experiments/` are câte un script per experiment; rezultatele sunt în `results/task2/`.**

---

## 1. Protocolul oficial (copiat din kit)

| Întrebarea | Unde | Ce face |
|---|---|---|
| Unde e split-ul comun 60/20/20, seed 42? | [qai/protocol.py:52](../qai/protocol.py#L52) `common_split` | Celula 35 din kit. Scrie `data/split_indices.npz` și verifică la fiecare rulare că e același. Notebook-ul Task 1 produce exact aceiași indici (verificat cu `np.array_equal`, vezi `task1/README.md`). |
| Unde faceți undersampling-ul? | [qai/protocol.py:69](../qai/protocol.py#L69) `undersample` | Celula 38. Păstrează toate fraudele din train și ia `neg_per_pos` tranzacții legitime per fraudă (kitul: 1, MOCO: 10). |
| Unde e encodarea în unghiuri? | [qai/protocol.py:78](../qai/protocol.py#L78) `AngleEncoder` | Celulele 41–42: StandardScaler pe Time și Amount, PCA la 4 componente, MinMax în [0, π]. Se antrenează doar pe setul de antrenare subeșantionat ([linia 92](../qai/protocol.py#L92)). |
| Și encodarea robustă? | [qai/protocol.py:101](../qai/protocol.py#L101) `RobustAngleEncoder` | Varianta noastră: MinMax pe percentilele [q, 100 − q]. Folosită doar ca termen de comparație, nealeasă pe validare. |
| Unde e modelul de zgomot? | [qai/protocol.py:143](../qai/protocol.py#L143) `noisy_qnode` | Celula 106: `qml.transforms.decompose` în porțile native, apoi `qml.noise.insert` pune `DepolarizingChannel(p)` după fiecare poartă, pe fiecare qubit atins, simulat exact pe `default.mixed`. |
| Unde se alege pragul? | [qai/protocol.py:121](../qai/protocol.py#L121) `choose_threshold` | Celula 92: pragul care maximizează F1 pe scorurile de validare, la același p. |
| Unde se calculează metricile? | [qai/protocol.py:128](../qai/protocol.py#L128) `evaluate` | AUPRC, AUROC, precision, recall, F1, MCC, TP/FP/FN/TN. Spearman ρ (zgomot vs fără zgomot) e în [qai/protocol.py:194](../qai/protocol.py#L194) `protocol_table`. |
| Unde e `evaluate_under_noise`? | [qai/protocol.py:206](../qai/protocol.py#L206) | Celula 125, împărțită în doi pași: `scores_under_noise` ([:189](../qai/protocol.py#L189)) și `protocol_table` ([:194](../qai/protocol.py#L194)). |
| Unde numărați porțile și locurile de zgomot? | [qai/protocol.py:170](../qai/protocol.py#L170) `gate_stats` | `qml.specs` pe circuitul descompus. Rezultatele sunt în `results/task2/circuits.csv`, scris de [qai/experiments/facts.py:24](../qai/experiments/facts.py#L24). |
| Unde e formula de cost (ledger)? | [qai/protocol.py:234](../qai/protocol.py#L234) `gradient_ledger` | Celula 86: (1 + 2 × parametri) execuții per exemplu și epocă, plus validarea după fiecare epocă. |
| Unde e verificarea cu `qml.Tracker`? | [qai/protocol.py:241](../qai/protocol.py#L241) și [:254](../qai/protocol.py#L254) | Celula 89 și varianta pentru citiri cu mai mulți qubiți. Pentru MOCO: [qai/moco.py:184](../qai/moco.py#L184) `tracker_check`. |

## 2. Baseline-ul kitului (R0) și R1

| Întrebarea | Unde | Ce face |
|---|---|---|
| Unde e circuitul baseline? | [qai/kit_vqc.py:27](../qai/kit_vqc.py#L27) `feature_map` (ZZ, 2 repetiții), [:39](../qai/kit_vqc.py#L39) `ansatz` (EfficientSU2, 32 de parametri), [:49](../qai/kit_vqc.py#L49) `vqc` (citire ⟨Z₀⟩) | Copiat din celulele 64, 69, 74. |
| Unde se antrenează baseline-ul? | [qai/kit_vqc.py:93](../qai/kit_vqc.py#L93) `train_vqc` | Celula 81: Adam, BCE, 30 de epoci, păstrează epoca cu cea mai bună AUPRC pe validare. |
| Unde rulați baseline-ul pe 5 seed-uri? | [qai/experiments/exp_task2_baseline.py:28](../qai/experiments/exp_task2_baseline.py#L28) `run_seed` | Baseline + regresia logistică 1:1, prin protocol. Rezultat: `results/task2/baseline_per_seed.csv`. |
| Unde e varianta fără porțile moarte (R1)? | [qai/kit_vqc.py:57](../qai/kit_vqc.py#L57) (conul de lumină) și [:61](../qai/kit_vqc.py#L61) `vqc_lightcone` | Scoate 12 rotații și 3 CNOT care nu ajung la qubitul măsurat. Testul [tests/test_protocol.py:35](../tests/test_protocol.py#L35) arată că scorurile rămân identice, chiar și cu zgomot. |

## 3. MOCO, metoda noastră (`qai/moco.py`)

| Întrebarea | Unde | Ce face |
|---|---|---|
| Unde e circuitul MOCO? | [qai/moco.py:33](../qai/moco.py#L33) `feature_map` (varianta `single`: RY(x_q) pe fiecare qubit) și [:156](../qai/moco.py#L156) `make_circuit` (adaugă Rot și citirea H = Σ w_q Z_q) | 8 porți, 0 CNOT, adâncime 2. Variantele `pair`, `linear_cnot`, `kit_zz` sunt ablațiile. |
| Unde se „măsoară o dată”? | [qai/moco.py:64](../qai/moco.py#L64) `bloch_exact` și [:78](../qai/moco.py#L78) `bloch_from_shots` | Vectorii Bloch ai fiecărei tranzacții (bazele X și Z, 2 execuții per rând), cu zgomot de shots realist. Câte baze trebuie: [:60](../qai/moco.py#L60). |
| Unde e antrenarea convexă? | [qai/moco.py:88](../qai/moco.py#L88) `train_logistic_torch` | Regresie logistică în PyTorch, cu LBFGS ([linia 94](../qai/moco.py#L94)) și BCE ([linia 100](../qai/moco.py#L100)). Problema e strict convexă, deci optimul e global. |
| Unde alegeți C? | [qai/moco.py:117](../qai/moco.py#L117) `fit_convex` | Cross-validation în 5 folduri, doar pe setul de antrenare: 0 execuții de validare. |
| Unde „compilați” rezultatul în circuit? | [qai/moco.py:134](../qai/moco.py#L134) `compile_to_circuit` | Coeficienții devin unghiuri Rot (arccos și arctan2, [liniile 145–146](../qai/moco.py#L145)) și ponderi w_q. |
| Unde e scorul în formă închisă? | [qai/moco.py:150](../qai/moco.py#L150) `score_exact` | p_fraud = (1 − ⟨H⟩)/2 calculat direct. Testul [tests/test_protocol.py:46](../tests/test_protocol.py#L46) arată că e identic cu circuitul. |
| Unde e mutarea analitică a pragului? | [qai/moco.py:166](../qai/moco.py#L166) `transfer_threshold` | t_p = ½ − λᵏ(½ − t₀) ([linia 168](../qai/moco.py#L168)). |
| Unde e tot pipeline-ul MOCO într-un apel? | [qai/moco.py:171](../qai/moco.py#L171) `fit` | Măsurare (cu shots), fit convex, compilare. |
| Unde se rulează MOCO prin protocol? | [qai/experiments/exp_moco.py:28](../qai/experiments/exp_moco.py#L28) `run_seed` | MOCO + regresia logistică pe aceleași intrări (1:10). Rezultate: `results/task2/moco_per_seed_single_10.csv`; ablațiile cu `--fm pair/linear_cnot/kit_zz`, encodarea robustă cu `--q 0.5`. |
| Unde ați ales 1:10 și `single`? | [qai/experiments/exp_selection.py:31](../qai/experiments/exp_selection.py#L31) | Grila fm × ratio × 5 seed-uri, doar pe validare. Regula: cea mai ieftină configurație aflată la cel mult o eroare standard de cea mai bună medie. Rezultat: `results/selection_on_validation.csv`. |
| Unde e versiunea inițială, de sine stătătoare? | [moco_task2.py](../moco_task2.py) | Prima implementare (păstrată ca referință); codul folosit în rezultate e cel din `qai/moco.py`. |

## 4. Teorema λ² (de ce zgomotul nu poate schimba ordonarea)

| Întrebarea | Unde |
|---|---|
| Unde e demonstrată numeric? | [tests/test_protocol.py:52](../tests/test_protocol.py#L52) `test_moco_noise_is_a_uniform_contraction`: cu modelul de zgomot al kitului, scorul zgomotos e exact ½ − λ²(½ − s₀), la p = 0.001, 0.05 și 0.3. |
| Și comprimarea măsurată? | [tests/test_protocol.py:76](../tests/test_protocol.py#L76) (comprimarea = λ², eroare < 1e-6) și `results/task2/noise_shots_checks.csv` (eroare 3e-17 pe date reale). |
| Unde e explicată? | `paper/moco.tex`, secțiunea Theory ([linia 172](../paper/moco.tex#L172)); exercițiile 3.2 și 10.1 din Task 1 sunt în `task1/README.md`. |

## 5. Experimentele

| Întrebarea | Unde | Rezultatul |
|---|---|---|
| R2: VQC cu entanglement și warm start din MOCO? | [qai/experiments/exp_entangled.py:52](../qai/experiments/exp_entangled.py#L52) `gates` (RY, Rot, 3 CNOT, Rot), [:86](../qai/experiments/exp_entangled.py#L86) `moco_init` (warm start), [:94](../qai/experiments/exp_entangled.py#L94) `random_init`, [:101](../qai/experiments/exp_entangled.py#L101) `run_seed` (inclusiv antrenarea sub zgomot) | `results/task2/entangled_per_seed.csv` |
| Shots finite și zgomot până la p = 0.5? | [qai/experiments/exp_noise_shots.py:212](../qai/experiments/exp_noise_shots.py#L212) `run_job`; eșantionarea exactă: binomială pe qubitul 0 pentru baseline ([:76](../qai/experiments/exp_noise_shots.py#L76)), binomiale independente pentru MOCO ([:89](../qai/experiments/exp_noise_shots.py#L89)), multinomială pe 16 rezultate pentru R2 ([:103](../qai/experiments/exp_noise_shots.py#L103)) | `results/task2/noise_shots.csv`, verificări în `noise_shots_checks.csv` |
| Comprimarea efectivă c(p)? | [qai/experiments/exp_noise_shots.py:68](../qai/experiments/exp_noise_shots.py#L68) `effective_compression` | coloana `compression` din `noise_shots.csv` |
| Taxa în shots 1/c²? | [qai/experiments/shot_tax.py:28](../qai/experiments/shot_tax.py#L28) `shot_tax_table` (cu verificarea MOCO = λ⁻⁴) | `results/task2/shot_tax.csv`, figura `figures/e_shot_tax.png` |
| CV pe toate cele 492 de fraude? | [qai/experiments/exp_allfrauds_cv.py:37](../qai/experiments/exp_allfrauds_cv.py#L37) `fold_split`, [:45](../qai/experiments/exp_allfrauds_cv.py#L45) `run_fold` | `results/task2/cv/cv_per_fold.csv` |
| Triajul adaptiv (shots doar unde trebuie)? | Intervalele valide oricând: [qai/adaptive.py:55](../qai/adaptive.py#L55) `AnytimeCI` (Hoeffding, Maurer-Pontil, plug-in Bernstein); triajul pe loturi: [qai/adaptive.py:122](../qai/adaptive.py#L122) `triage`; selecția pe validare: [qai/experiments/exp_adaptive_shots.py:249](../qai/experiments/exp_adaptive_shots.py#L249) `select`; testul: [:297](../qai/experiments/exp_adaptive_shots.py#L297) `test_seed` | `results/task2/adaptive_shots.csv`, `adaptive_selection_on_validation.csv`, `adaptive_checks.csv` |
| Noise meter-ul (estimarea lui p)? | [qai/adaptive.py:175](../qai/adaptive.py#L175) (mediana raportului), [:182](../qai/adaptive.py#L182) (cele mai mici pătrate), [:188](../qai/adaptive.py#L188) `p_from_lambda2`; rularea: [qai/experiments/exp_adaptive_shots.py:356](../qai/experiments/exp_adaptive_shots.py#L356) `noise_meter_seed` | `results/task2/noise_meter.csv` |
| Sweep-ul de încredere (α)? | [qai/experiments/exp_confidence_sweep.py:33](../qai/experiments/exp_confidence_sweep.py#L33) `sweep_seed` | `results/task2/confidence_sweep.csv`, figura `figures/l_confidence_sweep.png` |
| Marjele scorurilor? | [qai/experiments/exp_margins.py:21](../qai/experiments/exp_margins.py#L21) | `results/task2/margins.csv` |
| Corelațiile, PCA și exemplul cap-coadă? | [qai/experiments/exp_covariance_e2e.py:63](../qai/experiments/exp_covariance_e2e.py#L63) | `results/task2/corr_stats.csv`, `e2e_example.md`, figura `figures/m_corr_heatmap.png` |
| Precizia la un recall fix? | [qai/experiments/exp_precision_at_recall.py:24](../qai/experiments/exp_precision_at_recall.py#L24) | `results/task2/precision_at_recall.csv`, figura `figures/n_pr_curve.png` |

## 6. Costul (ledger-ul)

| Întrebarea | Unde | Rezultatul |
|---|---|---|
| Ledger-ul pe fiecare metodă? | scris de fiecare experiment (`exp_task2_baseline`, `exp_moco`, `exp_entangled`), cu formula și Tracker-ul | `results/task2/ledger_*.csv` |
| Ledger-ul final (antrenare + selecție + calibrarea pragului)? | [qai/experiments/ledger_final.py:28](../qai/experiments/ledger_final.py#L28) `build` | `results/task2/ledger_final.csv` |
| Costul inferenței cu triaj? | [qai/experiments/exp_adaptive_shots.py:509](../qai/experiments/exp_adaptive_shots.py#L509) `ledger_table`, cu verificare Tracker la [:406](../qai/experiments/exp_adaptive_shots.py#L406) | `results/task2/ledger_adaptive.csv` |

## 7. Teste, tabele, figuri, rulare

| Întrebarea | Unde |
|---|---|
| Ce teste aveți? | [tests/test_protocol.py](../tests/test_protocol.py): cifrele kitului (93 porți, 126 locuri de zgomot, ledger, Tracker 41), R1 identic, teorema λ², comprimarea, citirea R2, split-ul. [tests/test_adaptive.py](../tests/test_adaptive.py): intervalele ratează media în cel mult α din cazuri, deciziile timpurii greșite ≤ α, noise meter-ul exact. Rulare: `make task2-test`. |
| Unde sunt toate tabelele într-un loc? | `results/task2/TABLES.md`, generat de [qai/experiments/task2_report.py:60](../qai/experiments/task2_report.py#L60). |
| Unde se fac figurile? | [qai/experiments/task2_figures.py](../qai/experiments/task2_figures.py) (figurile a–k, `fig_a` … `fig_k`), restul în scripturile experimentelor; toate în `results/task2/figures/`. |
| Cum rulez tot? | `make task2-test` (teste), `make task2-light` (tabele și figuri din CSV-uri, câteva minute), `make task2` (tot de la zero, ore). Ordinea pașilor e în [Makefile](../Makefile), secțiunea Task 2. |
| Unde e Task 1? | `task1/task1_guided_vqc_fraud_detection.ipynb` (completat și rulat cap-coadă) și `task1/README.md`; materialul original al organizatorilor e în `kit/`. |
| Unde e documentul complet al proiectului? | `docs/PROJECT.md` (metoda, toate rezultatele cu sursa lor, ce n-a mers, răspunsuri pentru juriu); lucrarea: `paper/moco.pdf`; prezentarea: `presentation/MOCO_prezentare.pptx`. |

## 8. Cifrele-cheie și fișierul din care vin

| Cifra | Fișierul |
|---|---|
| Baseline: AUPRC 0.580 → 0.405 la p = 0.05, ρ 0.686 | `results/task2/baseline_per_seed.csv` |
| MOCO: AUPRC 0.716 ± 0.007 la orice p, ρ = 1 | `results/task2/moco_per_seed_single_10.csv` |
| Regresia logistică pe aceleași intrări: 0.710 ± 0.007 | `results/task2/moco_per_seed_single_10.csv` |
| 6.490 execuții față de 2.859.360 (−99,77%) | `results/task2/ledger_moco_single_10.csv`, `ledger_baseline.csv` |
| 7.490 cu noise meter (−99,74%), 63.452 conservator (−97,8%) | `results/task2/ledger_final.csv`, `noise_meter.csv` |
| 93 porți, 33 CNOT, adâncime 45, 126 locuri de zgomot (baseline); 8, 0, 2, 8 (MOCO) | `results/task2/circuits.csv` |
| Taxa în shots la p = 0.05: MOCO 1,32×, baseline ≥ 504× | `results/task2/shot_tax.csv` |
| Cu 1.024 de shots la p = 0.05: MOCO 0.705, baseline 0.007 | `results/task2/noise_shots.csv` |
| Triaj: 208 ± 47 shots, F1 0.801 față de 0.797 | `results/task2/adaptive_shots.csv` |
| CV pe 492 de fraude: MOCO 0.721 ± 0.020, LogReg 0.739 ± 0.030, baseline 0.499 ± 0.137 | `results/task2/cv/cv_per_fold.csv` |
