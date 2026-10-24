# Note pentru prezentare (MOCO, Task 2)

Prezentarea: `presentation/moco_slides.pdf`, generată de `presentation/build_slides.py` direct din CSV-uri (`python3 presentation/build_slides.py`). Toate numerele de pe slide-uri sunt citite din fișierele de mai jos în momentul generării.

Toate valorile sunt acum completate din CSV-uri, inclusiv triajul adaptiv, noise meter-ul și encodarea robustă (commit-ul 66a3e91). În dicționarul `PLACEHOLDERS` din `build_slides.py` a rămas doar numele colegei, Deliu Georgiana. Marja de 0,08 și deviația de 0,41 per shot din `PROJECT.md` nu sunt în niciun CSV, deci nu apar pe slide-uri.

Vorbitori: Darius (D) și Deliu Georgiana (C), care a făcut Task 1. Slide-urile 2–4 sunt ale lui C, iar restul alternează. Fiecare slide durează 30–45 de secunde, deci în total cam 9–10 minute.

Față de structura cerută, am separat trei slide-uri, ca figurile să rămână mari și fiecare slide să aibă un singur mesaj: pipeline-ul (4) de circuit (5), CV-ul (10) de paradox (11). De aceea sunt 15 slide-uri, nu 13. După slide-ul 15 urmează trei slide-uri de rezervă (B1–B3), doar pentru întrebări; nu le prezentăm.

---

## 1. Titlu (D)

„Suntem echipa MOCO: Measure Once, Compile Once. Am construit un clasificator variațional pentru frauda cu carduri pe care zgomotul nu-l poate strica, la un cost foarte mic. Trei idei: măsurăm fiecare tranzacție o singură dată, cu două execuții de circuit; antrenăm convex, deci găsim optimul global fără niciun circuit în buclă; iar zgomotul depolarizant nu poate schimba ordinea tranzacțiilor, la orice nivel.”

Surse: 2 execuții per rând, `results/task2/ledger_moco_single_10.csv` (`executions_per_example`).

## 2. Problema și baseline-ul (C)

„Avem 284.807 tranzacții, dintre care doar 492 sunt fraude, adică 0,17%. Un model care nu marchează nimic are 99,83% acuratețe, deci măsurăm AUPRC. Folosim split-ul comun 60/20/20 cu seed 42; testul are 56.962 de rânduri și 98 de fraude. Am auditat VQC-ul din kit: 93 de porți, 33 de CNOT, adâncime 45, 126 de locuri unde intră zgomotul și 2,86 milioane de execuții ca să-l antrenezi. Sub protocolul oficial, AUPRC-ul lui scade de la 0,580 la 0,405 la p = 0,05.”

Surse:
- dataset și split: `results/task2/dataset_split.csv`;
- porți, CNOT, adâncime, locuri de zgomot: `results/task2/circuits.csv` (rândul baseline VQC);
- execuțiile: `results/task2/ledger_baseline.csv` (`total executions`);
- AUPRC: `results/task2/baseline_per_seed.csv` (5 seed-uri, p = 0 și p = 0,05).

## 3. Din Task 1 la idee (C)

„Ideea vine din două exerciții din Task 1. În exercițiul 3.2 am văzut că dacă strângem scorurile spre 1/2, AUPRC-ul nu se schimbă, pentru că depinde doar de ordine; se mută doar pragul. În exercițiul 10.1 am văzut că un canal depolarizant după o poartă pe un qubit înmulțește ⟨Z⟩ cu 1 − 4p/3, același factor pentru orice intrare. Dacă fiecare qubit are același număr de porți și nu există CNOT, tot scorul se strânge cu un singur factor λ². Pe asta e construit MOCO.”

Surse: `task1/task1_guided_vqc_fraud_detection.ipynb` (exercițiile 3.2 și 10.1); `task1/README.md`. Slide-ul nu are cifre din date.

## 4. Pipeline-ul (C)

„Orice alegere o facem pe train sau pe validare, niciodată pe test. Encoder-ul (scalare Time și Amount, PCA la 4 componente, unghiuri în [0, π]) e antrenat doar pe subsetul de antrenare. Raportul 1:10 și feature map-ul le-am ales pe validare, pe 5 seed-uri: single 1:10 are cea mai bună medie, 0,675; pair 1:30 egalează, dar costă de 4 ori mai multe execuții. Constanta C a fitului convex o alegem prin CV în interiorul setului de antrenare. Testul îl atingem o singură dată, la final.”

Surse: tabelul și cifrele, `results/selection_on_validation.csv`.

## 5. Circuitul (D)

„Asta e tot circuitul MOCO: un RY care încarcă trăsătura și un Rot antrenat, pe fiecare qubit. Opt porți, niciun CNOT, adâncime 2. În protocol, după fiecare poartă vine un canal de zgomot pe fiecare qubit atins: MOCO are 8 astfel de locuri, baseline-ul 126.”

Surse: `results/task2/circuits.csv` (rândurile MOCO single și baseline VQC); figura `results/task2/figures/f_circuits.png`.

## 6. Metoda (C)

„Antrenarea are patru pași. Unu: starea RY(x)|0⟩ are vectorul Bloch (sin x, 0, cos x), așa că măsurăm fiecare tranzacție de antrenare în bazele X și Z: 2 execuții pe rând, 6.490 în total. Doi: în imaginea Heisenberg, după un Rot local, ⟨Z_q⟩ e exact un produs scalar cu vectorul Bloch, deci citirea e liniară în ce am măsurat. Trei: asta e o regresie logistică convexă, rezolvată în PyTorch, cu optim global. Patru: compilăm coeficienții înapoi în unghiuri Rot și ponderi, iar clasificarea o face circuitul. Ideea seamănă cu surogatele clasice ale modelelor cuantice (Schreiber și colegii, 2023) și cu achiziția de date urmată de procesare clasică (Cerezo și colegii); diferența noastră e că rezultatul se compilează înapoi în circuit.”

Surse: 2 execuții per rând și 6.490 de execuții, `results/task2/ledger_moco_single_10.csv`.

## 7. Teorema (D)

„De ce nu poate zgomotul să strice ordonarea? Fiecare canal micșorează vectorul Bloch cu λ = 1 − 4p/3. Fără CNOT și cu două porți pe fiecare qubit, starea rămâne un produs și fiecare ⟨Z_q⟩ se înmulțește cu λ². Citirea e liniară, deci scorul zgomotos e 1/2 − λ²(1/2 − scorul ideal), o funcție strict crescătoare. AUPRC, AUROC și ρ rămân identice, iar pragul se mută analitic. Am verificat cu modelul de zgomot al kitului: eroare maximă 5,7·10⁻¹⁴ pe 200 de rânduri, iar comprimarea măsurată e egală cu λ² până la 3·10⁻¹⁷.”

Surse: `results/task2/noise_shots_checks.csv` (rândurile MOCO: „binomial means vs kit noisy_predict” și „compression == lambda^2”). Diferența de 1,7·10⁻¹⁶ din verificarea mai veche nu e într-un CSV, deci nu apare pe slide.

## 8. Rezultatele oficiale (C)

„Sub protocolul oficial, pe 5 seed-uri: MOCO are 0,716 la orice nivel de zgomot și ρ = 1. Regresia logistică pe exact aceleași intrări are 0,710, deci MOCO o egalează. Baseline-ul cade de la 0,580 la 0,405, cu ρ 0,686. R2, varianta cu 3 CNOT pornită din MOCO, ajunge la 0,667 și e aproape la fel de robustă, dar entanglement-ul o costă AUPRC.”

Surse:
- baseline și LogReg 1:1: `results/task2/baseline_per_seed.csv`;
- MOCO și LogReg 1:10: `results/task2/moco_per_seed_single_10.csv`;
- R2 cu inițializare aleatoare și cu warm start: `results/task2/entangled_per_seed.csv`;
- figurile: `results/task2/figures/compact/a_auprc_protocol.png` și `b_spearman_protocol.png`.

## 9. Răspunsul la hintul mentorului (D)

„Mentorul ne-a întrebat ce se întâmplă când crește zgomotul. Protocolul folosește valori exacte, ca și cum am avea shots infinite. Cu shots reale, zgomotul devine o taxă: semnalul e comprimat, deci trebuie mai multe shots pentru aceeași calitate. Cu 1.024 de shots la p = 0,05, MOCO păstrează 0,705, R2 0,643, iar baseline-ul ajunge la 0,007, adică nivelul șansei. Taxa în shots e 1,32x pentru MOCO, exact λ⁻⁴, iar pentru baseline cel puțin 504x. E doar o limită inferioară, pentru că baseline-ul pierde ordonarea și cu shots infinite.”

Surse:
- AUPRC la S = 1.024: `results/task2/noise_shots.csv` (shots = 1024, p = 0,05, media pe seed-uri);
- taxa în shots: `results/task2/shot_tax.csv` (p = 0,05);
- figurile: `results/task2/figures/compact/d_auprc_1024_shots.png` și `e_shot_tax.png`.

Totul e diagnostic, măsurat pe test, nefolosit pentru nicio alegere.

## 10. Toate cele 492 de fraude (C)

„Testul oficial are doar 98 de fraude, așa că am făcut și o validare încrucișată pe 5 folduri, în care fiecare dintre cele 492 de fraude ajunge o dată în test. MOCO are 0,721 ± 0,020, regresia logistică 0,739 ± 0,030: diferența pereche e −0,018 ± 0,026, deci nu se pot distinge. Baseline-ul are 0,499 ± 0,137, de 7 ori mai instabil între folduri decât MOCO.”

Surse: `results/task2/cv/cv_per_fold.csv`; figura `results/task2/figures/i_cv_all_frauds.png`.

## 11. Paradoxul (D)

„Pe aceleași folduri apare paradoxul din articolul de la care am pornit. La zgomot mic, AUPRC-ul baseline-ului crește în toate cele 5 folduri, deși ordonarea lui se strică: ρ scade la 0,67 la p = 0,05. O singură metrică poate arăta o îmbunătățire acolo unde modelul se degradează. La MOCO asta nu se poate întâmpla, pentru că ordonarea lui e invariantă.”

Surse: `results/task2/cv/cv_per_fold.csv` (modelul baseline VQC, p de la 0 la 0,1); figura `results/task2/figures/k_paradox.png`.

## 12. Ledger-ul de cost (C)

„Costul pe hardware, în execuții de circuit. După definiția kitului, antrenare plus selecția modelului: baseline 2.859.360, MOCO 6.490, adică −99,77%. Dacă adăugăm și calibrarea pragului cu noise meter-ul, MOCO costă 6.490 plus 1.000, adică 7.490 de execuții, −99,74%. În varianta cea mai conservatoare, cu pragul calibrat pe toată validarea zgomotoasă, MOCO are 63.452 de execuții, adică −97,8%, de 46 de ori mai puțin decât baseline-ul. R2 cu warm start costă 469.717.”

Surse:
- `results/task2/ledger_final.csv` (rândurile baseline kit formula, MOCO single 1:10 și R2 warm start);
- `results/task2/ledger_baseline.csv` și `ledger_moco_single_10.csv`;
- N_cal = 1.000: `results/task2/noise_meter.csv` (MOCO kit, least squares, N_cal = 1000, S = 1024, p = 0,05, coloana `calibration executions`);
- figura `results/task2/figures/c_ledger.png`.

## 13. Triajul adaptiv și noise meter-ul (D)

„Taxa în shots o putem plăti inteligent: o tranzacție evident legitimă nu are nevoie de 1.024 de shots. Măsurăm în loturi și ne oprim când un interval valid oricând exclude pragul. Cu intervale de tip Hoeffding (Maurer-Pontil), encodarea kitului cere 992 de shots pentru același F1, aproape cât fix. Pe validare a câștigat intervalul plug-in Bernstein: pe test, la p = 0,05, 208 ± 47 de shots per tranzacție, F1 0,801 ± 0,018, față de 0,797 cu 1.024 de shots fixe, și zero decizii greșite oprite devreme. Cu același buget fix, 209 shots pentru toți, F1 scade la 0,702. Noise meter-ul estimează p din 1.000 de execuții de calibrare în loc de 56.962, cu eroare relativă de 2,4%, iar F1 practic nu se schimbă. Encodarea robustă e doar un termen de comparație: 0,738 AUPRC, dar validarea a preferat encodarea kitului. Atenție: triajul taie shots, nu execuții. Fiecare lot e o execuție, deci la p = 0,05 avem cam 13 execuții per tranzacție în loc de una. Cifrele principale de execuții din ledger rămân neschimbate.”

Surse:
- 13 execuții per tranzacție: `results/task2/ledger_adaptive.csv` (rândul „MOCO kit: inference, adaptive”, p = 0,05, coloana `inference executions per transaction` = 13,031);
- `results/task2/adaptive_shots.csv`: rândurile MOCO kit, p = 0,05: `adaptive` + `plugin-bernstein` + cap 4096 (coloanele `mean shots`, `F1`, `wrong early decisions`, `b`, `alpha`); `fixed 1024` (`F1`); `fixed same budget` cap 4096 (`S`, `F1`); baseline VQC `adaptive` cap 4096 (`capped`, `F1`);
- 992 de shots: `results/task2/adaptive_selection_on_validation.csv` (MOCO kit, ci = bernstein, F1 în 0,005 de cel selectat, minimul lui `mean shots`);
- noise meter-ul: `results/task2/noise_meter.csv` (MOCO kit, least squares, N_cal = 1000, S = 1024, p = 0,05: `calibration executions`, `relative RMSE`, `F1 drop`);
- encodarea robustă: `results/task2/moco_per_seed_single_10_q0.5.csv` (p = 0,05, `AUPRC`).

Figura din dreapta (figura l, sweep-ul de încredere, făcut doar pe validare): „Dacă scădem încrederea de la 0,999 la 0,51, costul scade de 4,3 ori. F1 rămâne în 0,01 de valoarea de la α = 0,05 până la încrederea 0,7, iar deciziile greșite oprite devreme sunt cel mult 0,0002 pe orice seed. α = 0,05 l-am fixat înainte de sweep și nu l-am schimbat.”

Surse pentru linia de sub figură, toate din `results/task2/confidence_sweep.csv` (MOCO kit, plug-in Bernstein, b = 16, S_max = 4096, 5 seed-uri):
- 4,3x: `mean shots` la p = 0,05, încrederea 0,999 (405) împărțit la încrederea 0,51 (95), medii pe seed-uri;
- 0,7: cea mai mică încredere pentru care F1 mediu e în 0,01 de cel de la 0,95, la ambele p (0,01 și 0,05);
- ≤ 0,0002: maximul coloanei `wrong early decisions` pe toate rândurile (cel mai rău seed are 0,00018, rotunjit în sus; media pe seed-uri e ≤ 0,00012);
- figura: `results/task2/figures/l_confidence_sweep.png`.

## 14. Ce n-a mers (C)

„Rezultatele negative, măsurate cinstit:
- Am scos din baseline cele 15 porți care nu ajung la qubitul măsurat, iar scorurile au rămas identice: câștig zero.
- Trei CNOT-uri scad deja AUPRC-ul și strică invarianța.
- Encodarea ZZ a kitului rămâne la 0,44 chiar cu citirea optimă.
- Antrenarea R2 sub zgomot n-a adus nimic.
- Tabelul cu shots necesare a fost prea zgomotos: validarea și testul coincid doar în 24 din 40 de celule.
- Căderea baseline-ului sub zgomot exact depinde de fold: apare în 3 din 5.”

Surse:
- 15 porți: `results/task2/circuits.csv` (93 − 78);
- ablațiile: `results/task2/moco_per_seed_{single,pair,linear_cnot,kit_zz}_10.csv`; figura `results/task2/figures/compact/g_ablations.png`;
- R2: `results/task2/entangled_per_seed.csv`;
- 24 din 40: `results/task2/shots_needed.csv` (coloana `agree`);
- 3 din 5: `results/task2/cv/cv_per_fold.csv`.

## 15. Ce pretindem, ce nu, și ce urmează (D)

„Ce pretindem: robustețe (ρ = 1 la orice nivel de zgomot, taxa în shots λ⁻⁴), stabilitate (±0,020 între folduri) și cost (6.490 de execuții de antrenare, −99,77%). Ce nu pretindem: avantaj cuantic, pentru că circuitul e o stare produs, simulabilă clasic; și nici că batem regresia logistică, pentru că o egalăm. Pașii următori: zgomot de hardware real (amortizare de amplitudine, erori de citire) și mai mulți qubiți, care la MOCO nu costă nimic în robustețe. Vă mulțumim.”

Surse:
- ±0,020 și −0,018 ± 0,026: `results/task2/cv/cv_per_fold.csv`;
- 6.490 și −99,77%: `results/task2/ledger_moco_single_10.csv` și `ledger_baseline.csv`.

---

## Slide-uri de rezervă (doar pentru întrebări)

### B1. Corelațiile trăsăturilor (D)

Întrebarea la care răspunde: „PCA-ul din encoder nu e inutil, dacă V1–V28 sunt deja componente PCA?”

„Pe tot setul de antrenare, V1–V28 sunt practic necorelate: |r| mediu 0,004, maxim 0,039. Pe setul nostru 1:10, unde fraudele sunt 9%, sunt puternic corelate: |r| mediu 0,207, maxim 0,84 între V16 și V17, și 75 de perechi peste 0,5. Encoder-ul e antrenat pe setul 1:10, deci PCA-ul lui găsește structură reală, dată de fraude.”

Surse: `results/task2/corr_stats.csv`; figura `results/task2/figures/m_corr_heatmap.png`.

### B2. Precizia la un recall dat (D)

Întrebarea la care răspunde: „Câte alarme false dați ca să prindeți fraudele?”

„La 80% recall, MOCO păstrează 0,817 ± 0,009 precizie, iar între 50% și 70% recall precizia stă între 0,829 și 0,833. La 90% recall se prăbușește, la 0,072: ultimele fraude arată ca tranzacții legitime în aceste 4 componente. În punctul de lucru, cu pragul ales pe validare, avem precizie 0,781 ± 0,048 și recall 0,822 ± 0,014: în medie prindem 81 din cele 98 de fraude de test, cu 23 de alarme false.”

Surse: `results/task2/precision_at_recall.csv` (5 seed-uri: `max precision at recall` și `operating point`); figura `results/task2/figures/n_pr_curve.png`. Totul e pe test, doar pentru raportare.

### B3. Un exemplu cap-coadă (C)

Întrebarea la care răspunde: „Ce face concret circuitul cu o tranzacție?”

„Am luat din validare tranzacția cu scorul median din fiecare clasă (seed 0). Frauda: patru unghiuri, ⟨Z_q⟩ după Rot, apoi ⟨H⟩ = −0,018 și scorul 0,509. E peste pragul 0,451, deci FRAUDĂ. La p = 0,05, scorul devine 0,508, iar pragul transferat analitic devine 0,457, deci verdictul nu se schimbă. Tranzacția legitimă are scorul 0,371, sub prag, iar la p = 0,05 scorul e 0,387 față de 0,457, deci tot legitimă. Scorurile recalculate aici coincid cu cele din protocol până la 2,8·10⁻¹⁶.”

Sursa: `results/task2/e2e_example.md`, generat de `qai/experiments/exp_covariance_e2e.py`. Exemplul nu are CSV propriu. Numerele de pe slide sunt citite din acest fișier generat în momentul în care se construiește prezentarea (`load_e2e` din `build_slides.py`), deci niciunul nu e scris de mână.
