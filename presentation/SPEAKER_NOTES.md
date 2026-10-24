# Note pentru prezentare (MOCO, Task 2)

Prezentarea: `presentation/moco_slides.pdf`, generată de `presentation/build_slides.py` direct din CSV-uri (`python3 presentation/build_slides.py`). Toate numerele de pe slide-uri sunt citite din fișierele de mai jos în momentul generării.

Valorile care încă se calculează în cealaltă sesiune (N_cal, MOCO-robust, triajul adaptiv, numele colegului) sunt toate în dicționarul `PLACEHOLDERS` din `build_slides.py`. Pe slide apar cu roșu, ca „[TBD: nume]”. Se completează acolo și se regenerează PDF-ul.

Vorbitori: Darius (D) și colegul (C), care a făcut Task 1. Slide-urile 2–4 sunt ale lui C, iar restul alternează. Fiecare slide durează 30–45 de secunde, deci în total cam 9–10 minute.

Față de structura cerută, am separat trei slide-uri, ca figurile să rămână mari și fiecare slide să aibă un singur mesaj: pipeline-ul (4) de circuit (5), CV-ul (10) de paradox (11). De aceea sunt 15 slide-uri, nu 13.

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

„Costul pe hardware, în execuții de circuit. După definiția kitului, antrenare plus selecția modelului: baseline 2.859.360, MOCO 6.490, adică −99,77%. Dacă adăugăm și calibrarea pragului cu noise meter-ul, MOCO costă 6.490 plus N_cal, iar N_cal vine din rularea celeilalte sesiuni. În varianta cea mai conservatoare, cu pragul calibrat pe toată validarea zgomotoasă, MOCO are 63.452 de execuții, adică −97,8%, de 46 de ori mai puțin decât baseline-ul. R2 cu warm start costă 469.717.”

Surse:
- `results/task2/ledger_final.csv` (rândurile baseline kit formula, MOCO single 1:10 și R2 warm start);
- `results/task2/ledger_baseline.csv` și `ledger_moco_single_10.csv`;
- figura `results/task2/figures/c_ledger.png`;
- N_cal: `PLACEHOLDERS`, de completat din `results/task2/noise_meter.csv`.

## 13. Encodarea pentru shots și triajul adaptiv (D)

„Ultimul pas e să plătim taxa în shots inteligent: o tranzacție evident legitimă nu are nevoie de 1.024 de shots. Pe encodarea kitului, triajul nu merge: marginea dintre scorurile fraudelor și ale tranzacțiilor legitime e [kit_margin], mult sub deviația standard a unui singur shot, [per_shot_std]. Cu o encodare robustă, pe percentile din setul de antrenare, clasele rămân separate pe sfera Bloch. Noise meter-ul estimează p din N_cal execuții de calibrare, mută pragul analitic, iar triajul oprește eșantionarea devreme.”

Surse: toate valorile sunt în `PLACEHOLDERS`, din `results/task2/adaptive_shots.csv` și `results/task2/noise_meter.csv`, după rularea completă a celeilalte sesiuni. Nu se spun cifre până nu sunt completate.

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
