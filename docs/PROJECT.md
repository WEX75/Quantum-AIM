# BreaQ 2026 · Task 2 · Proiectul nostru: MOCO

> **Document viu.** Sursa unică pentru echipă, pentru agenții AI care lucrează cu noi și pentru slide-uri.
> Se actualizează după fiecare rezultat nou (vezi *Changelog* la final). Cifrele marcate „rulează” sunt încă în calcul.
> Cifrele marcate [claude.ai] vin de la agentul din claude.ai și nu au fost încă reproduse în acest repo.
>
> **Ultima actualizare:** sâmbătă 24.10, 23:50
> **Deadline:** upload prezentare **duminică 25.10, 09:00**

---

## 1. Problema, pe scurt

- **Date:** ULB Credit Card Fraud, 284.807 tranzacții, dintre care 492 fraude (0,17%). Are 30 de coloane: V1–V28 (componente PCA), Time și Amount.
- **De ce acuratețea nu spune nimic:** un model care nu marchează nimic are 99,83% acuratețe. Metrica principală e **AUPRC pe test**.
- **Sarcina (Task 2):** un VQC (PennyLane + PyTorch) care detectează frauda cât mai bine sub zgomot depolarizant simulat, cu evidența costului pe hardware (execuții și shots).
- **Obligatoriu în livrare:**
  1. **Baseline-ul:** VQC-ul din Task 1, evaluat cu protocolul oficial pe 5 seed-uri.
  2. **Cel puțin o îmbunătățire** împotriva zgomotului, evaluată la fel, pe ≥ 5 seed-uri, comparată cu baseline-ul. Alegerea pragului pe validarea zgomotoasă face deja parte din protocol, deci nu contează ca îmbunătățire.
  3. **O referință clasică** antrenată pe aceleași intrări (cel puțin logistic regression).
  4. **Cost ledger:** execuții și shots pentru antrenare și validare, la baseline și la metoda noastră.
- **Livrabile:**
  - cod rulabil cap-coadă, cu seed-uri fixe;
  - prezentare (.pdf/.pptx) cu: pipeline, circuit (număr de porți, locurile de zgomot), îmbunătățire, rezultate (media ± abaterea, la toate nivelurile de zgomot), ledger, ce n-a mers.

### Protocolul oficial (cum suntem evaluați toți)

| Element | Regulă |
|---|---|
| Split | Stratificat 60/20/20, seed 42, înainte de orice balansare: train 170.883 rânduri (295 fraude), validare 56.962 (99), test 56.962 (98) |
| Zgomot | Circuitul descompus în {RX, RY, RZ, Rot, H, CNOT, CZ}. După fiecare poartă, `DepolarizingChannel(p)` pe fiecare qubit atins, simulat exact cu matrice de densitate. p ∈ {0, 0.001, 0.01, 0.05} |
| Prag | Cel care maximizează F1 pe scorurile de validare calculate la **același** p |
| Metrici | AUPRC (titlul se dă la p = 0.01 și 0.05), AUROC, Spearman ρ între scorurile cu zgomot și cele fără, precision, recall, F1, MCC. Media ± abaterea pe ≥ 5 seed-uri |
| Leakage | Balansarea, encodarea, arhitectura, epocile și pragul se aleg doar pe train și validare. Testul se folosește o singură dată, la final |

### Criteriile juriului

Performanța sub zgomot (AUPRC la p = 0.01 și 0.05, față de baseline și referința clasică) · robustețea (ρ și metricile de prag pe măsură ce crește zgomotul) · costul (execuții și shots) · calitatea circuitului (mai puțini qubiți sau adâncime mai mică e un plus) · stabilitatea pe seed-uri · designul experimental · originalitatea · codul · gândirea de cercetare (ipoteze, ablații, rezultate negative oneste) · înțelegerea soluției.

---

## 2. Ideea noastră: MOCO (Measure Once, Compile Once)

### Intuiția într-o frază

**Zgomotul e o taxă plătită în shots.** Am proiectat circuitul astfel încât zgomotul să **nu poată schimba ordinea tranzacțiilor**, doar să comprime scorurile spre ½. Am antrenat circuitul fără nicio execuție în bucla de optimizare, apoi am măsurat cât costă comprimarea pe hardware real.

### Circuitul (4 qubiți, 8 porți, 0 CNOT)

```
q0: ─RY(x₀)─Rot(φ₀,θ₀,0)─┐
q1: ─RY(x₁)─Rot(φ₁,θ₁,0)─┤  măsurăm H = Σ_q w_q Z_q   ->   p_fraud = (1 − ⟨H⟩) / 2
q2: ─RY(x₂)─Rot(φ₂,θ₂,0)─┤
q3: ─RY(x₃)─Rot(φ₃,θ₃,0)─┘
```
- **Encodare:** aceeași ca la kit. StandardScaler pe Time/Amount, PCA(4), MinMax la [0, π], totul antrenat doar pe train.
- **Dimensiuni:** 8 porți native, adâncime 2, **8 locuri de zgomot**. Baseline-ul are 93 de porți, 33 CNOT, adâncime 45 și 126 de locuri de zgomot.
- **Parametri:** 8 unghiuri de rotație plus 4 ponderi ale observabilului (normalizate, Σw = 1).

### Pașii metodei

1. **Măsurăm o singură dată.** RY(x)|0⟩ are vectorul Bloch r(x) = (sin x, 0, cos x). Pe hardware măsurăm fiecare tranzacție de train în bazele X și Z. Toți qubiții se citesc deodată, deci asta înseamnă **2 execuții per tranzacție, în total**.
2. **Imaginea Heisenberg.** După un Rot local, ⟨Z_q⟩ = m̂_q · r_q(x) exact, unde m̂_q e axa de măsurare rotită. Deci ⟨H⟩ = Σ_q w_q m̂_q · r_q(x) e **liniar în vectorii Bloch măsurați**.
3. **Antrenarea e convexă.** Antrenarea circuitului devine o regresie logistică pe [r_x, r_z] per qubit. O rezolvăm în PyTorch cu LBFGS, cu C ales prin cross-validation pe train.
   - Rezultatul e optimul global, nu depinde de inițializare și nu are barren plateaus.
   - Bucla de optimizare nu cere **nicio execuție**.
4. **Compilăm rezultatul înapoi în circuit.** Coeficienții c_q devin Rot(−φ_q, −θ_q, 0) și ponderi w_q = |c_q| / Σ|c|.
   - Scorul circuitului e o funcție monotonă a logit-ului, deci ordonarea e identică cu cea a modelului convex.
   - **Clasificarea o face circuitul**, cum cere regula.

### Teorema de robustețe (verificată numeric cu `noisy_qnode` din kit)

Un canal `DepolarizingChannel(p)` pe un qubit scalează vectorul Bloch cu **λ = 1 − 4p/3**. Circuitul nu are porți cu doi qubiți și are **același număr de porți k = 2 pe fiecare qubit**, deci starea rămâne produs și fiecare ⟨Z_q⟩ se scalează cu λ²:

> **s_zgomot(x) = ½ − λ²·(½ − s_ideal(x))**

Consecințe:
- Funcția e strict crescătoare, deci **AUPRC, AUROC și Spearman ρ = 1 sunt identice la orice p < 3/4**.
- **Pragul se transferă analitic:** t_p = ½ − λ²(½ − t₀) (`moco.transfer_threshold`).
- Testele confirmă formula la p ∈ {0.001, 0.05, 0.3}, cu eroare sub 1e-12 față de modelul oficial de zgomot.

### Unde contează totuși zgomotul: taxa în shots

Protocolul folosește valori așteptate exacte, adică shots infinite. Pe hardware, fiecare scor se estimează din S shots, cu varianță ≤ 1/S, iar semnalul e comprimat cu λ². Pentru aceeași calitate a ordonării:

> **S(p) ≈ S(0) / λ⁴**: de 1,32× mai multe shots la p = 0.05 și de 81× mai multe la p = 0.5

**Răspunsul nostru la hintul mentorului** („creșteți zgomotul și vedeți ce se întâmplă”) e **taxa în shots**, calculată determinist din comprimarea măsurată c(p) = mediana |s_p − ½| / |s_0 − ½|. Ca să păstreze același raport semnal/zgomot ca fără zgomot, un model are nevoie de **1/c² ori mai multe shots** (`qai/experiments/shot_tax.py`, `results/task2/shot_tax.csv`, figura `results/task2/shot_tax.png`; 5 seed-uri; diagnostic, nefolosit pentru nicio alegere):

| p | MOCO: c / taxa | R2 warm start: c / taxa | Baseline: c / taxa (limită inferioară) |
|---|---|---|---|
| 0.01 | 0.974 / 1,06× | 0.957 / 1,09× | 0.479 / 4,4× |
| **0.05** | **0.871 / 1,32×** | **0.795 / 1,58×** | **0.045 / 504×** (pe seed-uri: 250× – 2,2·10⁵×) |
| 0.1 | 0.751 / 1,77× | 0.616 / 2,6× | 0.0036 / 7,8·10⁴× |
| 0.2 | 0.538 / 3,5× | 0.339 / 8,7× | 1,4·10⁻⁵ / 4,9·10⁹× |
| 0.5 | 0.111 / 81× | 0.016 / 3.713× | 0 / infinit |

- **MOCO:** c = λ² exact, deci taxa e exact λ⁻⁴. E verificat cu `assert`, eroare 4·10⁻¹⁴.
- **Baseline:** 1/c² e doar o **limită inferioară** a costului. Comprimarea lui nu e uniformă pe tranzacții, iar ordonarea se pierde și cu shots infinite (ρ = 0.686 la p = 0.05), ceea ce nu se recuperează cu niciun număr de shots.

**Confirmarea cu shots reale, S = 1.024 per tranzacție, la p = 0.05:** AUPRC pe test **MOCO 0.705, R2 0.643, baseline 0.007**. Baseline-ul ajunge la nivelul șansei.

### Plătim taxa doar unde e nevoie: triaj adaptiv de shots (`exp_adaptive_shots`, 5 seed-uri, seturi complete)

Un shot citește toți cei 4 qubiți, deci dă estimatorul h = Σ_q w_q z_q ∈ [−1, 1], cu media ⟨H⟩_p. Măsurăm în loturi de b shots și ne oprim când un interval de încredere **valid oricând** (anytime-valid) pentru media lui h exclude pragul t_h = 1 − 2·t_p, cu un plafon S_max. Pragul t_p vine analitic din t₀ (`transfer_threshold`), iar t₀ se calculează clasic pe validare, deci nu e nevoie de nicio trecere zgomotoasă pe validare.

- **Garanția statistică:** P(intervalul ratează media la *oricare* lot) ≤ α. Prin urmare, P(o tranzacție e oprită devreme cu altă decizie decât cea a scorului exact) ≤ α.
  - Verificat pe 10.000 de secvențe simulate (`tests/test_adaptive.py`): rata de ratare e ≤ 0,08% la α = 1%, pentru toate cele trei intervale. Intervalele sunt Hoeffding și Bernstein Maurer-Pontil (cu α_k = 6α/(π²k²) pe lot) și Bernstein empiric cu varianță plug-in (Waudby-Smith & Ramdas 2024).
  - Controlul arată că testul are putere: un interval CLT reverificat la fiecare lot ratează în 11% din cazuri.
  - Pe date reale: **0 decizii greșite oprite devreme** în toate rulările. Toate cele 151 de verificări trec (`adaptive_checks.csv`).
- **Constatarea despre marje** (`results/task2/margins.csv`, validarea completă, circuitele din cache, 5 seed-uri). Cu encodarea kitului, ⟨H⟩ al tranzacțiilor legitime stă într-o bandă îngustă: între percentilele 5 și 95 sunt ~0.06 (seed 0: [0.226, 0.286]); pragul t_h variază de la un seed la altul.
  - Marja mediană |⟨H⟩ − t_h| e 0.161 ± 0.028, la o deviație standard per shot de 0.482 ± 0.050.
  - Cauza: MinMax-ul kitului e fixat de fraudele extreme, care strâng masa legitimă într-o bandă îngustă de unghiuri.
  - De aceea, cu intervalele care plătesc reuniunea pe loturi (Hoeffding, Maurer-Pontil), triajul nu economisește aproape nimic: ~1.000 de shots din 1.024.
- **Selecția s-a făcut doar pe validare** (`adaptive_selection_on_validation.csv`). Grila: q × interval × b × α × S_max, adică 270 de configurații MOCO. Criteriul: F1 maxim sub triaj la p = 0.05; la egalitate (în 0.005), mai puține shots.
  - Câștigă **encodarea kitului (q = 0) cu intervalul plug-in Bernstein, b = 16, α = 0.05, S_max = 4.096**: F1 pe validare 0.788 la 208 shots medii.
  - **Pârghia principală e intervalul:** la F1 egal, costul scade de la 992 de shots (Maurer-Pontil) la 208.
  - **Encodarea robustă mai taie ~2×, dar costă F1.** Ea fixează MinMax-ul pe percentilele [q, 100 − q] ale bag-ului de train (`RobustAngleEncoder`): q = 0.5% ajunge la 92 de shots (F1 0.778), q = 5% la 34 de shots (F1 0.732; F1 exact 0.737, față de 0.795).
  - Cel mai bun q > 0 (0.5%, aceeași regulă) e dus pe test doar ca termen de comparație. Nu a fost selectat.

Rezultate pe test, S_max = 4.096 (media ± abaterea pe 5 seed-uri; pragul e fixat înainte de test):

| Model | p | Shots medii (legit / fraudă) | Execuții / tranzacție | F1 | AUPRC | F1 cu S fix 1.024 | F1 cu S fix = bugetul adaptiv |
|---|---|---|---|---|---|---|---|
| MOCO kit (ales) | 0 | 161 ± 34 (160 / 636) | 10,1 | 0.801 ± 0.019 | 0.700 ± 0.013 | 0.804 ± 0.015 | 0.724 ± 0.046 |
| MOCO kit | 0.01 | 169 ± 36 (168 / 594) | 10,6 | 0.796 ± 0.023 | 0.694 ± 0.009 | 0.794 ± 0.020 | 0.716 ± 0.040 |
| MOCO kit | **0.05** | **208 ± 47** (208 / 739) | 13,0 | **0.801 ± 0.018** | 0.700 ± 0.009 | 0.797 ± 0.019 | 0.702 ± 0.060 |
| MOCO kit | 0.2 | 580 ± 147 (578 / 1.223) | 36,2 | 0.798 ± 0.020 | 0.695 ± 0.014 | 0.781 ± 0.006 | 0.686 ± 0.055 |
| MOCO robust q = 0.5% (neselectat) | 0.05 | 101 ± 9 (101 / 336) | 3,2 | 0.783 ± 0.023 | 0.726 ± 0.007 | 0.780 ± 0.029 | 0.744 ± 0.043 |
| MOCO robust q = 0.5% | 0.2 | 212 ± 26 | 6,6 | 0.784 ± 0.022 | 0.718 ± 0.027 | 0.778 ± 0.022 | 0.675 ± 0.081 |
| Baseline VQC | 0 | 40 ± 2 | 1,25 | 0.648 ± 0.034 | 0.576 ± 0.053 | 0.643 ± 0.033 | — |
| Baseline VQC | 0.05 | 4.089 (99,7% ating S_max) | 127,8 | 0.091 ± 0.072 | 0.086 ± 0.067 | 0.011 ± 0.005 | — |

Referința exactă (shots infinite): MOCO kit F1 0.800 ± 0.019, AUPRC 0.716 ± 0.007. Tabelul complet (S_max 1.024 și 4.096, toate p, varianta cu rafinare) e în `results/task2/adaptive_shots.csv`. Figuri: `adaptive_shots_histogram.png` și `adaptive_shots_f1_vs_shots.png`.

Ce arată:
- **−80% shots la p = 0.05** (208 față de 1.024), la același F1 (0.801 față de 0.797) și cu garanția α. Practic nicio tranzacție nu atinge plafonul.
- **La același buget, adaptivul bate S fix** (H3 confirmată): 0.801 față de 0.702 ± 0.060 cu 209 shots fixe la p = 0.05.
- **Shots cresc ca λ⁻⁴** (H4 confirmată): 161 → 208 → 580 de la p = 0 la 0.05 și 0.2, adică 1,30× și 3,6×, față de λ⁻⁴ = 1,32× și 3,46×.
- **Fraudele cer de 2–4× mai multe shots** decât tranzacțiile legitime, pentru că sunt mai aproape de prag.
- **AUPRC-ul calculat din scorurile de la oprire scade de la 0.716 la 0.700.** Varianta cu rafinare la 1.024 de shots doar pentru tranzacțiile marcate costă +1 shot în medie, dar nu îl recuperează (0.701). AUPRC-ul cere ordonarea fină a cozii legitime, pe care triajul o sacrifică deliberat.
- **Execuțiile cresc**, pentru că fiecare lot e o execuție: 13 per tranzacție la p = 0.05, față de 1. Pe hardware, câștigul real depinde de raportul dintre costul per shot și costul per job.
- **Baseline-ul (H5 confirmată):** la p = 0 triajul îl duce la 40 de shots, pentru că scorurile lui sunt răsfirate. La p = 0.05 însă 99,7% din tranzacții ating S_max = 4.096, iar F1 rămâne 0.09.
- **MOCO robust q = 0.5% prin protocolul oficial** (exact, `exp_moco --q 0.5`, 5 seed-uri): AUPRC 0.738 ± 0.006 la orice p, ρ = 1, F1 0.783 ± 0.024. LogReg 1:10 pe aceleași intrări robuste obține 0.731 ± 0.004, deci MOCO egalează din nou referința clasică. AUPRC-ul pe test e mai mare decât la kit (0.716), dar pe validare era mai mic (0.669 față de 0.675), iar F1 e mai mic. **Nu schimbăm alegerea după test.**

**Noise meter: estimăm p din N_cal tranzacții de calibrare.** Scorul s₀ se calculează clasic, deci λ² e panta lui (ŝ_p − ½) față de (s₀ − ½). Estimatorul celor mai mici pătrate e nedeplasat. Din λ² rezultă p̂ = ¾(1 − √λ²).
- **Eroarea relativă (RMSE) a lui p̂** (rânduri din validare, fără etichete, MOCO kit, S = 1.024), la N_cal = 50 / 200 / 1.000:

  | p | N_cal = 50 | N_cal = 200 | N_cal = 1.000 |
  |---|---|---|---|
  | 0.01 | 47% | 26% | 11% |
  | 0.05 | 11% | 5,3% | 2,4% |
  | 0.2 | 3,7% | 1,8% | 0,8% |

  Eroarea scade ~1/√(N_cal·S) (H6 confirmată). Mediana raportului are RMSE mai mare.
- **F1-ul se păstrează:** cu `transfer_threshold(t₀, p̂)`, F1 rămâne în 0.001 de F1-ul pragului calibrat pe toată validarea zgomotoasă (0.800), la orice N_cal.
- **Concluzie pentru ledger:** calibrarea pragului costă **N_cal = 1.000 de execuții în loc de 56.962**. Regula scrisă înainte de rulare (eroare relativă ≤ 10% la orice p) nu e atinsă la p = 0.01 pentru kit: acolo eroarea e 11%, adică 0.001 absolut, fără efect pe F1.

### Ce NU pretindem (poziționare onestă)

- **MOCO egalează referința clasică, nu o bate.** Clasa de modele e liniară în [sin z, cos z], practic o regresie logistică pe aceleași unghiuri. Noi pretindem un circuit robust, ieftin și ușor de antrenat, nu avantaj cuantic. Vezi [22] Bowles et al.
- **Circuitul e simulabil clasic** (stare produs). La 4 qubiți, toate circuitele din această competiție sunt simulabile clasic. Benchmark-ul măsoară zgomotul și costul.

---

## 3. Scara de metode (fiecare treaptă e o ipoteză)

| Treaptă | Ce schimbă | Ipoteză | Status |
|---|---|---|---|
| **R0** Baseline | VQC-ul kitului: ZZ×2 + EfficientSU2×3, 32 de parametri, 30 de epoci, 1:1 | referința | gata 0.580 la p = 0, 0.405 la p = 0.05 |
| **R1** Light-cone | Fără 12 rotații și 3 CNOT din afara conului de lumină al lui Z₀ | „Mai puține porți ajută sub zgomot” | gata **negativ:** scoruri identice la 1e-10, inclusiv la p = 0.05 |
| **R2** Superficial cu entanglement | RY(x), Rot(α), 3 CNOT, Rot(β), citire Σw Z. 15 porți, 18 locuri de zgomot. Variante: init aleatoare / warm start MOCO / warm start + antrenare la p = 0.05 | Warm start-ul din optimul convex bate inițializarea aleatoare (singura ipoteză scrisă înainte de rulare) | gata warm start 0.667 ± 0.010 față de aleator 0.635 ± 0.018; antrenarea sub zgomot nu ajută; sub MOCO |
| **R3** MOCO | Circuit produs, antrenare convexă, 8 porți | ρ = 1 prin construcție, cost minim | gata 5 seed-uri (secțiunea 4) |
| Ablații MOCO | `pair` (12 porți, 0 CNOT), `linear_cnot` (3 CNOT), `kit_zz` (encodarea baseline-ului) | Entanglement-ul rupe invarianța; encodarea ZZ e plafonul baseline-ului | gata toate patru reproduse local (5 seed-uri) |

**De ce R1 dă zero:** un canal pe un qubit care nu mai interacționează cu qubitul măsurat nu poate schimba starea redusă a acestuia. În plus, PennyLane sare deja la parameter-shift parametrii care nu ajung la Z₀ (Tracker: 41 de execuții per exemplu = 1 + 2×20, nu 65). Deci tăierea porților moarte nu aduce nici robustețe, nici execuții. Contează doar porțile din conul de lumină.

---

## 4. Rezultate până acum

### MOCO, protocolul oficial, 5 seed-uri, seturi complete gata (14:29, `exp_moco`, prin `noisy_qnode` din kit)

| p | AUPRC test | AUROC | Precision | Recall | F1 | MCC | Spearman ρ |
|---|---|---|---|---|---|---|---|
| 0 | 0.716 ± 0.007 | 0.967 ± 0.005 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.001 | 0.716 ± 0.007 | 0.967 ± 0.005 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.01 | 0.716 ± 0.007 | 0.967 ± 0.005 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.05 | 0.716 ± 0.007 | 0.967 ± 0.005 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |

Rezultatul e identic cu prima rulare (`moco_task2.py`, 13:30). Pragul se mută cu p exact după formula t_p = ½ − λ²(½ − t₀), deci și metricile de prag rămân identice.

### R0 baseline și LogReg, protocolul oficial, 5 seed-uri (14:27, `exp_task2_baseline`)

**Reproduce exact cifrele din enunț:** baseline 0.58 ± 0.05, LogReg 0.70 ± 0.01.

| p | Baseline AUPRC | Baseline ρ | Baseline F1 | Baseline MCC | LogReg 1:1 AUPRC | LogReg F1 / MCC |
|---|---|---|---|---|---|---|
| 0 | 0.580 ± 0.053 | 1.000 | 0.650 ± 0.031 | 0.651 ± 0.032 | 0.703 ± 0.010 | 0.791 / 0.791 |
| 0.001 | 0.580 ± 0.053 | 1.000 | 0.648 ± 0.037 | 0.649 ± 0.038 | 0.703 ± 0.010 | 0.791 / 0.791 |
| 0.01 | 0.564 ± 0.049 | 0.982 ± 0.010 | 0.659 ± 0.030 | 0.659 ± 0.030 | 0.703 ± 0.010 | 0.791 / 0.791 |
| 0.05 | **0.405 ± 0.177** | **0.686 ± 0.145** | 0.537 ± 0.184 | 0.545 ± 0.177 | 0.703 ± 0.010 | 0.791 / 0.791 |

Baseline, AUROC: 0.968 / 0.968 / 0.968 / 0.955. Ce se vede:
- **La p = 0.05 baseline-ul cade:** AUPRC −30%, iar abaterea pe seed-uri crește de 3×.
- **Pragul ajunge la 0.500 ± 0.001:** toate scorurile sunt strânse în jurul lui ½, ceea ce înseamnă o comprimare masivă din cauza celor 126 de locuri de zgomot.

### Comparația principală (AUPRC pe test, 5 seed-uri)

| Model | p = 0 | p = 0.01 | p = 0.05 | ρ la p = 0.05 |
|---|---|---|---|---|
| Baseline VQC (93 porți) | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 | 0.686 ± 0.145 |
| **MOCO (8 porți)** | **0.716 ± 0.007** | **0.716 ± 0.007** | **0.716 ± 0.007** | **1.000** |
| R2 entangled, warm start MOCO (15 porți) | 0.667 ± 0.010 | 0.667 ± 0.010 | 0.668 ± 0.009 | 0.999 ± 0.001 |
| **LogReg 1:10 = referința clasică corectă** (aceleași intrări ca MOCO: același set 1:10, aceleași 4 unghiuri) | **0.710 ± 0.007** | = | = | — |
| LogReg 1:1 (intrările baseline-ului) | 0.703 ± 0.010 | = | = | — |

**Referința clasică cu care se compară MOCO e LogReg 1:10, cu 0.710 ± 0.007** (`exp_moco`, 5 seed-uri), fiindcă are exact aceleași intrări. Diferența MOCO − LogReg (+0.006) e mai mică decât abaterea pe seed-uri, deci **MOCO egalează referința clasică**. CV-ul pe toate cele 492 de fraude (rulează) spune dacă diferența e reală. LogReg 1:10: F1 0.794 ± 0.023, MCC 0.795 ± 0.022, comparabile cu ale MOCO (0.800 / 0.801).

### R2 entangled, protocolul oficial, 5 seed-uri (15:01, `exp_entangled`)

| Variantă | AUPRC p = 0 | AUPRC p = 0.05 | ρ la p = 0.05 | F1 / MCC la p = 0.05 | Execuții (antrenare + validare) |
|---|---|---|---|---|---|
| inițializare aleatoare (ca în kit) | 0.635 ± 0.018 | 0.638 ± 0.015 | 0.999 ± 0.001 | 0.760 / 0.760 | 403.020 |
| warm start MOCO | **0.667 ± 0.010** | 0.668 ± 0.009 | 0.999 ± 0.001 | 0.798 / 0.798 | 412.755 |
| warm start MOCO + antrenare la p = 0.05 | 0.666 ± 0.009 | 0.667 ± 0.010 | 0.999 ± 0.001 | 0.796 / 0.796 | 412.755 |

Ce arată:
- **Warm start-ul ajută** (ipoteza scrisă dinainte, confirmată): +0.032 AUPRC și abaterea pe seed-uri înjumătățită.
- **Robust la zgomot, dar nu perfect:** cu 18 locuri de zgomot, ρ = 0.999 la p = 0.05, față de 0.686 la baseline.
- **Antrenarea sub zgomot nu aduce nimic.**
- **R2 rămâne sub MOCO single (0.716).** Cele 3 CNOT costă AUPRC pe aceste 4 componente PCA, la fel ca în ablația `linear_cnot`. Antrenarea variațională a lui α nu recuperează nimic peste optimul convex al lui `linear_cnot` (0.675).

Ipotezele H2–H4 din docstring au fost scrise după rulare și nu se prezintă ca predicții.

### Zgomot x shots, 5 seed-uri, seturi complete (16:35, `exp_noise_shots`) - diagnostic, măsurat pe test, nefolosit pentru nicio alegere

Comprimarea efectivă a scorurilor, mediana |s_p − ½| / |s_0 − ½| (1 = fără comprimare):

| p | MOCO (= λ²) | R2 warm start | Baseline |
|---|---|---|---|
| 0.01 | 0.974 | 0.957 | 0.479 |
| 0.05 | 0.871 | 0.795 | **0.045** |
| 0.1 | 0.751 | 0.616 | 0.004 |
| 0.5 | 0.111 | 0.016 | 0.000 |

AUPRC pe test cu S = 1024 shots per tranzacție (pragul pentru F1/MCC e ales pe validarea eșantionată la același p și S):

| p | MOCO | R2 warm start | Baseline |
|---|---|---|---|
| 0 | 0.708 | 0.660 | 0.577 |
| 0.01 | 0.702 | 0.658 | 0.553 |
| 0.05 | **0.705** | 0.643 | **0.007** |
| 0.1 | 0.703 | 0.642 | 0.002 |
| 0.3 | 0.634 | 0.297 | 0.002 |
| 0.5 | 0.213 | 0.003 | 0.002 |

Ce arată:
- **La p = 0.05 semnalul baseline-ului e comprimat de ~22 de ori.** Cu 1.024 de shots nu mai separă nimic: AUPRC 0.007, practic nivelul șansei. Cu valori exacte, protocolul oficial îi dădea încă 0.405.
- **MOCO pierde sub 0.01 AUPRC până la p = 0.1**, față de el însuși cu aceleași 1.024 de shots fără zgomot (0.708 → 0.703). Comprimarea lui e exact λ² (verificat, eroare 3e-17).
- **Aceasta e „taxa în shots”:** zgomotul nu strică ordonarea la MOCO, dar o face mai scumpă. La baseline o distruge.

Verificări: 44 din 45 trec (`results/task2/noise_shots_checks.csv`).
- Eșantionarea MOCO e exactă față de `noisy_predict` din kit pe 200 de rânduri, inclusiv distribuția comună egală cu produsul marginalelor.
- Citirea R2 din `qml.probs` coincide cu scorurile din protocol.
- **Un FAIL:** R2 (inițializare aleatoare) la p = 0, S = 16384. Cel mai rău seed iese la 0.0135 de valoarea exactă, peste limita de 0.01; media pe seed-uri e 0.0057.

Tabelul shots necesare (`results/task2/shots_needed.csv`) rămâne în repo doar ca diagnostic și **nu intră pe slide-uri**: e prea zgomotos (validarea și testul coincid în 24 din 40 de celule și nu e monoton în p). Criteriul lui nu se schimbă. În locul lui folosim taxa în shots deterministă (secțiunea 2). Detaliul e în „ce n-a mers”.

Figuri: `results/task2/noise_shots_heatmaps.png` și `results/task2/noise_shots_auprc_vs_p.png`.

### CV pe toate cele 492 de fraude (17:15, `exp_allfrauds_cv`, `results/task2/cv/`)

5 folduri stratificate ale întregului dataset (seed 42). În fiecare fold, restul se împarte 75/25 în train și validare. Fiecare fraudă e în test o singură dată. Protocolul oficial, media ± abaterea pe folduri:

| Model | AUPRC p = 0 | AUPRC p = 0.01 | AUPRC p = 0.05 | AUPRC p = 0.1 | ρ la p = 0.05 | ρ la p = 0.1 |
|---|---|---|---|---|---|---|
| MOCO | 0.721 ± 0.020 | 0.721 ± 0.020 | 0.721 ± 0.020 | 0.721 ± 0.020 | 1.000 | 1.000 |
| LogReg 1:10 (aceleași intrări) | 0.739 ± 0.030 | = | = | = | — | — |
| LogReg 1:1 | 0.698 ± 0.045 | = | = | = | — | — |
| Baseline VQC | 0.499 ± 0.137 | 0.529 ± 0.118 | 0.495 ± 0.139 | 0.434 ± 0.184 | 0.669 ± 0.306 | 0.433 ± 0.293 |

Ipotezele, scrise înainte de rulare:
- **H1 confirmată, după criteriul scris:** diferența MOCO − LogReg 1:10 pe fold e −0.018 ± 0.026, adică în limita abaterii dintre folduri (0.020 / 0.030). MOCO și LogReg nu se pot distinge. Avantajul de +0.006 al MOCO pe testul oficial nu rezistă însă: pe toate fraudele, LogReg e în față în 3 din 5 folduri (în al patrulea diferența e +0.0001). Formularea corectă: MOCO egalează referința clasică, nu o bate.
- **H2 neconfirmată:** căderea baseline-ului la p = 0.05 apare doar în 3 din 5 folduri (media 0.499 → 0.495). Căderea mare de pe testul oficial (0.580 → 0.405) depinde de fold. Ce se vede consecvent e pierderea ordonării: ρ scade la 0.67 la p = 0.05 și la 0.43 la p = 0.1.
- **H3 infirmată:** la zgomot mic, AUPRC-ul baseline-ului crește în toate cele 5 folduri (p = 0.001 și p = 0.01; în 2 din 5 la p = 0.05), în timp ce ρ scade (0.93 la p = 0.01). E exact paradoxul din articolul de pornire [1]: o singură metrică arată o îmbunătățire în timp ce modelul se strică. MOCO nu are această problemă, pentru că ordonarea lui e invariantă.

AUPRC-ul „pooled” (`cv_pooled_auprc.csv`, scorurile celor 5 modele de fold concatenate) **nu e valid pentru MOCO și baseline**: scorurile lor nu sunt calibrate între folduri (MOCO, de exemplu, nu are termen liber la compilare), deci concatenarea amestecă scale diferite. Pentru ei contează doar media pe folduri. Pentru LogReg, care dă probabilități calibrate, valoarea pooled (0.736) e consecventă cu media.

### Alte referințe

| Model | AUPRC test, fără zgomot | Sursa |
|---|---|---|
| Articolul Dinuț et al.: VQC / clasic tunat | 0.494 ± 0.139 / 0.60–0.74 | [1] |

### Selecția configurației, făcută doar pe validare (`results/selection_on_validation.csv`, generat de `qai/experiments/exp_selection.py`)

| Configurație | AUPRC validare (5 seed-uri) | Execuții de antrenare |
|---|---|---|
| single 1:1 | 0.564 ± 0.098 | 1.180 |
| single 1:3 | 0.663 ± 0.020 | 2.360 |
| **single 1:10** (ales) | **0.675 ± 0.012** | **6.490** |
| single 1:30 | 0.673 ± 0.018 | 18.290 |
| single 1:100 | 0.648 ± 0.015 | 59.590 |
| pair 1:30 (cea mai bună medie) | 0.676 ± 0.016 | 27.435 |

Regula de alegere: cea mai bună medie pe validare (pair 1:30) dă pragul; orice configurație aflată la cel mult o eroare standard de ea (abaterea pe seed-uri / √5) contează ca egală, iar dintre ele se alege cea mai ieftină: single 1:10. Regula a fost precizată după reproducerea studiului (rândurile pair s-au schimbat puțin); înainte, cu formularea „cea mai bună medie, la egalitate varianta mai ieftină”, se alesese tot single 1:10. Media ± abaterea standard de eșantion (ddof = 1), 5 seed-uri. Testul a fost atins o singură dată, la final.

### Ablații cu entanglement (aceeași antrenare MOCO, doar feature map-ul diferă)

| Feature map | Porți | CNOT | Locuri de zgomot | AUPRC p = 0 | AUPRC p = 0.05 | ρ la p = 0.05 | Sursa |
|---|---|---|---|---|---|---|---|
| single | 8 | 0 | 8 | 0.716 ± 0.007 | 0.716 ± 0.007 | 1.000 | local, 5 seed-uri |
| pair | 12 | 0 | 12 | 0.707 ± 0.024 | 0.707 ± 0.024 | 1.000 | local, 5 seed-uri |
| linear_cnot | 11 | 3 | 14 | 0.675 ± 0.034 | 0.667 ± 0.032 | 0.992 ± 0.002 | local, 5 seed-uri |
| kit_zz | 56 | 24 | 80 | 0.440 ± 0.104 | 0.450 ± 0.114 | 0.870 ± 0.126 | local, 5 seed-uri |

Ce arată (rezultatele locale confirmă ce raportase agentul claude.ai pe 3 seed-uri):
- doar 3 CNOT scad AUPRC cu ~0.04–0.05 și rup invarianța;
- encodarea ZZ a kitului rămâne la ~0.44 chiar și cu citirea optimă global, deci ea e plafonul baseline-ului. Sub zgomot, AUPRC-ul ei crește puțin (0.440 -> 0.450, adică în limita abaterii de 0.10 dintre seed-uri), în timp ce ρ scade la 0.87. Seamănă cu paradoxul din articol, unde o singură metrică arată o îmbunătățire deși ordonarea se strică. CV-ul pe toate fraudele verifică dacă efectul e real;
- `pair` nu aduce nimic peste `single`, pentru că plafonul e informația din cele 4 componente PCA.

### Alte rezultate [claude.ai] (de verificat)

- **ZNE pe linear_cnot**, p = 0.05, subset cu toate fraudele de test + 5.000 de tranzacții normale: ρ crește de la 0.9935 la 0.9991 cu Richardson 1-3-5. Pe kit_zz, rezultatele lin și Richardson sunt identice, suspect: probabil un bug de folding. De verificat cu `qml.noise.mitigate_with_zne` din kit.
- **Re-uploading pe qubiți independenți, antrenat cu Adam:** 0.554 ± 0.06, sub baseline.
- **Ansamblul de circuite mici „NEPQE”:** 0.554.
- **Selecție Pearson top-4** în loc de PCA4: 0.713 față de 0.717. Nu e un câștig.
- **Validare pe un subset de 20%:** ledger 29.276 de execuții (−99,0%), dar F1 cu ~0.02 mai mic (0.779 față de 0.800).

---

## 5. Cost ledger (execuții pe hardware, 1.024 shots per execuție)

### Rezumat (trei definiții, de la cea a kitului la cea mai conservatoare)

1. **Definiția kitului (antrenare + selecția modelului):** baseline 2.859.360 execuții, MOCO 6.490. Reducere **−99,77%**.
2. **Plus calibrarea pragului cu noise meter-ul:** baseline 2.916.322, MOCO 6.490 + 1.000 = **7.490** execuții. Reducere **−99,74%**. N_cal = 1.000 vine din `results/task2/noise_meter.csv` (estimatorul celor mai mici pătrate, 5 seed-uri; secțiunea 2).
3. **Cea mai conservatoare, pragul calibrat pe toată validarea zgomotoasă:** MOCO 63.452. Reducere **−97,8%**.

### Inferență pe hardware: shots per tranzacție, fix vs adaptiv (`results/task2/ledger_adaptive.csv`)

| Metodă | p | Execuții / tranzacție | Shots / tranzacție (inferență) | Shots pe tot testul (56.962) |
|---|---|---|---|---|
| MOCO, S fix (oricare model) | oricare | 1 | 1.024 | 58,3 mil |
| MOCO kit, adaptiv (plug-in Bernstein, b = 16, S_max = 4.096) | 0 | 10,1 | 161 | 9,2 mil |
| MOCO kit, adaptiv | 0.05 | 13,0 | 208 | 11,9 mil |
| MOCO kit, adaptiv | 0.2 | 36,2 | 580 | 33,0 mil |
| MOCO robust q = 0.5%, adaptiv (neselectat) | 0.05 | 3,2 | 101 | 5,8 mil |
| Baseline VQC, adaptiv | 0.05 | 127,8 | 4.089 | 232,9 mil |
| Calibrarea pragului: trecere zgomotoasă pe validare (protocol) | oricare | — | — | 56.962 execuții × 1.024 |
| **Calibrarea pragului cu noise meter-ul** | oricare | — | — | **1.000 execuții × 1.024** |

Formula pentru adaptiv: Σ_i K_i execuții și b·Σ_i K_i shots, unde K_i e numărul de loturi ale tranzacției i. Verificarea cu `qml.Tracker`, pe dispozitivul zgomotos al kitului (5 rânduri, p = 0.05): 79 de execuții și 1.264 de shots, egale cu formula.

### Ledger final (`results/task2/ledger_final.csv`, generat de `qai/experiments/ledger_final.py` doar din CSV-urile existente)

| Metodă | Antrenare | Selecția modelului | Calibrarea pragului | Total execuții | Total shots | Reducere vs baseline (formulă / Tracker) | Taxa în shots la p = 0.05 | Sursa |
|---|---|---|---|---|---|---|---|---|
| baseline VQC (kit formula, 65 per example) | 1.150.500 | 1.708.860 | 56.962 | 2.916.322 | 2.986.313.728 | — / — | 504× | `ledger_baseline.csv` |
| baseline VQC (qml.Tracker, 41 per example) | 725.700 | 1.708.860 | 56.962 | 2.491.522 | 2.551.318.528 | −14,57% / — | 504× | `ledger_baseline.csv` |
| MOCO single 1:10 | 6.490 | 0 | 56.962 | 63.452 | 64.974.848 | −97,82% / −97,45% | 1,32× | `ledger_moco_single_10.csv` |
| R2 entangled (warm start) | 298.835 | 113.920 | 56.962 | 469.717 | 480.990.208 | −83,89% / −81,15% | 1,58× | `ledger_entangled.csv` |
| R2 entangled (random init) | 289.100 | 113.920 | 56.962 | 459.982 | 471.021.568 | −84,23% / −81,54% | 1,97× | `ledger_entangled.csv` |
| footnote: one-off selection study (fm x ratio on validation, not in MOCO's total) | 0 | 390.661 | 0 | 390.661 | 400.036.864 |  |  | `ledger_moco_single_10.csv` |

Calibrarea pragului = o trecere de scorare a validării (56.962 de rânduri) la nivelul de zgomot al dispozitivului, pentru pragul max-F1. E pasul de evaluare din protocol, identic pentru toate metodele. **MOCO îl poate înlocui cu `moco.transfer_threshold`:** scorurile lui fără zgomot pe validare sunt o formulă închisă a unghiurilor clasice (nicio execuție), iar pragul la p rezultă analitic, t_p = ½ − λ²(½ − t₀). Așa MOCO coboară la 6.490 de execuții în total. Taxa în shots înmulțește costul de inferență pe hardware la p = 0.05, pentru aceeași calitate a ordonării ca fără zgomot; la baseline e o limită inferioară.


### Detaliu pe metode, cu formula și verificarea Tracker

Fiecare rând are formula și verificarea cu `qml.Tracker` pe 5 exemple reale de antrenare (coloana „Tracker”).

| Metodă | Execuții per exemplu: formulă / Tracker | Antrenare | Validare | Total | Shots | Față de baseline |
|---|---|---|---|---|---|---|
| **Baseline** (formula kitului: 30 × 590 × (1 + 2×32)) | 65 / **41** | 1.150.500 (Tracker: 725.700) | 1.708.860 (30 × 56.962) | 2.859.360 (Tracker: 2.434.560) | 2,93 mld | — |
| **MOCO single** (2 baze × 3.245 rânduri; C prin CV pe train) | 2 / 2 | **6.490** | **0** | **6.490** | **6,6 mil** | **−99,77%** |
| MOCO linear_cnot / kit_zz (ablații, 3 baze) | 3 / 3 | 9.735 | 0 | 9.735 | 10,0 mil | −99,66% |
| R2 aleator (10 epoci × 590 × 49; validare pe 20% din setul de validare) | 49 / 49 | 289.100 | 113.920 | 403.020 | 413 mil | −85,9% |
| R2 warm start MOCO (+ 3 baze × 3.245 pentru warm start) | 49 / 49 | 298.835 | 113.920 | 412.755 | 423 mil | −85,6% |
| Studiul de selecție, cost unic (10 configurații, un seed + validarea măsurată o dată în X/Y/Z) | — | 219.775 | 170.886 | 390.661 | 400 mil | se raportează separat |

Tracker-ul confirmă formulele pentru MOCO și R2. La baseline numără 41, nu 65: PennyLane nu deplasează parametrii care nu ajung la Z₀ (vezi R1).

- **Evaluarea din protocol** (scorarea validării și a testului la fiecare p) e identică pentru toate metodele: 2 × 56.962 execuții per nivel de zgomot. Se raportează separat.
- **Taxa în shots sub zgomot** (de câte ori mai multe shots trebuie la p, față de fără zgomot, pentru același raport semnal/zgomot; 1/c² din comprimarea măsurată; diagnostic, nefolosit pentru nicio alegere). Execuțiile de pe hardware ale oricărui model se înmulțesc cu acest factor dacă vrem aceeași calitate a ordonării ca fără zgomot:

| p | MOCO: c / taxa | R2 warm start: c / taxa | Baseline: c / taxa (limită inferioară) |
|---|---|---|---|
| 0.01 | 0.974 / 1,06× | 0.957 / 1,09× | 0.479 / 4,4× |
| **0.05** | **0.871 / 1,32×** | **0.795 / 1,58×** | **0.045 / 504×** (pe seed-uri: 250× – 2,2·10⁵×) |
| 0.1 | 0.751 / 1,77× | 0.616 / 2,6× | 0.0036 / 7,8·10⁴× |
| 0.2 | 0.538 / 3,5× | 0.339 / 8,7× | 1,4·10⁻⁵ / 4,9·10⁹× |
| 0.5 | 0.111 / 81× | 0.016 / 3.713× | 0 / infinit |

  La p = 0.05, MOCO costă 1,32× shots, iar baseline-ul cel puțin 504×. Cu S = 1.024: AUPRC MOCO 0.705, R2 0.643, baseline 0.007. Orice parametru care *e* o alegere (de exemplu b, α, S_max în experimentul cu shots adaptive) se alege pe validare.

---

## 6. Codul (cum se reproduce totul)

| Fișier | Rol |
|---|---|
| `qai/protocol.py` | Protocolul **copiat exact din kit**, cu celula de notebook trecută la fiecare funcție: split + `split_indices.npz`, undersample, AngleEncoder, `noisy_qnode`, prag, metrici, ledger, Tracker |
| `qai/kit_vqc.py` | Baseline-ul kitului (R0) și `vqc_lightcone` (R1) |
| `qai/moco.py` | MOCO: feature map-uri, vectori Bloch, fit convex, compilare, circuit, transfer de prag |
| `qai/experiments/exp_task2_baseline.py` | R0 + LogReg 1:1, 5 seed-uri, cu cache de scoruri |
| `qai/experiments/exp_moco.py` | R3 + LogReg 1:10. `--fm linear_cnot / kit_zz / pair` pentru ablații, `--q 0.5` pentru encodarea robustă |
| `qai/adaptive.py` | Intervale valide oricând (Hoeffding, Maurer-Pontil, plug-in Bernstein), triajul pe loturi, eșantionarea per qubit, noise meter |
| `qai/experiments/exp_adaptive_shots.py` | Triajul adaptiv: grila pe validare, selecția, testul, noise meter-ul, verificările, ledger-ul de inferență, figurile |
| `tests/test_adaptive.py` | Acoperirea pe 10.000 de secvențe (≤ α), controlul CLT naiv, deciziile timpurii greșite ≤ α, encoder-ul robust cu q = 0 identic cu cel al kitului |
| `qai/experiments/exp_entangled.py` | R2: 3 variante × 5 seed-uri |
| `qai/experiments/exp_noise_shots.py` | p până la 0.5 × S ∈ {64…16384}, eșantionare exactă per model (binomial pe Z₀ / binomiale independente / multinomial pe 16 rezultate), prag pe validarea eșantionată la același (p, S), comprimare, shots necesare (diagnostic) |
| `qai/experiments/exp_allfrauds_cv.py` | CV pe 5 folduri a întregului dataset: toate cele 492 de fraude în test |
| `qai/experiments/task2_report.py` | Totul adunat în `results/task2/TABLES.md` |
| `tests/test_protocol.py` | 16 teste: numerele kitului, R1 identic, teorema λ², comprimarea = λ², Heisenberg cu CNOT, citirea R2 din distribuția comună, Tracker = formula, split-ul |
| `moco_task2.py` | Versiunea inițială, de sine stătătoare (păstrată ca referință) |
| `task1/` | Notebook-ul Task 1 completat de echipă (copie, `kit/` neatins) + README: 20/20 exerciții trecute, celulele kitului refolosite în Task 2, exercițiile 3.2 și 10.1 pe care se sprijină teorema λ²; se rulează o dată cap-coadă înainte de predare |

```bash
make task2-test     # testele (~10 s)
make task2-fast     # smoke run pe 10% din date (~2 min)
make task2          # tot cap-coadă, seed-uri fixe, cache în results/task2/ (ore, în fundal)
python3 -m qai.experiments.task2_report   # regenerează TABLES.md
```

**Reguli pentru agenți:**
- Nu se aleg configurații după test.
- Orice cifră nouă intră în acest document, cu sursa ei.
- Cifrele de pe 10% din date (`_fast`) nu se raportează niciodată.

---

## 7. Răspunsuri pregătite pentru juriu

| Întrebare | Răspuns |
|---|---|
| „Nu e doar o regresie logistică?” | Ca clasă de modele, da, și o arătăm: AUPRC egal cu LogReg pe aceleași intrări. Contribuția e că circuitul care o implementează are 8 porți, ρ = 1 garantat sub zgomot și se antrenează cu 2 execuții per rând |
| „Mai e VQC?” | Da. Parametrii sunt unghiurile de rotație ale circuitului, iar clasificarea vine din observabilul măsurat. Antrenarea e o reparametrizare convexă exactă a pierderii VQC-ului, din imaginea Heisenberg. R2 arată și varianta variațională clasică |
| „Invarianța la zgomot e banală” | E o teoremă, și tocmai asta e scopul designului. Partea care nu e banală e costul: pe hardware, comprimarea λ² se plătește în shots, iar noi am măsurat cât |
| „E simulabil clasic” | Adevărat, ca toate circuitele de 4 qubiți din competiție. Benchmark-ul măsoară zgomotul și costul, nu avantajul cuantic [22] |
| „De ce 1:10?” | Tabelul de selecție pe validare: 5 seed-uri, testul neatins |
| „Leakage?” | Encoder-ul e antrenat doar pe train, C e ales prin CV pe train, pragul pe validare, iar testul e folosit o singură dată |
| „Și cu entanglement?” | Ablațiile și R2: entanglement-ul costă AUPRC și rupe invarianța. Rezultatul îl raportăm cinstit |
| „De ce căderea baseline-ului sub zgomot nu apare în fiecare fold?” | AUPRC-ul exact al baseline-ului e instabil: ±0.137 între folduri, de 7 ori mai mult decât MOCO (±0.020). Pe testul oficial a căzut de la 0.580 la 0.405, dar pe toate cele 492 de fraude scade doar în 3 din 5 folduri. Efectele consecvente sunt altele: pierderea ordonării (ρ = 0.67 la p = 0.05, în toate foldurile) și prăbușirea cu shots reale (AUPRC 0.007 cu 1.024 de shots la p = 0.05). La zgomot mic, AUPRC-ul chiar crește în 5 din 5 folduri în timp ce ρ scade, adică paradoxul din articolul [1] (figura k) |
| „E MOCO mai bun decât regresia logistică?” | Nu. O egalează: pe toate cele 492 de fraude, MOCO − LogReg 1:10 = −0.018 ± 0.026 pe fold, în limita variației. Contribuția e alta: robustețea (ρ = 1 la orice p, taxa în shots 1,32× la p = 0.05), stabilitatea (±0.020 între folduri, față de ±0.137 la baseline) și costul (6.490 de execuții de antrenare, −99,77%) |
| „De ce ar conta pe hardware real?” | Pe hardware fiecare scor se plătește în shots. **Cu encodarea kitului și shots fixe, MOCO plătește ~1.000 de shots per tranzacție (1.024)**, pentru că tranzacțiile legitime stau la o marjă mediană de doar 0.16 de prag, față de o deviație standard per shot de 0.48 (`margins.csv`; MinMax-ul e fixat de fraudele extreme). Costul îl coborâm cu două pârghii. **Triajul adaptiv**, cu un interval valid oricând, îl reduce la 161–208 shots la p ≤ 0.05 (−80%), la același F1 (0.801) și cu garanție statistică: cel mult α decizii greșite oprite devreme (măsurat: 0). **Scalarea robustă a encodării** (percentile în loc de min/max) lărgește marjele și mai taie costul absolut ~2×, la ~100 de shots, cu 0.01–0.02 F1 mai puțin, așa că pe validare n-a fost aleasă. Pragul îl calibrăm cu noise meter-ul din 1.000 de execuții, în loc de 56.962. La baseline triajul nu funcționează sub zgomot: 99,7% din tranzacții ating plafonul |

---

## 8. Ce mai e de făcut

| # | Sarcină | Cine | Status |
|---|---|---|---|
| 1 | Protocolul oficial + teste | Claude Code | gata |
| 2 | Baseline + LogReg, 5 seed-uri | Claude Code | gata 14:27 (reproduce 0.58 / 0.70 din enunț) |
| 3 | MOCO prin protocol + ablații linear_cnot / kit_zz | Claude Code | gata, 14:29–14:47 |
| 4 | R2 entangled, 3 variante | Claude Code | gata, 15:01 |
| 5 | Zgomot × shots (diagnostic) | Claude Code | gata, 16:35 (44/45 verificări trec) |
| 6 | CV pe toate cele 492 de fraude | Claude Code | gata, 17:15 (H1 confirmată, H2 neconfirmată, H3 infirmată) |
| 7 | Grafice pentru slide-uri (AUPRC(p), harta shots, schema circuitului, ledger) | Claude Code | gata (`results/task2/figures/`, a–n) |
| 8 | ZNE pe R2, cu execuțiile trecute în ledger (opțional) | — | renunțat (nu a fost rulat) |
| 9 | Slide-urile | echipa | gata (`presentation/`) |
| 10 | Commit + push pe branch (`TDWEX`) după fiecare experiment | Claude Code | rulează (primul: f540bc9) |
| 11 | Cifrele finale în slide-uri, repetiție | toți | de făcut până la 07:00 |
| 12 | Shots adaptive + noise meter + MOCO robust prin protocol | Claude Code | gata, 23:50 (151/151 verificări trec) |

**Opriri ferme:** experimente noi până la 05:00 · cifrele blocate la 07:00 · upload până la 08:30.

---

## 9. Prezentarea, lucrarea și documentele de diagnostic

- **Prezentarea:** `presentation/moco_slides.pdf` (15 slide-uri + 3 de rezervă, B1–B3), generată de `presentation/build_slides.py` direct din CSV-uri; versiunea de 5 minute `presentation/moco_5min.pdf`; notele vorbitorilor în `presentation/SPEAKER_NOTES.md`.
- **Lucrarea:** `paper/moco.pdf` (sursa `paper/moco.tex`).
- **Diagnostice:** `docs/DIAGNOSTICS.md` (encoder, corelații, exemplu cap-coadă, precizia la recall fix) și `docs/CONFIDENCE_SWEEP.md` (sweep-ul de încredere al triajului); figurile l (`l_confidence_sweep.png`), m (`m_corr_heatmap.png`) și n (`n_pr_curve.png`) din `results/task2/figures/`.
- Cifre de referință pentru prezentare: encodarea ZZ a kitului rămâne la 0.440 chiar cu citirea optimă; referința clasică corectă e LogReg 1:10, 0.710 ± 0.007.

---

## 10. Referințe folosite

[1] Dinuț et al. 2026 (punctul de plecare) · [2] Dal Pozzolo 2015 (undersampling) · [7] Schuld et al. 2019 (parameter shift) · [11] Spall 1992 (SPSA) · [13] McClean 2018 / [14] Grant 2019 (barren plateaus, inițializare) · [15] Wang 2021 (barren plateaus induse de zgomot) · [16] QuantumNAT 2022 (antrenare cu zgomot) · [17] Temme 2017 (ZNE) · [20] Schreiber 2023 (surogate clasice) · [22] Bowles 2024 (benchmarking QML) · Huang, Kueng, Preskill 2020 (classical shadows: învățare din măsurători făcute o dată) · Havlíček 2019 (ZZ feature map).

---

## Changelog

- **24.10, 23:50:** shots adaptive (`exp_adaptive_shots`, 5 seed-uri, selecție doar pe validare). MOCO kit cu plug-in Bernstein: 208 ± 47 shots per tranzacție la p = 0.05 (−80% față de 1.024), F1 0.801 ± 0.018, 0 decizii greșite oprite devreme. Shots cresc ca λ⁻⁴. Baseline-ul atinge plafonul în 99,7% din cazuri. Constatarea despre marje (corectat pe 25.10 din `margins.csv`: mediana 0.16 în unități ⟨H⟩) și rezultatul negativ cu Hoeffding / Maurer-Pontil. MOCO robust q = 0.5% prin protocol: 0.738 ± 0.006 (LogReg pe aceleași intrări 0.731 ± 0.004), neselectat pe validare. Noise meter: N_cal = 1.000 de execuții în loc de 56.962.

- **24.10, 17:40:** figurile i (CV: MOCO egalează LogReg, de 7x mai stabil decât baseline-ul) și k (paradoxul: AUPRC-ul baseline-ului crește la zgomot mic în timp ce ρ scade); două răspunsuri noi pentru juriu în secțiunea 7.
- **24.10, 17:30:** ablația `pair` reprodusă local pe 5 seed-uri: 0.707 ± 0.024 la orice p, ρ = 1 (claude.ai raportase 0.708 ± 0.023 pe 3 seed-uri). Figura g nu mai amestecă sursele.
- **24.10, 17:25:** rezumat în trei rânduri deasupra ledger-ului (secțiunea 5): definiția kitului −99,77%, varianta cu noise meter (N_cal de completat din `noise_meter.csv`), varianta conservatoare −97,8%.
- **24.10, 17:20:** CV pe toate cele 492 de fraude: MOCO 0.721 ± 0.020 la orice p, LogReg 1:10 0.739 ± 0.030 (H1 confirmată: nu se pot distinge; MOCO egalează, nu bate), baseline 0.499 ± 0.137, la p = 0.05 0.495 ± 0.139 (H2 neconfirmată: cade doar în 3 din 5 folduri), ρ 0.669; AUPRC-ul baseline-ului crește la zgomot mic în 5 din 5 folduri în timp ce ρ scade (H3 infirmată, paradoxul din articol). AUPRC-ul pooled nu e valid pentru modelele necalibrate.
- **24.10, 17:10:** ledger final (`ledger_final.csv`): total cu calibrarea pragului, baseline 2.916.322 (formula) / 2.491.522 (Tracker), MOCO 63.452 (−97,8% / −97,5%; 6.490 dacă pragul se transferă analitic), R2 warm start 469.717; taxa în shots la p = 0.05 pe fiecare rând; studiul de selecție ca notă separată.
- **24.10, 17:00:** taxa în shots deterministă 1/c² din comprimarea măsurată (`shot_tax.csv`, `shot_tax.png`), ca răspuns principal la hintul mentorului: la p = 0.05, MOCO 1,32× (= λ⁻⁴, verificat), R2 1,58×, baseline cel puțin 504×; cu S = 1.024, AUPRC MOCO 0.705, R2 0.643, baseline 0.007. Tabelul „shots necesare” scos de pe slide-uri (H3 neconfirmată), păstrat ca diagnostic. Notebook-ul completat al colegei e în `task1/`; `kit/` conține materialul organizatorilor neschimbat (notebook-ul original a fost restaurat la 25.10, iar ghidul echipei a fost mutat în `docs/ghid_task1.md`).
- **24.10, 16:45:** zgomot × shots gata (diagnostic): la p = 0.05 și S = 1024, baseline 0.007 față de MOCO 0.705; comprimarea baseline-ului 0.045, a MOCO 0.871 = λ². 44/45 verificări trec (FAIL: R2 aleator, un seed, 0.0135 > 0.01). H3 (validare = test pentru shots necesare) nu se confirmă. Adăugat `task1/` cu notebook-ul Task 1 completat (20/20 exerciții, 0 erori, split 170883/56962/56962) și README. CV pe toate fraudele pornit la 16:37.
- **24.10, 16:05:** referința clasică corectă e LogReg 1:10 = 0.710 ± 0.007 (aceleași intrări ca MOCO). Ablațiile linear_cnot (0.675 ± 0.034) și kit_zz (0.440 ± 0.104) reproduse local pe 5 seed-uri. R2: warm start 0.667 ± 0.010, față de aleator 0.635 ± 0.018. Ledger cu Tracker: baseline 41/ex (formula 65), MOCO 2, R2 49. Zgomot × shots rescris după specificație, pornit la 15:29 (diagnostic, nefolosit pentru alegeri). Marcajele grafice au fost înlocuite cu cuvinte.
- **24.10, 14:27:** baseline-ul și LogReg pe 5 seed-uri, prin protocolul oficial. Reproduce enunțul (0.580 ± 0.053 / 0.703 ± 0.010). La p = 0.05 baseline-ul cade la 0.405 ± 0.177 (ρ 0.686), iar MOCO rămâne la 0.716 ± 0.007.
- **24.10, 14:25:** document creat. Protocolul și testele sunt gata, R1 e negativ, MOCO are 5 seed-uri, selecția s-a făcut pe validare. Baseline-ul rulează, iar restul experimentelor sunt în coadă.
