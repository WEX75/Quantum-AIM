# BreaQ 2026 · Task 2 · Proiectul nostru: MOCO

> **Document viu.** Sursa unică pentru echipă, pentru agenții AI care lucrează cu noi și pentru slide-uri.
> Se actualizează după fiecare rezultat nou (vezi *Changelog* la final). Cifrele marcate „rulează” sunt încă în calcul.
> Cifrele marcate [claude.ai] vin de la agentul din claude.ai și nu au fost încă reproduse în acest repo.
>
> **Ultima actualizare:** sâmbătă 24.10, 16:45
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

La baseline, comprimarea vine din ~126 de locuri de zgomot și nu e uniformă. Strică ordonarea chiar și cu shots infinite, iar taxa în shots e mult mai mare (rulează `exp_noise_shots`). **Acesta e răspunsul nostru la hintul mentorului** („creșteți zgomotul și vedeți ce se întâmplă”).

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
| Ablații MOCO | `pair` (12 porți, 0 CNOT), `linear_cnot` (3 CNOT), `kit_zz` (encodarea baseline-ului) | Entanglement-ul rupe invarianța; encodarea ZZ e plafonul baseline-ului | gata linear_cnot și kit_zz reproduse local (5 seed-uri); `pair` rămâne [claude.ai] |

**De ce R1 dă zero:** un canal pe un qubit care nu mai interacționează cu qubitul măsurat nu poate schimba starea redusă a acestuia. În plus, PennyLane sare deja la parameter-shift parametrii care nu ajung la Z₀ (Tracker: 41 de execuții per exemplu = 1 + 2×20, nu 65). Deci tăierea porților moarte nu aduce nici robustețe, nici execuții. Contează doar porțile din conul de lumină.

---

## 4. Rezultate până acum

### MOCO, protocolul oficial, 5 seed-uri, seturi complete gata (14:29, `exp_moco`, prin `noisy_qnode` din kit)

| p | AUPRC test | AUROC | Precision | Recall | F1 | MCC | Spearman ρ |
|---|---|---|---|---|---|---|---|
| 0 | 0.716 ± 0.007 | 0.967 ± 0.006 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.001 | 0.716 ± 0.007 | 0.967 ± 0.006 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.01 | 0.716 ± 0.007 | 0.967 ± 0.006 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |
| 0.05 | 0.716 ± 0.007 | 0.967 ± 0.006 | 0.781 ± 0.048 | 0.822 ± 0.014 | 0.800 ± 0.019 | 0.801 ± 0.019 | 1.0 |

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

Tabelul shots necesare (`results/task2/shots_needed.csv`) coincide între validare și test doar în 24 din 40 de celule. Ipoteza H3 („validare = test pentru MOCO”) **nu se confirmă**. Cu 98–99 de fraude per set, pragul de 0.01 AUPRC e mai fin decât zgomotul statistic al estimării.

Figuri: `results/task2/noise_shots_heatmaps.png` și `results/task2/noise_shots_auprc_vs_p.png`.

### Alte referințe

| Model | AUPRC test, fără zgomot | Sursa |
|---|---|---|
| Articolul Dinuț et al.: VQC / clasic tunat | 0.494 ± 0.139 / 0.60–0.74 | [1] |

### Selecția configurației, făcută doar pe validare (`results/selection_on_validation.csv`)

| Configurație | AUPRC validare (5 seed-uri) | Execuții de antrenare |
|---|---|---|
| single 1:1 | 0.564 ± 0.087 | 1.180 |
| single 1:3 | 0.663 ± 0.018 | 2.360 |
| **single 1:10** (ales) | **0.675 ± 0.011** | **6.490** |
| single 1:30 | 0.673 ± 0.016 | 18.290 |
| single 1:100 | 0.648 ± 0.013 | 59.590 |
| pair 1:30 | 0.675 ± 0.009 | 27.435 |

Regula de alegere: cea mai bună medie pe validare, iar la egalitate varianta mai ieftină. Testul a fost atins o singură dată, la final.

### Ablații cu entanglement (aceeași antrenare MOCO, doar feature map-ul diferă)

| Feature map | Porți | CNOT | Locuri de zgomot | AUPRC p = 0 | AUPRC p = 0.05 | ρ la p = 0.05 | Sursa |
|---|---|---|---|---|---|---|---|
| single | 8 | 0 | 8 | 0.716 ± 0.007 | 0.716 ± 0.007 | 1.000 | local, 5 seed-uri |
| pair | 12 | 0 | 12 | 0.708 ± 0.023 | 0.708 ± 0.023 | 1.000 | [claude.ai], 3 seed-uri |
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
- **Taxa în shots sub zgomot:** câte shots trebuie ca AUPRC să rămână la cel mult 0.01 de valoarea fără zgomot, în funcție de p. rulează `results/task2/shots_needed.csv`. **Diagnostic, măsurat pe test (și pe validare, ca verificare), nefolosit pentru nicio alegere.** Niciun model, prag sau parametru nu se alege cu el. Orice parametru care *e* o alegere (de exemplu b, α, S_max în experimentul viitor cu shots adaptive) se alege pe validare.

---

## 6. Codul (cum se reproduce totul)

| Fișier | Rol |
|---|---|
| `qai/protocol.py` | Protocolul **copiat exact din kit**, cu celula de notebook trecută la fiecare funcție: split + `split_indices.npz`, undersample, AngleEncoder, `noisy_qnode`, prag, metrici, ledger, Tracker |
| `qai/kit_vqc.py` | Baseline-ul kitului (R0) și `vqc_lightcone` (R1) |
| `qai/moco.py` | MOCO: feature map-uri, vectori Bloch, fit convex, compilare, circuit, transfer de prag |
| `qai/experiments/exp_task2_baseline.py` | R0 + LogReg 1:1, 5 seed-uri, cu cache de scoruri |
| `qai/experiments/exp_moco.py` | R3 + LogReg 1:10. `--fm linear_cnot / kit_zz / pair` pentru ablații |
| `qai/experiments/exp_entangled.py` | R2: 3 variante × 5 seed-uri |
| `qai/experiments/exp_noise_shots.py` | p până la 0.5 × S ∈ {64…16384}, eșantionare exactă per model (binomial pe Z₀ / binomiale independente / multinomial pe 16 rezultate), prag pe validarea eșantionată la același (p, S), comprimare, shots necesare (diagnostic) |
| `qai/experiments/exp_allfrauds_cv.py` | CV pe 5 folduri a întregului dataset: toate cele 492 de fraude în test |
| `qai/experiments/task2_report.py` | Totul adunat în `results/task2/TABLES.md` |
| `tests/test_protocol.py` | 16 teste: numerele kitului, R1 identic, teorema λ², comprimarea = λ², Heisenberg cu CNOT, citirea R2 din distribuția comună, Tracker = formula, split-ul |
| `moco_task2.py` | Versiunea inițială, de sine stătătoare (păstrată ca referință) |
| `task1/` | Notebook-ul Task 1 completat de echipă (copie, `kit/` neatins) + README: 20/20 exerciții trecute, celulele kitului refolosite în Task 2, exercițiile 3.2 și 10.1 pe care se sprijină teorema λ²; se rulează o dată cap-coadă înainte de predare |

```bash
make task2-test     # testele (~1 min)
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

---

## 8. Ce mai e de făcut

| # | Sarcină | Cine | Status |
|---|---|---|---|
| 1 | Protocolul oficial + teste | Claude Code | gata |
| 2 | Baseline + LogReg, 5 seed-uri | Claude Code | gata 14:27 (reproduce 0.58 / 0.70 din enunț) |
| 3 | MOCO prin protocol + ablații linear_cnot / kit_zz | Claude Code | gata, 14:29–14:47 |
| 4 | R2 entangled, 3 variante | Claude Code | gata, 15:01 |
| 5 | Zgomot × shots (diagnostic) | Claude Code | gata, 16:35 (44/45 verificări trec) |
| 6 | CV pe toate cele 492 de fraude | Claude Code | rulează (pornit 16:37) |
| 7 | Grafice pentru slide-uri (AUPRC(p), harta shots, schema circuitului, ledger) | Claude Code | de făcut |
| 8 | ZNE pe R2, cu execuțiile trecute în ledger (opțional) | — | de făcut |
| 9 | Scheletul slide-urilor (secțiunea 9) | Persoana B | de făcut |
| 10 | Commit + push pe branch (`TDWEX`) după fiecare experiment | Claude Code | rulează (primul: f540bc9) |
| 11 | Cifrele finale în slide-uri, repetiție | toți | de făcut până la 07:00 |

**Opriri ferme:** experimente noi până la 05:00 · cifrele blocate la 07:00 · upload până la 08:30.

---

## 9. Structura prezentării (mapată pe cerințe)

1. **Titlu + mesajul principal:** „Zgomotul e o taxă în shots: un VQC de 8 porți care nu o plătește în ordonare”.
2. **Problema:** 0,17% fraudă, de ce AUPRC, baseline-ul (0.58) și referința clasică (0.70).
3. **Pipeline-ul:** split comun, undersampling 1:10 ales pe validare (cu tabelul), encodarea PCA4 -> [0, π].
4. **Circuitul:** schema, 8 porți, 0 CNOT, 8 locuri de zgomot (față de 126), conul de lumină.
5. **Metoda MOCO:** măsurăm o dată, antrenare convexă (Heisenberg), compilăm înapoi.
6. **De ce ajută sub zgomot:** teorema λ², transferul analitic al pragului.
7. **Rezultate, protocolul oficial:** R0 / R2 / R3 / LogReg la toate p, media ± abaterea pe 5 seed-uri.
8. **Răspunsul la hint:** harta AUPRC(p, shots), taxa în shots față de baseline, toate cele 492 de fraude.
9. **Cost ledger:** −99,77% execuții, plus taxa în shots.
10. **Ce n-a mers:** R1 identic cu baseline-ul, entanglement-ul, kit_zz la 0.43, re-uploading 0.554, Pearson top-4, ZNE pe ZZ.
11. **Ce am încerca în continuare:** mai mulți qubiți (costul de robustețe e 0 la circuite produs), zgomot de hardware real (amortizare, erori de citire), Pauli propagation.
12. **Referințe.**

---

## 10. Referințe folosite

[1] Dinuț et al. 2026 (punctul de plecare) · [2] Dal Pozzolo 2015 (undersampling) · [7] Schuld et al. 2019 (parameter shift) · [11] Spall 1992 (SPSA) · [13] McClean 2018 / [14] Grant 2019 (barren plateaus, inițializare) · [15] Wang 2021 (barren plateaus induse de zgomot) · [16] QuantumNAT 2022 (antrenare cu zgomot) · [17] Temme 2017 (ZNE) · [20] Schreiber 2023 (surogate clasice) · [22] Bowles 2024 (benchmarking QML) · Huang, Kueng, Preskill 2020 (classical shadows: învățare din măsurători făcute o dată) · Havlíček 2019 (ZZ feature map).

---

## Changelog

- **24.10, 16:45:** zgomot × shots gata (diagnostic): la p = 0.05 și S = 1024, baseline 0.007 față de MOCO 0.705; comprimarea baseline-ului 0.045, a MOCO 0.871 = λ². 44/45 verificări trec (FAIL: R2 aleator, un seed, 0.0135 > 0.01). H3 (validare = test pentru shots necesare) nu se confirmă. Adăugat `task1/` cu notebook-ul Task 1 completat (20/20 exerciții, 0 erori, split 170883/56962/56962) și README. CV pe toate fraudele pornit la 16:37.
- **24.10, 16:05:** referința clasică corectă e LogReg 1:10 = 0.710 ± 0.007 (aceleași intrări ca MOCO). Ablațiile linear_cnot (0.675 ± 0.034) și kit_zz (0.440 ± 0.104) reproduse local pe 5 seed-uri. R2: warm start 0.667 ± 0.010, față de aleator 0.635 ± 0.018. Ledger cu Tracker: baseline 41/ex (formula 65), MOCO 2, R2 49. Zgomot × shots rescris după specificație, pornit la 15:29 (diagnostic, nefolosit pentru alegeri). Marcajele grafice au fost înlocuite cu cuvinte.
- **24.10, 14:27:** baseline-ul și LogReg pe 5 seed-uri, prin protocolul oficial. Reproduce enunțul (0.580 ± 0.053 / 0.703 ± 0.010). La p = 0.05 baseline-ul cade la 0.405 ± 0.177 (ρ 0.686), iar MOCO rămâne la 0.716 ± 0.007.
- **24.10, 14:25:** document creat. Protocolul și testele sunt gata, R1 e negativ, MOCO are 5 seed-uri, selecția s-a făcut pe validare. Baseline-ul rulează, iar restul experimentelor sunt în coadă.
