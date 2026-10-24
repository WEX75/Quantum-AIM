# Diagnostice (25.10, ~02:10)

Doar fișiere noi. Fără simulări noi de circuit, seed 0, iar testul e folosit doar pentru raportare. `docs/PROJECT.md` nu a fost modificat.

## (a) Pe ce rânduri e antrenat `AngleEncoder`

Nici baseline-ul, nici MOCO nu antrenează encoder-ul pe tot setul de antrenare (170.883 de rânduri). Amândouă îl antrenează pe subsetul subeșantionat, ca în kit (celula 42 face `fit(X_bal)`).

| Model | Rânduri pe care se antrenează encoder-ul | Unde în cod |
|---|---|---|
| Baseline-ul kitului | subsetul 1:1: toate cele 295 de fraude de antrenare + 295 de tranzacții legitime = 590 de rânduri | `qai/experiments/exp_task2_baseline.py:32` (`undersample(..., seed)`) și `:33` (`AngleEncoder(...).fit(X_bal)`) |
| MOCO | subsetul 1:10: 295 de fraude + 2.950 de tranzacții legitime = 3.245 de rânduri | `qai/experiments/exp_moco.py:32` (`undersample(..., seed, neg_per_pos)`, cu `NEG_PER_POS = 10`) și `:33` (`.fit(X_b)`) |

Aceeași rețetă 1:10 apare în `exp_noise_shots.py:128-129`, `exp_entangled.py:105-107`, `exp_allfrauds_cv.py:62-63` și `exp_adaptive_shots.py:144, 161`. Funcția `undersample` e în `qai/protocol.py:69` (celula 38 din kit), iar `AngleEncoder.fit` în `qai/protocol.py:92`.

## (b) Corelații, PCA și un exemplu cap-coadă (`qai/experiments/exp_covariance_e2e.py`)

### Corelațiile dintre V1..V28 (`results/task2/corr_stats.csv`, figura `results/task2/figures/m_corr_heatmap.png`)

| Set | Rânduri | Fraude | Media |r| în afara diagonalei | |r| maxim | Perechi cu |r| > 0.5 |
|---|---|---|---|---|---|
| Tot setul de antrenare | 170.883 | 295 | 0.004 | 0.039 (V5–V7) | 0 |
| Setul MOCO 1:10 (seed 0) | 3.245 | 295 | 0.207 | 0.844 (V16–V17) | 75 |

V1..V28 sunt componentele PCA publicate de proprietarii datelor, deci pe tot dataset-ul sunt practic necorelate. Pe setul de antrenare bogat în fraude însă, fraudele se deplasează împreună pe multe componente (V1, V3, V7, V10, V12, V14, V16, V17), iar aceste componente devin puternic corelate. De aceea, PCA-ul din encoder găsește în primul rând direcția fraudei.

### PCA-ul din encoder-ul MOCO (seed 0)

| Componentă | Varianță explicată | Primele 5 încărcări (loadings) |
|---|---|---|
| PC1 | 0.5575 | V3 +0.404, V7 +0.361, V17 +0.343, V1 +0.321, V10 +0.295 |
| PC2 | 0.0859 | V8 +0.535, V1 +0.409, V14 −0.370, V7 +0.322, V12 −0.256 |
| PC3 | 0.0595 | V8 +0.738, V1 −0.405, V5 −0.252, V14 +0.238, V12 +0.176 |
| PC4 | 0.0350 | V3 +0.569, V17 −0.432, V2 −0.334, V5 −0.236, V16 −0.225 |

Cele 4 componente explică împreună 73,8% din varianță. PC1, singură, explică 55,8%.

### Exemplul cap-coadă (`results/task2/e2e_example.md`)

Am ales două tranzacții după o regulă fixă: pentru fiecare clasă, tranzacția din validare al cărei scor MOCO fără zgomot e mediana clasei.

| | Frauda (rândul 33519 din validare) | Tranzacția legitimă (rândul 47146 din validare) |
|---|---|---|
| ⟨H⟩ = Σ w_q ⟨Z_q⟩ | −0.0177 | 0.2587 |
| Scorul la p = 0 | 0.5089 | 0.3706 |
| Verdictul la p = 0 (pragul t₀ = 0.4510) | FRAUDĂ | legitimă |
| Scorul la p = 0.05 (λ² = 0.8711) | 0.5077 | 0.3873 |
| Verdictul la p = 0.05 (pragul transferat t_p = 0.4573) | FRAUDĂ | legitimă |

Verificări:
- Scorurile recalculate pas cu pas coincid cu cele din protocol (modelul de zgomot al kitului) pe toate rândurile de validare: diferența maximă e 2.8·10⁻¹⁶.
- Pragul transferat analitic la p = 0.05 (0.4573) coincide cu pragul ales de protocol pe validarea zgomotoasă: diferența e 1·10⁻¹⁴.

## (c) Precizia la un recall fix (`qai/experiments/exp_precision_at_recall.py`, `results/task2/precision_at_recall.csv`, figura `results/task2/figures/n_pr_curve.png`)

Modelele MOCO oficiale (single 1:10, seed-urile 0..4), pe test. Ordonarea MOCO e aceeași la orice p, deci cifrele sunt valabile la orice nivel de zgomot din protocol.

| Recall cel puțin | Precizia maximă pe test (media ± abaterea, 5 seed-uri) |
|---|---|
| 0.5 | 0.833 ± 0.008 |
| 0.6 | 0.833 ± 0.008 |
| 0.7 | 0.829 ± 0.007 |
| 0.8 | 0.817 ± 0.009 |
| 0.9 | 0.072 ± 0.032 |

Punctul de funcționare cu pragul protocolului (F1 maxim pe validare):
- precizie 0.781 ± 0.048 și recall 0.822 ± 0.014;
- din 98 de fraude din test sunt prinse în medie 80.6, cu 23.0 alarme false, adică 103.6 tranzacții marcate din 56.962.

Ce arată:
- **Până la un recall de ~0.8, precizia rămâne peste 0.8.** Peste ~0.84, curba se prăbușește: la un recall de 0.9 precizia scade la 0.07.
- **Explicația:** cam 15% dintre fraude au scoruri ca tranzacțiile legitime, pe cele 4 componente PCA. Ca să le prinzi, trebuie să marchezi mii de tranzacții legitime.
- **Plafonul vine din informația din cele 4 componente, nu din circuit.** Regresia logistică pe aceleași intrări are aceeași AUPRC.
