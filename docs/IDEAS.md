# Quantum AI — teme, 30 de idei, cum o alegem pe cea mai bună

Track-ul 2026: *„explore a quantum AI pipeline and how it must adapt to realistic, noisy conditions”* (PennyLane).
Anul trecut (Entangled Learning) juriul a cerut: baseline clasic solid, ≥1 model pur cuantic + ≥1 hibrid,
encoding justificat, analiză onestă (acuratețe, cost de antrenare, sensibilitate la zgomot, rolul entanglementului),
cod rulabil + write-up. Presupunem același tip de evaluare, cu accent pe zgomot.

## Temele principale (cu ce se mănâncă)

| Temă | Ce înseamnă, pe scurt | Unde e în repo / PennyLane |
|---|---|---|
| **Encoding** | Cum bagi date clasice în qubiți: unghiuri (RY), IQP, amplitude, data re-uploading (re-encodezi la fiecare strat) | `AngleEmbedding`, `IQPEmbedding`, `AmplitudeEmbedding`; `qai/model.py` |
| **Circuite variaționale (VQC)** | Circuit cu rotații antrenabile + CNOT-uri; optimizat clasic (Adam) pe gradienți. Rețeaua neuronală „cuantică” | `qai/model.py`, `StronglyEntanglingLayers` |
| **Modele hibride** | Bucăți clasice (encoder/head) + un strat cuantic la mijloc | `qml.qnn`, de adăugat |
| **Quantum kernels** | Circuitul doar mapează datele, SVM-ul clasic face restul; kernel = suprapunerea stărilor | `qml.kernels` |
| **Zgomot NISQ** | Depolarizare (gate-uri imperfecte), T1/amplitude damping (qubitul „cade” în 0), T2/dephasing, readout (citire greșită), shot noise (măsurători finite) | `qai/noise.py`, `qml.noise.NoiseModel`, `qml.set_shots` |
| **Trainability** | Barren plateaus: gradienții dispar exponențial cu nr. de qubiți/adâncimea; zgomotul îi face și mai mici | de măsurat varianța gradientului |
| **Error mitigation** | Nu corectezi erorile, le compensezi statistic: ZNE (rulezi la zgomot ×1,×2,×3 și extrapolezi la 0), readout mitigation | `qai/mitigation.py`, `qml.noise.mitigate_with_zne`, `fold_global` |
| **Noise-aware training** | Antrenezi direct sub zgomot ca modelul să învețe să-l compenseze | `qai/bench.py` (avem deja rezultatul: 96.7% vs 51% la p=0.1) |
| **Onestitate / avantaj** | Majoritatea „avantajelor cuantice” dispar la un baseline clasic bun. Juriul știe asta | `qai/baseline.py` + nr. de parametri |

## 30 de idei
Efort: **S** ≤2h · **M** 3–5h · **L** 6h+ (cu repo-ul actual). ★ = valoare mare pentru pitch.

**A. Zgomot și robustețe (miezul track-ului)**
1. ★ Noise-aware vs clean training, curbă acuratețe vs zgomot (S, avem deja)
2. Curriculum de zgomot: p crește treptat în antrenare → robust pe tot intervalul (S)
3. Zgomotul ca regularizare: ajută la generalizare când ai puține date? (M)
4. ★ Harta adâncime × zgomot: unde e sweet spot-ul de straturi (heatmap) (S)
5. Ablație pe tipul de zgomot: depolarizare vs T1 vs T2 vs readout — care doare cel mai mult (S)
6. Shot budget: câte shots pentru 95% din acuratețea ideală (S)
7. ★ Noise-induced barren plateaus: varianța gradientului vs zgomot și adâncime (M)
8. Model de zgomot din calibrări publice ale unui device real (T1/T2, erori de gate) (M)
9. Drift: antrenezi la p, testezi la alt p (calibrare veche); noise-aware rezistă? (S)

**B. Error mitigation**
10. ZNE oficial PennyLane (`fold_global` + Richardson vs exponential) vs varianta noastră (M)
11. Readout mitigation cu matrice de confuzie inversată — ieftin, câștig mare (S)
12. Mitigation în bucla de antrenare vs doar la inferență (L)
13. Cost-beneficiu: acuratețe câștigată per shot suplimentar, ZNE vs mai multe shots (M)
14. ★ „Learned mitigation”: un MLP mic corectează ieșirile zgomotoase ale circuitului (M)

**C. Arhitectură și encoding**
15. Encoding-uri sub zgomot: angle vs IQP vs amplitude vs re-uploading (M)
16. ★ Rolul entanglementului: ring vs line vs fără CNOT — ajută sau doar adaugă zgomot? (S; cerut explicit anul trecut)
17. Hardware-efficient ansatz vs StronglyEntanglingLayers: nr. CNOT vs acuratețe (S)
18. ★ Hibrid: encoder clasic → strat cuantic → head clasic vs pur cuantic vs pur clasic (M; cerut anul trecut)
19. Pruning de gate-uri: scoți ce contează puțin → circuit mai scurt, mai puțin zgomot (M)
20. Strategie de măsurare: un ⟨Z⟩ vs toți qubiții + head clasic (S)

**D. Quantum kernels**
21. Kernel SVM sub zgomot: concentrarea exponențială a kernelului (M)
22. Kernel alignment antrenat sub zgomot (L)
23. Projected kernel vs fidelity kernel — care e mai robust (L)

**E. Eficiență și onestitate (ce iubește juriul)**
24. ★ Același număr de parametri: VQC vs MLP, curbe de învățare (S)
25. Date puține: 10/20/50 exemple de antrenare — cine generalizează mai bine (S)
26. Dequantization check: random features clasice care imită VQC-ul (L)

**F. Securitate (punte spre Quantum Hacking)**
27. ★ Atacuri adversariale (FGSM pe input) pe VQC: zgomotul crește sau scade robustețea? (M)
28. Zgomotul hardware ca „randomized smoothing” gratuit — robustețe certificată (M)
29. Backdoor / data poisoning într-un VQC (M)

**G. Demo**
30. ★ Dashboard live: glisezi zgomotul și vezi acuratețea + decision boundary în timp real (S–M; excelent pe scenă)

## Cum alegem cea mai bună idee (10:30–13:00)

**Pas 1 — după kit, 15 min.** Rulezi `/kickoff`, apoi dai prompt-ul de mai jos lui Fable, în chat, cu `docs/KIT.md` + lista asta.

**Pas 2 — scor pe fiecare idee (1–5):**

| Criteriu | Pondere | Întrebarea |
|---|---|---|
| Fit cu challenge-ul | ×3 | Răspunde direct la ce cere kitul / metrica lor? |
| Poveste | ×3 | Pot spune rezultatul într-o propoziție + un grafic? |
| Fezabil până la 20:00 | ×2 | Se face cu repo-ul actual în ≤6h? |
| Rigoare | ×2 | Are baseline, seed-uri, explicație fizică? |
| Originalitate | ×1 | O face și altă echipă? |

**Pas 3 — combinația câștigătoare: 1 idee principală + 2 analize ieftine.**
Ideea principală răspunde challenge-ului; analizele (S) arată profunzime.
Combinație implicită dacă nu ai timp de gândit: **1 + 16 + 24** (+ **30** ca demo, + **27** dacă rămâne timp pentru „wow”).

**Pas 4 — kill test la 13:00.** Rularea fast trebuie să arate un semnal (o diferență clară pe grafic).
Dacă nu: treci pe ideea de rezervă, fără regrete. Nu salva o idee moartă la 3 noaptea.

## Prompt pentru Fable (chat)
```
Sunt la BreaQ 2026, track Quantum AI, echipă de 2, 24h, PennyLane, doar simulator.
Mai jos: challenge-ul (KIT.md), ce avem deja implementat (CLAUDE.md) și 30 de idei (IDEAS.md).
1. Spune-mi în 3 propoziții ce vrea de fapt juriul de la challenge-ul ăsta.
2. Notează fiecare idee pe criteriile din tabel și dă-mi top 5 cu scorul.
3. Pentru top 1: ipoteza, experimentul minim, graficul care o demonstrează, riscul principal
   și ideea de rezervă.
4. Arhitectura în maxim o pagină: ce module din repo modificăm, ce adăugăm, în ce ordine.
Fii critic: dacă o idee pare impresionantă dar nu dovedește nimic, spune.
```
