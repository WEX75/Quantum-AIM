# Task 1: VQC-ul ghidat (notebook-ul completat)

`task1_guided_vqc_fraud_detection.ipynb` e notebook-ul ghidat din kit, cu toate exercițiile completate și rulate de echipă. Fișierul din `kit/` a rămas neatins: acolo e versiunea originală, cu spații de completat.

## Ce e Task 1

Etapa de învățare a competiției. Notebook-ul construiește cap-coadă un VQC pentru frauda cu carduri (ULB Credit Card Fraud):
- bazele de ML în PyTorch;
- metricile pentru evenimente rare (AUPRC ca metrică principală);
- split-ul stratificat 60/20/20 cu seed 42, înainte de orice balansare;
- undersampling, apoi encodare: scaler, PCA la 4 componente, unghiuri în [0, π];
- PennyLane;
- VQC-ul baseline: ZZ feature map + ansatz EfficientSU2, 32 de parametri;
- antrenarea, costul pe hardware (execuții, shots, `qml.Tracker`);
- testarea pe 5 seed-uri lângă o regresie logistică;
- zgomotul depolarizant și protocolul de evaluare pentru Task 2.

Doar Task 2 e jurizat. Task 1 e baza lui.

## Verificarea notebook-ului

Notebook-ul a fost rulat o dată de sus până jos (duminică, 25.10, 01:29–01:46), cu `jupyter nbconvert --execute`, pe o copie. Copia a înlocuit fișierul de aici doar după ce a rulat fără erori. Conținutul celulelor nu s-a schimbat (129 de celule, sursa identică). Fișierul din `kit/` a rămas neatins.

| Verificare | Rezultat |
|---|---|
| Celule de cod executate | 70 din 70 |
| Output-uri de eroare | 0 |
| Mesaje „Exercise X: all checks passed” | 20 din 20 (1.1 … 10.3) |
| `split_indices.npz` raportat de celula 123 | train 170883, val 56962, test 56962 |
| `split_indices.npz` din notebook față de split-ul din `qai/protocol.py` (`np.array_equal`) | identic pe train, val și test |

## Cum e refolosit în Task 2

Protocolul, split-ul și baseline-ul din Task 2 sunt copiate din acest notebook, fiecare funcție cu celula ei citată:

| Fișier | Celulele kitului |
|---|---|
| `qai/protocol.py` | 17 și 33 (încărcarea datelor), 35 (split-ul comun), 123 (`split_indices.npz`), 38 (undersampling), 41–42 (`AngleEncoder`), 81 (`predict`), 92 (pragul max-F1 și metricile), 106 (`noisy_qnode`, `noisy_predict`, locurile de zgomot), 125 (`evaluate_under_noise`), 86 și 89 (ledger-ul și `qml.Tracker`) |
| `qai/kit_vqc.py` | 3 (constante), 64 (ZZ feature map), 69 (ansatz), 74 (`vqc`, citirea, inițializarea), 81 (`train_vqc`) |

## Exercițiile pe care se sprijină teorema λ² din MOCO

Teorema din Task 2 (scor_zgomot = ½ − λ²·(½ − scor_ideal), cu λ = 1 − 4p/3, deci ordonarea neschimbată la orice p) combină două fapte arătate în acest notebook:

1. **Exercițiul 3.2:** comprimarea scorurilor spre 0.5 (s -> 0.5 + 0.2·(s − 0.5)) păstrează AUPRC-ul, fiindcă AUPRC depinde doar de ordine. Recall-ul la un prag fix se schimbă însă. De aceea, la MOCO zgomotul nu atinge AUPRC-ul, dar pragul trebuie mutat (`moco.transfer_threshold`).
2. **Exercițiul 10.1:** un RY(θ) urmat de un canal depolarizant de probabilitate p dă ⟨Z⟩ = (1 − 4p/3)·cos θ. Canalul scalează vectorul Bloch cu λ. Pe un circuit fără CNOT, cu 2 porți pe fiecare qubit, asta dă factorul λ² pe fiecare ⟨Z_q⟩.
