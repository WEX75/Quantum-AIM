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

## Verificarea notebook-ului (din output-urile salvate, fără să fie rulat)

| Verificare | Rezultat |
|---|---|
| Mesaje „Exercise X: all checks passed” | 20 din 20 (1.1 … 10.3) |
| Output-uri de eroare | 0 |
| `split_indices.npz` raportat de celula 123 | train 170883, val 56962, test 56962 |
| Celule de cod cu număr de execuție | 67 din 70 |
| `FILL_IN` rămase în cod | 0 |

Celulele fără număr de execuție în versiunea salvată:
- 6, graficul pierderii BCE;
- **41, clasa `AngleEncoder`**;
- 43, verificarea 4.3, care are totuși output-ul „passed” salvat.

Celulele următoare care folosesc `AngleEncoder` au rulat, deci clasa a fost definită într-o sesiune anterioară a kernel-ului. Ca versiunea predată să fie curată, **notebook-ul se rulează o dată de sus până jos înainte de predare** (după ce se termină experimentele Task 2, când procesorul e liber).

Lângă notebook nu exista un `split_indices.npz`, așa că verificarea cu `np.array_equal` față de split-ul din `qai/protocol.py` nu s-a putut face. Split-ul din Task 2 e generat cu același cod (celula 35) și e verificat de `tests/test_protocol.py`: 170883 / 56962 / 56962 rânduri, cu 295 / 99 / 98 fraude.

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
