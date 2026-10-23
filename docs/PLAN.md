# Planul zilei — BreaQ 2026

Sâmbătă 24 oct 09:00 → Duminică 25 oct 14:00. **La 04:00 ceasul se dă înapoi la 03:00** (o oră în plus).
La final: cod public + prezentare scurtă la juriu, premii pe fiecare track (locurile 1–3).

## Seara de dinainte (azi)
- [ ] Repo pe GitHub, clonat pe laptop, `make install` + `make test` verde
- [ ] `make data` → `data/creditcard.csv` descărcat (284.807 rânduri, 492 fraude), apoi `make check` = READY
- [ ] Notebook-ul kitului în `kit/`, rulată celula de încărcare (cere organizatorii)
- [ ] `make fraud-fast` și `make hybrid-fast` merg pe datele reale
- [ ] Dacă notebook-ul e deja disponibil: `/kickoff` în seara asta (câștigi 1h mâine)
- [ ] Claude Code logat, rulat o dată în repo (`/init` NU e nevoie, CLAUDE.md există)
- [ ] Laptop încărcat, încărcător, prelungitor, buletin
- [ ] Somn. Serios.

## Sâmbătă
| Ora | Ce |
|---|---|
| 09:00–10:30 | Deschidere, se anunță challenge-urile, starter kit pe Discord. **Alegeți track-ul în 15 min.** |
| 10:30–11:00 | Kitul în `kit/`, în Claude Code: `/kickoff` → `docs/KIT.md` + TODO |
| 11:00–13:00 | Baseline-ul kitului rulează prin pipeline-ul nostru. Primul `python3 run.py summary` cu numere reale. |
| 13:00–14:00 | Masă + decizia: care e **o singură idee** pe care o demonstrăm (vezi idei mai jos) |
| 14:00–20:00 | Bucle `/exp`: o idee → test → rezultat → keep/drop → commit. Mentorii: întrebați-i devreme. |
| 20:00–21:00 | Masă. `/clear` în Claude Code, sesiune nouă. |
| 21:00–02:00 | Experimentul principal la scară completă (în fundal) + ablații. Plot-urile finale. |
| 02:00–06:00 | Unul doarme 2–3h, celălalt rulează sweep-uri lungi. Apoi schimbați. |

## Duminică
| Ora | Ce |
|---|---|
| 06:00–10:00 | Freeze pe idei noi. Robustețe: seed-uri multiple (3), bare de eroare. |
| 10:00–12:00 | `/ship`: README, plot-uri, PITCH.md, slide-uri |
| 12:00–13:00 | Repetiție pitch de 2 ori, cronometrat |
| 13:00–14:00 | Push final, buffer pentru probleme |

## Idei Quantum AI (pentru o pipeline QML afectată de zgomot)
1. **Noise-aware training** vs clean training, evaluat pe zgomot crescător (avem deja: la p=0.1 pe moons, 96.7% vs 51%)
2. **ZNE** (noise-scaling / unitary folding) — recuperează mult la zgomot mic, puțin la zgomot mare
3. **Arhitectură**: data re-uploading vs encoding o singură dată; adâncime vs zgomot (sweet spot de layere)
4. **Shot noise**: `qml.set_shots` — câte shots îți trebuie pentru o acuratețe dată
5. **Parametri**: VQC 40 params vs MLP 49 vs SVM — argumentul de eficiență
6. Noise model din hardware real (T1/T2, erori de gate) dacă kitul dă calibrări

## Idei Quantum Hacking
1. Intercept-resend parțial: QBER = f/4, prag 11%, câte bucăți publice trebuie sacrificate pentru detecție
2. Breidbart: mai multă informație pentru Eve la aceeași perturbare
3. **Trojan horse**: invizibil în QBER, apărare prin izolator optic (câți dB ajung)
4. Atacuri pe device (detector blinding, time-shift) — modelate statistic
5. Adversarial attacks pe modele QML (punte între cele două track-uri!)

## Ce iubește juriul
- Un singur mesaj clar, susținut de un grafic
- Comparație corectă cu baseline clasic (cu număr de parametri)
- Limitări spuse onest (simulator, model de zgomot)
- Repo reproductibil: o comandă → graficul
