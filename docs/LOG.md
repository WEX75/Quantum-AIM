# Experiment log

`HH:MM | idee | rezultat | keep/drop`

- pre | VQC 4q×3L reupload, moons, p_train=0.05 | clean 1.00→0.51 la p=0.1; noise-aware 0.967 constant; ZNE 0.62 la p=0.1, 0.978 la p=0.05; SVM 0.967 | keep
- pre | BB84 sweeps | IR QBER≈f/4+zgomot; Trojan horse QBER neschimbat, cheie 0 sub ~25 dB izolare | keep
- 13:30 | MOCO single 1:10 (moco_task2.py), protocol pe split comun, 5 seeds | AUPRC 0.716±0.007 la orice p, ρ=1, 8 porți, 6.490 exec | keep (mentor: direcția e bună)
- 14:00 | selecție pe validare (fm × ratio, 5 seeds, test neatins) | single 1:10 val 0.675±0.011 = maxim; pair 1:30 egal dar 4x mai scump | keep 1:10
- 14:10 | qai/protocol.py = kitul exact; tests/test_protocol.py | 93 porți/33 CNOT/126 loc./depth 45, ledger 1.150.500/1.708.860, Tracker 41/ex, split 295/99/98 | keep
- 14:10 | R1 light-cone: baseline fără 12 rotații + 3 CNOT moarte | scoruri identice la 1e-10 și la p=0.05; Tracker deja 41 (PennyLane sare param. morți) → câștig 0 | drop (negativ onest)
- 14:27 | R0 baseline kit + LogReg 1:1, protocol oficial, 5 seeds | VQC 0.580±0.053 → 0.564 (p=.01) → 0.405±0.177 (p=.05, ρ 0.686); LogReg 0.703±0.010; reproduce enunțul | keep (referință)
- 14:10 | teorema MOCO verificată cu noisy_qnode din kit | s_p = ½ − λ²(½ − s_0) exact până la p=0.3; prag transferabil analitic | keep
