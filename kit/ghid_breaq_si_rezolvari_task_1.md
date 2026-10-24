# BreaQ 2026: Fraud on a Quantum Budget — War Room Guide

Acest document conține rezumatul tehnic al temei, vulnerabilitățile din Baseline pe care le exploatăm în **Task 2**, checklist-ul de predare și rezolvările complete pentru toate exercițiile `FILL_IN` din **Task 1**.

---

## 1. Miza Problemei și Setul de Date

* **Obiectiv:** Construirea și optimizarea unui **Variational Quantum Classifier (VQC)** în **PennyLane** și **PyTorch** care detectează tranzacții frauduloase sub **zgomot cuantic depolarizant simulat**, minimizând totodată **bugetul de execuții pe hardware** (*circuit executions* și *shots*).
* **Dataset (ULB Credit Card Fraud):**
  * **284.807 tranzacții** totale, **30 de trăsături** (`V1`–`V28` rezultate din PCA, plus `Time` și `Amount` brute).
  * Doar **492 sunt fraude ($0.17\%$)**.
  * Sursa: `data/creditcard.csv` (Kaggle) sau descărcare automată din OpenML (`data_id=42175`). **Nu** folosiți OpenML `1597` (nu are coloana `Time`).
* **De ce Accuracy este inutilă:** Un model care prezice mereu $0$ (legitim) are o acuratețe de $99.83\%$, dar $0$ fraude prinse.
* **Metrica principală:** **AUPRC** (*Area Under Precision-Recall Curve* / `average_precision_score`), în special la nivelurile de zgomot $p = 0.01$ și $p = 0.05$, alături de **AUROC**, **Spearman $\rho$** (între scorurile cu zgomot și fără zgomot), **Precision**, **Recall**, **F1** și **MCC**.
* **Common Split Obligatoriu (`split_indices.npz`):**
  * Împărțire stratificată **$60\%$ train / $20\%$ validation / $20\%$ test** cu `SEED = 42` și `SAMPLE_FRACTION = 1.0`, făcută *înainte* de orice balansare sau preprocesare.
  * **Fără Data Leakage:** Orice balansare, scalare sau PCA se face `.fit()` **strict pe setul de train**. Pragul de decizie (*threshold*) și epoca optimă se aleg **strict pe validation**. Setul de **test** ($56.962$ tranzacții, $98$ fraude) se folosește doar la final.

---

## 2. Analiza Baseline-ului (Task 1) și Vulnerabilitățile de Atac în Task 2

### Cum funcționează Baseline-ul:
1. **Balansare & Preprocesare (`AngleEncoder`):**
   * *Random undersampling* $1:1$ pe train: păstrează cele $295$ fraude + $295$ tranzacții legitime alese aleator ($590$ rânduri în total).
   * Aplică `StandardScaler` pe `Time` și `Amount`, apoi `PCA(n_components=4)` peste toate cele $30$ de trăsături, urmat de `MinMaxScaler(feature_range=(0, np.pi))` cu clip în $[0, \pi]$.
2. **Arhitectura VQC ($4$ qubiți, $93$ porți native, adâncime $45$, $126$ locații de zgomot):**
   * **Feature Map:** `ZZFeatureMap` cu $2$ repetiții (`FEATURE_MAP_REPS = 2`) și entanglement *full* (toate cele $6$ perechi de qubiți conectate prin blocuri `CNOT - RZ - CNOT`).
   * **Ansatz:** Stil `EfficientSU2` cu $3$ straturi (`ANSATZ_REPS = 3`) de rotații `RY` + `RZ` și CNOT-uri liniare ($0 \to 1, 1 \to 2, 2 \to 3$), plus un strat final de rotații $\implies (3+1) \times 4 \times 2 = 32$ parametri antrenabili.
   * **Readout:** Măsoară $\langle Z_0 \rangle$ pe qubitul $0$ și calculează probabilitatea de fraudă ca $p_{\text{fraud}} = \frac{1 - \langle Z_0 \rangle}{2}$.
3. **Performanță de referință (fără zgomot, pe $5$ seed-uri):**
   * Baseline VQC test AUPRC: $0.58 \pm 0.05$.
   * Regresie Logistică clasică (pe aceleași $4$ unghiuri): $0.70 \pm 0.01$.

### Cele 3 Vulnerabilități Majore ale Baseline-ului (Exploatabile în Task 2):
1. **Porți și parametri în afara conului de lumină (*Light-Cone Waste*):**
   * Din cei $32$ de parametri ai ansatz-ului, **$12$ parametri nu pot influența niciodată qubitul $0$** (se află după ultimele porți CNOT care mai conectează acei qubiți la qubitul $0$).
   * Al **$13$-lea parametru** (ultimul `RZ` de pe qubitul $0$) comută cu măsurătoarea `PauliZ(0)`, deci are gradientul mereu $0$.
   * Doar **$19$ din $32$ parametri** contează efectiv, dar porțile inutile adaugă canale de zgomot depolarizant!
2. **Exces de porți CNOT în Feature Map:**
   * `ZZFeatureMap` cu `FEATURE_MAP_REPS = 2` și entanglement *full* folosește $2 \times 6 \times 2 = 24$ porți CNOT doar pentru încărcarea datelor. Cum fiecare CNOT adaugă **două** canale de zgomot depolarizant, circuitul acumulează $126$ locații de zgomot, iar la $p = 0.05$ ordonarea tranzacțiilor se degradează masiv ($\rho \approx 0.65$).
3. **Risipa uriașă de buget la validare și gradienți:**
   * Cu *parameter-shift rule* ($1 + 2P = 65$ execuții/exemplu teoretic, sau $41$ cu optimizarea din `qml.Tracker`), antrenarea pe $30$ de epoci $\times 590$ exemple costă **$1.150.500$ execuții** ($1.17$ miliarde *shots* la $1024$ shots/execuție).
   * Evaluarea întregului set de validare ($56.961$ tranzacții) după *fiecare* din cele $30$ de epoci costă încă **$1.708.860$ execuții** — mai mult decât antrenarea!

---

## 3. Reguli de Evaluare și Checklist Livrabile (Task 2)

### Protocolul Oficial (`evaluate_under_noise`):
* **Setul de porți native:** `NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}`.
* **Zgomot:** După descompunerea în porți native, se aplică `qml.DepolarizingChannel(p)` după fiecare poartă, pe fiecare qubit pe care acționează, pentru $p \in \{0.0, 0.001, 0.01, 0.05\}$ pe simulatorul `default.mixed`.
* **Prag (Threshold):** Ales pe scorurile de validare evaluate **sub același nivel de zgomot $p$** astfel încât să maximizeze F1. *(Nu se punctează ca îmbunătățire proprie, fiind deja în protocol).*
* **Seed-uri:** Toate tabelele trebuie raportate ca **medie $\pm$ deviație standard pe minimum $5$ seed-uri** (`SEEDS = [0, 1, 2, 3, 4]`).

### Checklist Livrabile (în cele 24h):
- [ ] **1. Codul pentru Task 2 (`.ipynb` sau `.py`):** Rulabil cap-coadă cu seed-uri fixe, pornind de la `split_indices.npz` până la tabelele finale.
- [ ] **2. Prezentarea (`.pdf` sau `.pptx`) încărcată în timp util:**
  - [ ] Pipeline-ul end-to-end (preprocesare, balansare, encodare);
  - [ ] Designul circuitului cuantic, motivația, numărul de porți și `noise_locations`;
  - [ ] Îmbunătățirile propuse și explicația fizică/matematică pentru care ajută sub zgomot și buget;
  - [ ] Tabelul de rezultate (Baseline VQC vs. Metoda Voastră vs. Referința Clasică pe aceleași intrări) la $p \in \{0.0, 0.001, 0.01, 0.05\}$ ca $\text{mean} \pm \text{std}$ pe $5$ seed-uri;
  - [ ] **Cost Ledger:** Tabel comparativ cu execuțiile de circuit și *shots* pentru antrenare și validare (Baseline vs. Metoda Voastră);
  - [ ] Secțiune dedicată: **„What did not work, and what we would try next”** (ablations și rezultate negative oneste).

---

## 4. Plan de Atac Recomandat pentru Task 2

1. **Arhitectură Light-Cone / Hardware-Efficient (Reducerea `noise_locations` de la $126$ la $\sim 25\text{--}35$):**
   * Înlocuiți `ZZFeatureMap` (2 reps, full entanglement) cu `qml.AngleEmbedding` sau un singur strat de entanglement liniar / **Data Re-uploading** ($x$ injectat în straturile ansatz-ului împreună cu greutățile antrenabile).
   * Eliminați toate rotațiile care nu se află în conul de lumină al qubitului măsurat (sau folosiți un circuit de tip *tree tensor network* / *reverse light-cone* care converge direct spre qubitul $0$).
   * Eliminați porțile `RZ` finale înainte de măsurătoarea în baza $Z$.
2. **Optimizare de Buget (Reducerea execuțiilor cu $>85\%$ în Cost Ledger):**
   * **Validare inteligentă:** Evaluați pe un subset stratificat de $10\%\text{--}20\%$ din setul de validare și doar din $3$ în $3$ epoci (sau cu *early stopping*), reducând costul validării de la $1.7\text{M}$ execuții la $<100\text{k}$.
   * **Antrenare cu SPSA sau mai puțini parametri:** SPSA folosește doar $2$ execuții per pas indiferent de numărul de parametri (față de $2P$ la parameter-shift), sau combinați parameter-shift pe un circuit redus la $12\text{--}16$ parametri activi.
3. **Antrenare și Date (Creșterea AUPRC la $p = 0.01$ și $p = 0.05$):**
   * **Initializare aproape de zero/identitate:** Inițializați greutățile cu valori mici $\mathcal{N}(0, 0.01)$ în loc de $[0, 2\pi)$ pentru a evita *barren plateaus* și a stabiliza varianța între seed-uri.
   * **Noise-Aware Fine-Tuning:** Antrenați rapid fără zgomot primele $15\text{--}20$ de epoci și faceți *fine-tuning* $5$ epoci direct pe `noisy_qnode` la $p = 0.01$.
   * **Undersampling $1:3$ sau $1:4$ cu Class Weights / Ensemble peste mai multe trageri de undersampling:** Păstrează mai multă informație despre tranzacțiile legitime decât raportul $1:1$.

---

## 5. Rezolvările Complete pentru `FILL_IN` din Task 1 (`task1_guided_vqc_fraud_detection.ipynb`)

Copiază direct liniile de mai jos în celulele cu ✏️ din notebook-ul de Task 1 pentru a trece toate verificările `assert` și a salva `split_indices.npz`.

### Exercise 1.1 — Binary Cross-Entropy
```python
def binary_cross_entropy(p, y):
    '''Mean binary cross-entropy between predicted probabilities p and 0/1 labels y.'''
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
```

### Exercise 1.2 — Gradient Descent by Hand
```python
    with torch.no_grad():        # the update itself must not be recorded by autograd
        theta -= lr * theta.grad # one gradient-descent step: learning rate times gradient
```

### Exercise 2.1 — How rare is fraud?
```python
n_fraud = int((df["Class"] == 1).sum())
n_legit = int((df["Class"] == 0).sum())
fraud_rate = n_fraud / len(df)
always_legit_accuracy = n_legit / len(df)
```

### Exercise 3.1 — Confusion Matrix, Precision, Recall, F1
```python
def confusion_counts(y_true, y_pred):
    '''(TP, FP, FN, TN) for arrays of 0/1 labels and 0/1 predictions.'''
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

### Exercise 3.2 — Only the order matters
```python
scores = demo_scores["decent"]
squashed = 0.5 + 0.2 * (scores - 0.5)
```

### Exercise 4.1 — Split into train, validation and test
```python
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
```

### Exercise 4.2 — Balance the training set (Undersampling)
```python
def undersample(X_part, y_part, seed):
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]
```

### Exercise 4.3 — AngleEncoder
```python
    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        self.angles.fit(self.pca.fit_transform(self._scaled(frame)))
        return self

    def transform(self, frame):
        return np.clip(self.angles.transform(self.pca.transform(self._scaled(frame))), 0, np.pi)
```

### Exercise 5.1 — One qubit QNode
```python
@qml.qnode(dev1)
def ry_circuit(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))
```

### Exercise 5.2 — Two qubits and entanglement (Bell state)
```python
@qml.qnode(dev2)
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])
```

### Exercise 5.3 — The parameter-shift rule
```python
def parameter_shift(f, theta):
    '''Derivative of f at theta from two shifted evaluations of f.'''
    return (f(theta + np.pi / 2) - f(theta - np.pi / 2)) / 2
```

### Exercise 6.1 — The ZZ Feature Map
```python
def feature_map(x):
    '''ZZ feature map. x holds N_QUBITS angles in [0, pi], or a batch of them (shape (..., N_QUBITS)).'''
    for _ in range(FEATURE_MAP_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[..., i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])
```

### Exercise 6.2 — The EfficientSU2-style Ansatz
```python
def ansatz(weights):
    '''EfficientSU2-style ansatz; weights has shape (ANSATZ_REPS + 1, N_QUBITS, 2).'''
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
```

### Exercise 6.3 — Readout and the full circuit
```python
def vqc(x, weights):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


device = qml.device("default.qubit", wires=N_QUBITS)
circuit = qml.QNode(vqc, device, interface="torch", diff_method="backprop")


def fraud_probability(x, weights):
    return (1 - circuit(x, weights)) / 2
```

### Exercise 7.1 — The Training Loop step
```python
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx], weights), y_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
```

### Exercise 8.1 — Hardware Cost Ledger
```python
n_params = weights.numel()
executions_per_sample = 1 + 2 * n_params
```

### Exercise 9.1 — Choose the threshold on validation
```python
def choose_threshold(y_true, scores):
    '''The threshold that maximises F1 on (y_true, scores).'''
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]
```

### Exercise 10.1 — Noisy RY with Depolarizing Channel
```python
@qml.qnode(mixed1)
def noisy_ry(theta, p):
    qml.RY(theta, wires=0)
    qml.DepolarizingChannel(p, wires=0)
    return qml.expval(qml.PauliZ(0))
```

### Exercise 10.2 — Spearman rank correlation under noise
```python
for p in NOISE_LEVELS:
    start = time.time()
    noisy_test[p] = noisy_predict(Z_test, weights, p)
    rho = spearmanr(vqc_test_scores, noisy_test[p]).statistic
    noise_rows.append({"p": p, **evaluate(y_test, noisy_test[p], vqc_threshold), "Spearman rho": rho})
```

### Exercise 10.3 — Recalibrating the threshold on noisy validation scores
```python
for p in NOISE_LEVELS:
    noisy_val[p] = noisy_predict(Z_val, weights, p)
    threshold_p = choose_threshold(y_val, noisy_val[p])
    recalibrated_rows.append({"p": p, "threshold": threshold_p, **evaluate(y_test, noisy_test[p], threshold_p)})
```