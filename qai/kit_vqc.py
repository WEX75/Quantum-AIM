"""The baseline VQC of the kit (cells 3, 64, 69, 74, 81): ZZ feature map (2 reps, all pairs) followed by an
EfficientSU2-style ansatz (3 entangling layers, 32 parameters), readout p_fraud = (1 - <Z_0>) / 2.
Trained with Adam on BCE, keeping the epoch with the best validation AUPRC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.metrics import average_precision_score

from .protocol import fraud_readout

N_QUBITS = 4
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 3
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_map(x):
    for _ in range(FEATURE_MAP_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[..., i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])


def ansatz(weights):
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


def vqc(x, weights):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


# Backward light cone of Z_0: qubits whose rotations can still reach qubit 0 in each ansatz layer, and the CNOTs
# after each layer that do. Everything else (12 rotations, 3 CNOTs) acts outside the cone.
LIGHTCONE_QUBITS = {0: [0, 1, 2, 3], 1: [0, 1, 2], 2: [0, 1], 3: [0]}
LIGHTCONE_CNOTS = {0: [(0, 1), (1, 2), (2, 3)], 1: [(0, 1), (1, 2)], 2: [(0, 1)]}


def vqc_lightcone(x, weights):
    """The baseline with the gates outside the light cone of the readout removed (R1). Same weights tensor."""
    feature_map(x)
    for layer in range(ANSATZ_REPS + 1):
        for i in LIGHTCONE_QUBITS[layer]:
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for c in LIGHTCONE_CNOTS.get(layer, []):
            qml.CNOT(wires=list(c))
    return qml.expval(qml.PauliZ(0))


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator,
                                                     dtype=torch.float64))


_circuit = qml.QNode(vqc, qml.device("default.qubit", wires=N_QUBITS), interface="torch", diff_method="backprop")


def fraud_probability(x, weights):
    return fraud_readout(_circuit(x, weights))


def predict(Z, weights, model=None, chunk=20000):
    model = model or fraud_probability
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()


def train_vqc(Z_train, y_train, Z_valid, y_valid, seed, model=None, init=None, epochs=EPOCHS,
              lr=LEARNING_RATE, verbose=False):
    """Cell 81."""
    model, init = model or fraud_probability, init or init_weights
    set_seed(seed)
    weights = init(seed)
    optimizer = torch.optim.Adam([weights], lr=lr)
    loss_fn = torch.nn.BCELoss()
    X_t = torch.tensor(Z_train)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    history, best_ap, best_weights = [], -1.0, None
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t))
        losses = []
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx], weights), y_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_ap = average_precision_score(y_valid, predict(Z_valid, weights, model))
        history.append({"epoch": epoch, "train loss": np.mean(losses), "validation AUPRC": val_ap})
        if val_ap > best_ap:
            best_ap, best_weights = val_ap, weights.detach().clone()
        if verbose:
            print(f"epoch {epoch:2d}   train loss {np.mean(losses):.4f}   validation AUPRC {val_ap:.3f}")
    return best_weights, pd.DataFrame(history)
