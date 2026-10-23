"""Hybrid quantum-classical network in PyTorch (the kit's stack: pennylane + torch).

    x --Linear--> sigmoid*pi --[ quantum layer: re-uploading VQC + noise ]--> <Z_i> --Linear--> logits

`ClassicalTwin` swaps the quantum layer for a classical Tanh layer of the same width, so the
question "does the quantum layer do anything?" has a direct, same-size answer.
`set_noise(model, nm)` rebuilds the quantum layer under another noise model, keeping weights.
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
import torch
from torch import nn

from .metrics import from_logits
from .noise import NoiseModel, readout_noise, single_qubit_noise, two_qubit_noise


def make_qlayer(n_qubits: int, n_layers: int, noise: NoiseModel | None = None):
    noisy = noise is not None and not noise.is_zero()
    dev = qml.device("default.mixed" if noisy else "default.qubit", wires=n_qubits)
    nm = noise if noisy else None
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            for w in wires:
                qml.Rot(*weights[layer, w], wires=w)
            single_qubit_noise(nm, wires)
            if n_qubits > 1:
                for w in wires:
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                two_qubit_noise(nm, wires)
        readout_noise(nm, wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})


class HybridNet(nn.Module):
    def __init__(self, n_in: int, n_qubits: int = 4, n_layers: int = 2, n_classes: int = 2,
                 noise: NoiseModel | None = None):
        super().__init__()
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.pre = nn.Linear(n_in, n_qubits)
        self.q = make_qlayer(n_qubits, n_layers, noise)
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        z = torch.pi * torch.sigmoid(self.pre(x))
        return self.post(self.q(z).to(z.dtype))


class ClassicalTwin(nn.Module):
    """Same pre/post layers, quantum layer replaced by Linear+Tanh of the same width."""

    def __init__(self, n_in: int, n_qubits: int = 4, n_classes: int = 2, **_):
        super().__init__()
        self.pre = nn.Linear(n_in, n_qubits)
        self.mid = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        return self.post(self.mid(torch.sigmoid(self.pre(x))))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def set_noise(model: HybridNet, noise: NoiseModel | None) -> HybridNet:
    state = {k: v.clone() for k, v in model.q.state_dict().items()}
    model.q = make_qlayer(model.n_qubits, model.n_layers, noise)
    model.q.load_state_dict(state)
    return model


def train_torch(model: nn.Module, X, y, epochs: int = 20, lr: float = 0.02, batch_size: int = 32,
                class_weight=None, seed: int = 0, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist, t0 = {"loss": []}, time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist["loss"].append(tot / len(Xt))
        if verbose:
            print(f"ep {ep:3d} loss {hist['loss'][-1]:.4f}")
    hist["seconds"] = round(time.time() - t0, 2)
    return hist


@torch.no_grad()
def evaluate_torch(model: nn.Module, X, y) -> dict:
    logits = model(torch.tensor(np.asarray(X), dtype=torch.float32)).double().numpy()
    return from_logits(y, logits)
