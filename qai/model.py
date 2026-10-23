"""Variational quantum classifier (VQC) with optional noise.

Architecture: [encode -> Rot layer -> noise -> ring of CNOTs -> noise] x n_layers,
then readout noise and <Z_i> on the first n_classes wires. Logits = scale * <Z> + bias.
With encoding="reupload" the data is re-encoded before every layer (data re-uploading),
which usually helps a lot on small qubit counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .noise import NoiseModel, readout_noise, single_qubit_noise, two_qubit_noise


@dataclass
class VQCConfig:
    n_qubits: int = 4
    n_layers: int = 3
    n_classes: int = 2
    encoding: str = "reupload"  # "angle" | "reupload"
    entangler: str = "ring"     # "ring" | "line" | "none"

    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * 3 + 2 * self.n_classes


def _entangle(cfg: VQCConfig):
    n = cfg.n_qubits
    if cfg.entangler == "none" or n < 2:
        return []
    pairs = [(i, i + 1) for i in range(n - 1)]
    if cfg.entangler == "ring" and n > 2:
        pairs.append((n - 1, 0))
    return pairs


def build_circuit(cfg: VQCConfig, noise: NoiseModel | None = None):
    noisy = noise is not None and not noise.is_zero()
    dev = qml.device("default.mixed" if noisy else "default.qubit", wires=cfg.n_qubits)
    nm = noise if noisy else None
    wires = list(range(cfg.n_qubits))
    pairs = _entangle(cfg)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(x, weights):
        for layer in range(cfg.n_layers):
            if layer == 0 or cfg.encoding == "reupload":
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
            for w in wires:
                qml.Rot(*weights[layer, w], wires=w)
            single_qubit_noise(nm, wires)
            for a, b in pairs:
                qml.CNOT(wires=[a, b])
            if pairs:
                two_qubit_noise(nm, wires)
        readout_noise(nm, wires[: cfg.n_classes])
        return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_classes)]

    return circuit


def init_params(cfg: VQCConfig, seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.normal(0, 0.3, (cfg.n_layers, cfg.n_qubits, 3)), requires_grad=True)
    scale = pnp.array(np.full(cfg.n_classes, 3.0), requires_grad=True)
    bias = pnp.array(np.zeros(cfg.n_classes), requires_grad=True)
    return [weights, scale, bias]


@dataclass
class VQC:
    cfg: VQCConfig
    noise: NoiseModel | None = None
    params: list = field(default=None)
    seed: int = 0

    def __post_init__(self):
        self.circuit = build_circuit(self.cfg, self.noise)
        if self.params is None:
            self.params = init_params(self.cfg, self.seed)

    def expvals(self, X, params=None, circuit=None):
        weights, _, _ = params or self.params
        out = (circuit or self.circuit)(X, weights)
        out = pnp.stack(out)  # (n_classes, batch)
        return out.T if out.ndim == 2 else out[None, :]

    def logits(self, X, params=None, circuit=None):
        p = params or self.params
        return self.expvals(X, p, circuit) * p[1] + p[2]

    def predict(self, X, noise: NoiseModel | None = "same"):
        """Predict with the training noise ("same") or under a different noise model."""
        circ = None if noise == "same" else build_circuit(self.cfg, noise)
        return np.argmax(np.asarray(self.logits(X, circuit=circ)), axis=1)

    def accuracy(self, X, y, noise: NoiseModel | None = "same") -> float:
        return float(np.mean(self.predict(X, noise) == y))
