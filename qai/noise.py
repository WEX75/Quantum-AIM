"""Noise models for near-term hardware, applied as channels inside the circuit.

All probabilities are per gate layer. `scaled(k)` multiplies every probability by k,
which is what the noise-scaling extrapolation in `mitigation.py` relies on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import pennylane as qml


@dataclass(frozen=True)
class NoiseModel:
    depolarizing: float = 0.0       # after every single-qubit layer
    two_qubit_depol: float = 0.0    # after every entangling layer (CNOTs are noisier)
    amplitude_damping: float = 0.0  # T1-like relaxation
    phase_damping: float = 0.0      # T2-like dephasing
    readout: float = 0.0            # bit flip right before measurement

    def is_zero(self) -> bool:
        return all(v == 0 for v in asdict(self).values())

    def scaled(self, k: float) -> "NoiseModel":
        return NoiseModel(**{f: min(1.0, v * k) for f, v in asdict(self).items()})

    def with_(self, **kw) -> "NoiseModel":
        return replace(self, **kw)

    @classmethod
    def uniform(cls, p: float) -> "NoiseModel":
        """One-knob model for sweeps: 2-qubit noise 2x, readout 2x, damping 0.5x."""
        return cls(depolarizing=p, two_qubit_depol=min(1.0, 2 * p),
                   amplitude_damping=0.5 * p, phase_damping=0.5 * p,
                   readout=min(0.5, 2 * p))


def single_qubit_noise(nm: NoiseModel | None, wires) -> None:
    if nm is None:
        return
    for w in wires:
        if nm.depolarizing:
            qml.DepolarizingChannel(nm.depolarizing, wires=w)
        if nm.amplitude_damping:
            qml.AmplitudeDamping(nm.amplitude_damping, wires=w)
        if nm.phase_damping:
            qml.PhaseDamping(nm.phase_damping, wires=w)


def two_qubit_noise(nm: NoiseModel | None, wires) -> None:
    if nm is None or not nm.two_qubit_depol:
        return
    for w in wires:
        qml.DepolarizingChannel(nm.two_qubit_depol, wires=w)


def readout_noise(nm: NoiseModel | None, wires) -> None:
    if nm is None or not nm.readout:
        return
    for w in wires:
        qml.BitFlip(nm.readout, wires=w)
