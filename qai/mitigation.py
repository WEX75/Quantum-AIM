"""Error mitigation.

`zne_logits`: noise-scaling extrapolation. Run the same trained circuit with every noise
probability multiplied by k in `scales`, then Richardson/poly-extrapolate the expectation
values back to k=0. On the simulator we scale channel strengths directly; on hardware the
equivalent is unitary folding (`qml.transforms.mitigate_with_zne` + `fold_global`).
"""
from __future__ import annotations

import numpy as np

from .model import VQC, build_circuit
from .noise import NoiseModel


def extrapolate_to_zero(scales, values, order: int | None = None):
    """values: (len(scales), ...) -> extrapolated value at scale 0. order=None -> Richardson."""
    scales = np.asarray(scales, float)
    values = np.asarray(values, float)
    order = len(scales) - 1 if order is None else order
    flat = values.reshape(len(scales), -1)
    coeffs = np.polyfit(scales, flat, order)  # (order+1, n)
    return coeffs[-1].reshape(values.shape[1:])


def zne_expvals(model: VQC, X, noise: NoiseModel, scales=(1, 2, 3), order: int | None = None):
    evs = []
    for k in scales:
        circ = build_circuit(model.cfg, noise.scaled(k))
        evs.append(np.asarray(model.expvals(X, circuit=circ)))
    ev0 = extrapolate_to_zero(scales, np.stack(evs), order)
    return np.clip(ev0, -1, 1)


def zne_accuracy(model: VQC, X, y, noise: NoiseModel, scales=(1, 2, 3), order=None) -> float:
    ev = zne_expvals(model, X, noise, scales, order)
    _, scale, bias = model.params
    logits = ev * np.asarray(scale) + np.asarray(bias)
    return float(np.mean(np.argmax(logits, axis=1) == y))
