"""Training loop: softmax cross-entropy, Adam, minibatches, best-on-validation checkpoint."""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .model import VQC


def cross_entropy(logits, y_onehot):
    z = logits - pnp.max(logits, axis=1, keepdims=True)
    log_probs = z - pnp.log(pnp.sum(pnp.exp(z), axis=1, keepdims=True))
    return -pnp.mean(pnp.sum(y_onehot * log_probs, axis=1))


def train(model: VQC, X, y, epochs: int = 30, lr: float = 0.05, batch_size: int = 32,
          X_val=None, y_val=None, seed: int = 0, verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    opt = qml.AdamOptimizer(lr)
    Y = np.eye(model.cfg.n_classes)[y]
    params = model.params
    best = (-1.0, [p.copy() for p in params])
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        losses = []
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb = pnp.array(X[idx], requires_grad=False)
            yb = pnp.array(Y[idx], requires_grad=False)
            cost = lambda w, s, b: cross_entropy(model.logits(xb, [w, s, b]), yb)
            new_params, loss = opt.step_and_cost(cost, *params)
            params = list(new_params)
            losses.append(float(loss))
        model.params = params
        tr = model.accuracy(X, y)
        va = model.accuracy(X_val, y_val) if X_val is not None else tr
        hist["loss"].append(float(np.mean(losses)))
        hist["train_acc"].append(tr)
        hist["val_acc"].append(va)
        if va > best[0]:
            best = (va, [p.copy() for p in params])
        if verbose:
            print(f"ep {ep:3d} loss {hist['loss'][-1]:.4f} train {tr:.3f} val {va:.3f}")
    model.params = best[1]
    hist["best_val_acc"] = best[0]
    hist["seconds"] = round(time.time() - t0, 2)
    return hist
