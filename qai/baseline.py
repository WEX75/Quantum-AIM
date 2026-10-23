"""Classical baselines with parameter counts, so every quantum claim has a fair comparison.

Same training data as the quantum model (fair, small-data comparison). If `full_train` is given
(e.g. the whole imbalanced fraud train split), also a logistic regression with class weights on
all of it: the strong reference a jury will ask about.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .metrics import classification_metrics


def _mlp_params(m: MLPClassifier) -> int:
    return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))


def _score(model, X, y, n_params) -> dict:
    pred = model.predict(X)
    p = None
    if len(np.unique(y)) == 2:  # scores for ROC/PR-AUC: probabilities, else decision function
        p = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba")
             else model.decision_function(X))
    m = classification_metrics(y, pred, p)
    m["test_acc"] = m["acc"]
    m["n_params"] = int(n_params)
    return m


def run_baselines(split, seed: int = 0, mlp_hidden=(8,), full_train=None) -> dict:
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test
    out = {}
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    out["logreg"] = _score(lr, Xte, yte, lr.coef_.size + lr.intercept_.size)
    mlp = MLPClassifier(hidden_layer_sizes=mlp_hidden, max_iter=3000, random_state=seed).fit(Xtr, ytr)
    out[f"mlp{list(mlp_hidden)}"] = _score(mlp, Xte, yte, _mlp_params(mlp))
    svm = SVC(kernel="rbf", random_state=seed).fit(Xtr, ytr)
    out["svm_rbf"] = _score(svm, Xte, yte, svm.support_vectors_.size)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed).fit(Xtr, ytr)
    out["rf"] = _score(rf, Xte, yte, sum(t.tree_.node_count for t in rf.estimators_))
    if full_train is not None:
        Xf, yf = full_train
        lrf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xf, yf)
        out["logreg_fulltrain"] = _score(lrf, Xte, yte, lrf.coef_.size + lrf.intercept_.size)
    return out
