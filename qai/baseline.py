"""Classical baselines with parameter counts, so every quantum claim has a fair comparison."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _mlp_params(m: MLPClassifier) -> int:
    return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))


def run_baselines(split, seed: int = 0, mlp_hidden=(8,)) -> dict:
    out = {}
    lr = LogisticRegression(max_iter=2000).fit(split.X_train, split.y_train)
    out["logreg"] = {"test_acc": float(lr.score(split.X_test, split.y_test)),
                     "n_params": int(lr.coef_.size + lr.intercept_.size)}
    mlp = MLPClassifier(hidden_layer_sizes=mlp_hidden, max_iter=3000, random_state=seed)
    mlp.fit(split.X_train, split.y_train)
    out[f"mlp{list(mlp_hidden)}"] = {"test_acc": float(mlp.score(split.X_test, split.y_test)),
                                     "n_params": _mlp_params(mlp)}
    svm = SVC(kernel="rbf").fit(split.X_train, split.y_train)
    out["svm_rbf"] = {"test_acc": float(svm.score(split.X_test, split.y_test)),
                      "n_params": int(svm.support_vectors_.size)}
    return out
