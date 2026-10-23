import numpy as np
import pytest

from qai.fraud import load_creditcard, make_synthetic, read_creditcard
from qai.metrics import classification_metrics, from_logits


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("d") / "creditcard.csv", n=20_000, seed=1)


def test_schema_matches_real_file(csv):
    X, y, cols = read_creditcard(csv)
    assert cols[0] == "Time" and cols[-1] == "Amount" and len(cols) == 30
    assert set(np.unique(y)) == {0, 1} and y.mean() < 0.01


def test_train_is_balanced_test_keeps_all_frauds(csv):
    s = load_creditcard(csv, n_features=4, n_per_class=30, test_neg_ratio=10, seed=0)
    assert s.X_train.shape[1] == 4 and len(s.features) == 4
    assert (s.y_train == 1).sum() == (s.y_train == 0).sum() == 30
    n_pos = (s.y_test == 1).sum()
    assert (s.y_test == 0).sum() == 10 * n_pos
    assert s.test_fraud_rate == pytest.approx(1 / 11, abs=1e-6)
    assert s.X_test.min() >= 0 and s.X_test.max() <= np.pi + 1e-9


def test_no_leakage_between_train_and_test(csv):
    s, (Xf, yf) = load_creditcard(csv, n_features=4, n_per_class=30, seed=0, return_full_train=True)
    train_rows = {tuple(np.round(r, 8)) for r in s.X_train}
    test_rows = {tuple(np.round(r, 8)) for r in s.X_test}
    assert not train_rows & test_rows
    assert len(Xf) > len(s.X_train) and yf.mean() < 0.01  # full train keeps real imbalance


def test_pca_option(csv):
    s = load_creditcard(csv, n_features=3, n_per_class=20, select="pca")
    assert s.features == ["PC1", "PC2", "PC3"]


def test_metrics_catch_the_accuracy_trap():
    y = np.array([0] * 99 + [1])
    m = classification_metrics(y, np.zeros(100, int), np.zeros(100))
    assert m["acc"] == 0.99 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["bal_acc"] == 0.5


def test_from_logits_perfect():
    y = np.array([0, 1, 0, 1])
    logits = np.array([[2, -2], [-2, 2], [1, 0], [0, 1]], float)
    m = from_logits(y, logits)
    assert m["acc"] == m["roc_auc"] == m["pr_auc"] == 1.0


@pytest.mark.slow
def test_bench_runs_on_creditcard(csv):
    from qai.bench import run
    res = run(dataset="creditcard", data_path=csv, qubits=3, layers=1, epochs=2, n_per_class=20,
              noise_levels=(0.0, 0.05))
    assert res["config"]["metric"] == "pr_auc"
    assert "logreg_fulltrain" in res["classical"]
    assert 0 <= res["sweep"][1]["noise_aware"]["pr_auc"] <= 1


torch = pytest.importorskip("torch")


def test_hybrid_forward_and_gradients():
    from qai.noise import NoiseModel
    from qai.torch_hybrid import HybridNet
    m = HybridNet(n_in=6, n_qubits=3, n_layers=1, noise=NoiseModel.uniform(0.02))
    x = torch.rand(5, 6)
    out = m(x)
    assert out.shape == (5, 2)
    out.sum().backward()
    assert m.q.weights.grad is not None and torch.isfinite(m.q.weights.grad).all()


def test_set_noise_keeps_weights_and_twin_is_same_size():
    from qai.noise import NoiseModel
    from qai.torch_hybrid import ClassicalTwin, HybridNet, n_params, set_noise
    m = HybridNet(n_in=6, n_qubits=3, n_layers=1)
    w = m.q.weights.detach().clone()
    set_noise(m, NoiseModel.uniform(0.05))
    assert torch.equal(m.q.weights.detach(), w)
    assert abs(n_params(m) - n_params(ClassicalTwin(6, 3))) <= 6


@pytest.fixture(scope="module")
def easy_csv(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("e") / "creditcard.csv", n=20_000,
                          fraud_rate=0.01, seed=2, shift=2.5)


def test_hybrid_learns(easy_csv):
    from qai.torch_hybrid import HybridNet, evaluate_torch, train_torch
    s = load_creditcard(easy_csv, n_features=6, n_per_class=80, seed=0)
    m = HybridNet(n_in=6, n_qubits=3, n_layers=1)
    h = train_torch(m, s.X_train, s.y_train, epochs=15, lr=0.05)
    assert h["loss"][-1] < h["loss"][0]
    assert evaluate_torch(m, s.X_test, s.y_test)["roc_auc"] > 0.75
