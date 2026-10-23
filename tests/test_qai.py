import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from qai.baseline import run_baselines
from qai.data import load_dataset, load_file
from qai.mitigation import extrapolate_to_zero, zne_accuracy
from qai.model import VQC, VQCConfig, build_circuit
from qai.noise import NoiseModel
from qai.train import cross_entropy, train


@pytest.fixture(scope="module")
def moons():
    return load_dataset("moons", n_features=4, max_samples=120, seed=0)


def test_dataset_shapes_and_ranges(moons):
    assert moons.X_train.shape[1] == 4 and moons.n_classes == 2
    assert moons.X_train.min() >= 0 and moons.X_train.max() <= np.pi + 1e-9
    assert moons.X_test.min() >= 0 and moons.X_test.max() <= np.pi + 1e-9


def test_pca_reduces_features():
    s = load_dataset("breast_cancer", n_features=3, max_samples=150)
    assert s.X_train.shape[1] == 3


def test_csv_loader(tmp_path):
    p = tmp_path / "kit.csv"
    rows = ["f1,f2,label"] + [f"{i},{i*2},{'a' if i % 2 else 'b'}" for i in range(20)]
    p.write_text("\n".join(rows))
    X, y = load_file(p)
    assert X.shape == (20, 2) and set(y) == {0, 1}
    s = load_dataset(f"file:{p}", n_features=2, max_samples=None)
    assert s.n_classes == 2


def test_noise_model_scaling():
    nm = NoiseModel.uniform(0.02)
    assert nm.scaled(2).depolarizing == pytest.approx(0.04)
    assert NoiseModel().is_zero() and not nm.is_zero()


def test_noiseless_circuit_matches_mixed_with_zero_noise():
    cfg = VQCConfig(n_qubits=3, n_layers=2)
    m = VQC(cfg, None, seed=1)
    x = pnp.array(np.random.default_rng(0).uniform(0, np.pi, (5, 3)), requires_grad=False)
    pure = np.asarray(m.expvals(x))
    # a mixed-state device with a negligible channel must agree with the pure device
    tiny = VQC(cfg, NoiseModel(depolarizing=1e-9), params=m.params)
    assert np.allclose(pure, np.asarray(tiny.expvals(x)), atol=1e-6)


def test_noise_shrinks_expectation_values():
    cfg = VQCConfig(n_qubits=3, n_layers=2)
    m = VQC(cfg, None, seed=2)
    x = pnp.array(np.random.default_rng(0).uniform(0, np.pi, (8, 3)), requires_grad=False)
    clean = np.abs(np.asarray(m.expvals(x))).mean()
    noisy_c = build_circuit(cfg, NoiseModel.uniform(0.1))
    noisy = np.abs(np.asarray(m.expvals(x, circuit=noisy_c))).mean()
    assert noisy < clean


def test_cross_entropy_gradient_flows(moons):
    cfg = VQCConfig(n_qubits=4, n_layers=1)
    m = VQC(cfg, NoiseModel.uniform(0.01))
    xb = pnp.array(moons.X_train[:8], requires_grad=False)
    yb = pnp.array(np.eye(2)[moons.y_train[:8]], requires_grad=False)
    g = qml.grad(lambda w, s, b: cross_entropy(m.logits(xb, [w, s, b]), yb))(*m.params)
    assert all(np.isfinite(gi).all() for gi in g) and np.abs(g[0]).sum() > 0


def test_training_beats_chance(moons):
    m = VQC(VQCConfig(n_qubits=4, n_layers=2), None, seed=0)
    hist = train(m, moons.X_train, moons.y_train, epochs=8, lr=0.1)
    assert hist["loss"][-1] < hist["loss"][0]
    assert m.accuracy(moons.X_test, moons.y_test) > 0.7


def test_richardson_extrapolation_exact_on_polynomial():
    scales = [1, 2, 3]
    vals = [0.8 - 0.1 * k + 0.01 * k**2 for k in scales]
    assert extrapolate_to_zero(scales, vals) == pytest.approx(0.8)


def test_zne_does_not_crash_and_is_valid(moons):
    m = VQC(VQCConfig(n_qubits=4, n_layers=1), None, seed=0)
    acc = zne_accuracy(m, moons.X_test, moons.y_test, NoiseModel.uniform(0.05))
    assert 0.0 <= acc <= 1.0


def test_baselines(moons):
    b = run_baselines(moons)
    assert all(0 <= v["test_acc"] <= 1 and v["n_params"] > 0 for v in b.values())


@pytest.mark.slow
def test_bench_fast_end_to_end(tmp_path, monkeypatch):
    from qai import bench
    monkeypatch.chdir(tmp_path)
    bench.main(["--fast"])
    assert (tmp_path / "results" / "qai_moons.json").exists()
