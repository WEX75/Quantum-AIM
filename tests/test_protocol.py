"""The Task 2 protocol reproduces the kit, and the physics claims behind our methods hold."""
import numpy as np
import pytest
import torch

from qai import kit_vqc as kit
from qai import moco
from qai.protocol import (DATA_PATH, clean_predict, gate_stats, gradient_ledger, noisy_predict,
                          tracked_executions_per_example)

rng = np.random.default_rng(0)
Z = rng.uniform(0, np.pi, (40, 4))
W = kit.init_weights(0).detach()


def test_kit_circuit_counts():
    assert gate_stats(kit.vqc, Z[0], W) == {"gates": 93, "two_qubit": 33, "noise_locations": 126, "depth": 45}


def test_kit_ledger_formula():
    row = gradient_ledger("baseline", 30, 590, 32, 56_962)
    assert row["train executions"] == 1_150_500 and row["validation executions"] == 1_708_860


def test_tracker_skips_parameters_outside_light_cone():
    # PennyLane's parameter shift already skips the 12 dead parameters: 1 + 2 * 20, not 1 + 2 * 32
    assert tracked_executions_per_example(kit.vqc, W, Z[:4], np.array([0, 1, 0, 1])) == 41


def test_density_matrix_at_zero_noise_matches_state_vector():
    assert np.allclose(noisy_predict(Z, W, 0.0, kit.vqc), clean_predict(Z, W, kit.vqc), atol=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.05])
def test_lightcone_pruning_leaves_noisy_scores_unchanged(p):
    full, pruned = noisy_predict(Z, W, p, kit.vqc), noisy_predict(Z, W, p, kit.vqc_lightcone)
    assert np.allclose(full, pruned, atol=1e-10)
    assert gate_stats(kit.vqc_lightcone, Z[0], W)["noise_locations"] == 126 - 18


def _moco_params(fm="single"):
    coef = rng.normal(size=4 * (2 if fm == "single" else 3))
    return moco.compile_to_circuit(coef, fm)


def test_moco_compiled_circuit_matches_closed_form():
    params = _moco_params()
    assert np.allclose(clean_predict(Z, None, moco.make_circuit(params)), moco.score_exact(params, moco.bloch_exact(Z)))


@pytest.mark.parametrize("p", [0.001, 0.05, 0.3])
def test_moco_noise_is_a_uniform_contraction(p):
    params = _moco_params()
    clean, noisy = clean_predict(Z, None, moco.make_circuit(params)), noisy_predict(Z, None, p, moco.make_circuit(params))
    lam = 1 - 4 * p / 3
    assert np.allclose(noisy, 0.5 - lam ** 2 * (0.5 - clean), atol=1e-12)
    assert np.allclose(moco.transfer_threshold(clean, p, 2), noisy, atol=1e-12)


def test_moco_heisenberg_readout_holds_with_entangling_feature_map():
    params = _moco_params("linear_cnot")
    exact = moco.score_exact(params, moco.bloch_exact(Z, "linear_cnot"))
    assert np.allclose(clean_predict(Z, None, moco.make_circuit(params)), exact, atol=1e-10)


@pytest.mark.slow
@pytest.mark.skipif(not DATA_PATH.exists(), reason="needs data/creditcard.csv")
def test_common_split_matches_kit():
    from qai.protocol import common_split, load_df
    s = common_split(load_df())
    assert (len(s.y_train), len(s.y_val), len(s.y_test)) == (170_883, 56_962, 56_962)
    assert (s.y_train.sum(), s.y_val.sum(), s.y_test.sum()) == (295, 99, 98)


@pytest.mark.parametrize("p", [0.05, 0.3])
def test_moco_effective_compression_is_lambda_squared(p):
    from qai.experiments.exp_noise_shots import effective_compression
    circ = moco.make_circuit(_moco_params())
    lam = 1 - 4 * p / 3
    assert abs(effective_compression(noisy_predict(Z, None, p, circ), clean_predict(Z, None, circ)) - lam ** 2) < 1e-6


def test_r2_joint_distribution_readout_matches_expectation_values():
    from qai.experiments import exp_entangled as r2
    from qai.experiments.exp_noise_shots import SIGNS, identity
    theta = torch.tensor(rng.normal(size=r2.N_ROT + r2.N))
    w = torch.softmax(theta[r2.N_ROT:], 0).numpy()
    probs = noisy_predict(Z, theta, 0.05, r2.probs_circuit, identity)
    expected = noisy_predict(Z, theta, 0.05, r2.circuit, r2.readout_for(theta.numpy()))
    assert np.allclose((1 - (probs @ SIGNS) @ w) / 2, expected, atol=1e-10)


def test_tracker_matches_ledger_formulas():
    from qai.experiments import exp_entangled as r2
    from qai.protocol import tracked_executions_per_example_readout
    assert moco.tracker_check(Z[:5], "single")["executions_per_row"] == 2
    assert moco.tracker_check(Z[:5], "linear_cnot")["executions_per_row"] == 3
    theta = torch.zeros(r2.N_ROT + r2.N, dtype=torch.float64)
    readout = lambda out, w: (1 - torch.stack(list(out), -1) @ torch.softmax(w[r2.N_ROT:], 0)) / 2
    assert tracked_executions_per_example_readout(r2.circuit, theta, Z[:5], np.array([0, 1, 0, 1, 0]),
                                                  readout) == 1 + 2 * r2.N_ROT
