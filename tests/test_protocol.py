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
