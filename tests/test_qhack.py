import numpy as np
import pytest

from qhack.attacks import (breidbart, detection_probability, intercept_resend,
                           min_sample_for_detection, trojan_horse, trojan_leak_probability)
from qhack.bb84 import QBER_ABORT, h2, run_bb84, secret_key_rate

N = 40_000


def test_h2_and_key_rate_threshold():
    assert h2(0.5) == pytest.approx(1.0)
    assert secret_key_rate(0.0) == pytest.approx(1.0)
    assert secret_key_rate(0.11) < 0.01  # ~11% is where BB84 stops producing key
    assert secret_key_rate(0.2) == 0.0


def test_no_eve_no_noise_is_perfect():
    r = run_bb84(N, 0.0, None, seed=1)
    assert r.qber_true == 0.0 and not r.aborted
    assert r.eve_info == pytest.approx(0.0, abs=0.02)
    assert 0.45 < len(r.key_alice) / (N * 0.8) < 0.55  # half the bases match


def test_channel_noise_sets_qber():
    r = run_bb84(N, 0.05, None, seed=2)
    assert r.qber_true == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("f", [0.25, 0.5, 1.0])
def test_intercept_resend_qber_is_f_over_4(f):
    r = run_bb84(N, 0.0, intercept_resend(f), seed=3)
    assert r.qber_true == pytest.approx(f / 4, abs=0.01)
    assert r.eve_info == pytest.approx(f / 2, abs=0.03)


def test_full_intercept_resend_is_aborted():
    assert run_bb84(N, 0.0, intercept_resend(1.0), seed=4).aborted


def test_breidbart_more_info_same_disturbance():
    r = run_bb84(N, 0.0, breidbart(1.0), seed=5)
    assert r.qber_true == pytest.approx(0.25, abs=0.01)
    assert r.eve_info == pytest.approx(2 * np.cos(np.pi / 8) ** 2 - 1, abs=0.03)


def test_trojan_horse_is_invisible_to_qber_but_kills_key():
    clean = run_bb84(N, 0.02, None, seed=6)
    th = run_bb84(N, 0.02, trojan_horse(mu_in=1e3, isolation_db=0), seed=6)
    assert th.qber_true == pytest.approx(clean.qber_true, abs=0.005)
    assert not th.aborted and th.key_rate == 0.0
    assert th.eve_info > 0.95


def test_isolation_defends_against_trojan_horse():
    assert trojan_leak_probability(1e3, 60) < 1e-3
    assert run_bb84(N, 0.02, trojan_horse(1e3, 60), seed=7).key_rate > 0.7


def test_detection_statistics():
    assert detection_probability(1) == pytest.approx(0.25)
    assert detection_probability(10) == pytest.approx(1 - 0.75 ** 10)
    k = min_sample_for_detection(0.999)
    assert detection_probability(k) >= 0.999 > detection_probability(k - 1)
    assert QBER_ABORT == 0.11
