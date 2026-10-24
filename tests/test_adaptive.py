"""Statistical guarantees of the adaptive-shot triage (qai/adaptive.py)."""
import numpy as np
import pytest
from scipy.stats import norm

from qai import adaptive as ad

N_SEQ, B, K, ALPHA = 10_000, 16, 64, 0.01            # 10,000 sequences, looks every 16 shots up to 1024 shots
DISTRIBUTIONS = {                                    # (P(z_q = +1) per qubit, weights)
    "fair coin, mu = 0 (Hoeffding worst case)": ([0.5], [1.0]),
    "skewed coin, mu = 0.9 (low variance)": ([0.95], [1.0]),
    "MOCO-like h, 4 qubits": ([0.8, 0.3, 0.6, 0.9], [0.47, 0.22, 0.25, 0.06]),
}


def anytime_miss_rate(prob_up, w, kind, alpha=ALPHA, seed=0):
    """Fraction of sequences whose interval misses the true mean at ANY of the K looks."""
    prob_up, w = np.asarray(prob_up), np.asarray(w)
    mu = float(w @ (2 * prob_up - 1))
    rng = np.random.default_rng(seed)
    P = np.tile(prob_up, (N_SEQ, 1))
    rows = np.arange(N_SEQ)
    ci = None if kind == "naive CLT" else ad.AnytimeCI(kind, alpha, N_SEQ)
    s1, s2, missed = np.zeros(N_SEQ), np.zeros(N_SEQ), np.zeros(N_SEQ, bool)
    for k in range(1, K + 1):
        a1, a2 = ad.sample_shot_sums(P, w, B, rng)
        s1, s2, n = s1 + a1, s2 + a2, k * B
        if ci is None:                                # fixed-n 99% interval, re-checked at every look
            center = s1 / n
            eps = norm.ppf(1 - alpha / 2) * np.sqrt(np.maximum(s2 - s1 ** 2 / n, 0) / (n - 1) / n)
        else:
            center, eps = ci.update(rows, a1, a2, B)
        missed |= np.abs(center - mu) > eps
    return missed.mean()


@pytest.mark.parametrize("kind", ad.CI_KINDS)
@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_anytime_interval_misses_at_most_alpha(kind, name):
    rate = anytime_miss_rate(*DISTRIBUTIONS[name], kind)
    print(f"{kind:10s} {name:45s} miss rate {rate:.4f} (alpha {ALPHA})")
    assert rate <= ALPHA


def test_naive_interval_rechecked_every_batch_violates_alpha():
    # the coverage test has power: optional stopping breaks a fixed-n interval
    rate = anytime_miss_rate(*DISTRIBUTIONS["fair coin, mu = 0 (Hoeffding worst case)"], "naive CLT")
    print(f"naive CLT re-checked at {K} looks: miss rate {rate:.4f} (alpha {ALPHA})")
    assert rate > 2 * ALPHA


def test_alpha_schedule_sums_to_alpha():
    assert np.isclose(ad.alpha_schedule(ALPHA, np.arange(1, 10 ** 6)).sum(), ALPHA, rtol=1e-5)


def _rows(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, (n, 4)), np.array([0.47, 0.22, 0.25, 0.06])


def test_no_stopping_reproduces_fixed_shots():
    P, w = _rows()
    reps, S = 100, 256
    tiled = np.tile(P, (reps, 1))
    rng = np.random.default_rng(2)
    seq = ad.triage(tiled, w, 0.0, B, ALPHA, [S], "hoeffding", rng, stop=False)[S]
    fixed = ad.fixed_shot_mean(tiled, w, S, rng)
    assert (seq["shots"] == S).all() and not seq["decided"].any()
    mu = (2 * P - 1) @ w
    var = ((1 - (2 * P - 1) ** 2) * w ** 2).sum(1) / S          # Var of the mean of h, independent qubits
    for est in (seq["mean"], fixed):
        m = est.reshape(reps, -1)
        z = (m.mean(0) - mu) / np.sqrt(var / reps)
        assert np.abs(z).max() < 5
        assert 0.9 < (m.var(0, ddof=1) / var).mean() < 1.1


def test_smaller_cap_is_exactly_the_truncated_run():
    P, w = _rows()
    both = ad.triage(P, w, -0.1, B, ALPHA, [256, 1024], "bernstein", np.random.default_rng(3))
    only = ad.triage(P, w, -0.1, B, ALPHA, [256], "bernstein", np.random.default_rng(3))
    for key in ("shots", "mean", "decided"):
        assert np.array_equal(both[256][key], only[256][key])


@pytest.mark.parametrize("kind", ad.CI_KINDS)
def test_early_decisions_are_wrong_at_most_alpha(kind):
    P, w = _rows(2000, seed=4)
    mu = (2 * P - 1) @ w
    t_h = np.quantile(mu, 0.5)
    res = ad.triage(P, w, t_h, B, ALPHA, [1024], kind, np.random.default_rng(5))[1024]
    wrong = res["decided"] & (res["fraud"] != (mu <= t_h))
    assert wrong.mean() <= ALPHA


def test_plugin_bernstein_is_narrower_than_union_bound_intervals_at_low_variance():
    P, w = np.full((2000, 1), 0.95), np.array([1.0])
    widths = {}
    for kind in ad.CI_KINDS:
        ci, rng = ad.AnytimeCI(kind, ALPHA, 2000), np.random.default_rng(8)
        for _ in range(K):
            a1, a2 = ad.sample_shot_sums(P, w, B, rng)
            _, eps = ci.update(np.arange(2000), a1, a2, B)
        widths[kind] = eps.mean()
    assert widths["plugin-bernstein"] < widths["bernstein"] < widths["hoeffding"]


def test_robust_encoder_with_q_zero_is_the_kit_encoder():
    import pandas as pd
    from qai.protocol import AngleEncoder, RobustAngleEncoder
    rng = np.random.default_rng(9)
    cols = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    frame = pd.DataFrame(rng.standard_t(3, (2000, 30)), columns=cols)
    train, test = frame.iloc[:1500], frame.iloc[1500:]
    kit = AngleEncoder(4, 0).fit(train)
    assert np.allclose(RobustAngleEncoder(4, 0, q=0).fit(train).transform(test), kit.transform(test), atol=1e-12)
    robust = RobustAngleEncoder(4, 0, q=5).fit(train).transform(train)
    assert robust.min() >= 0 and robust.max() <= np.pi
    assert np.isclose((robust == 0).mean(0), 0.05, atol=0.002).all()     # 5% of the train bag clipped at each end


def test_refine_tops_up_flagged_rows_only():
    P, w = _rows(50)
    shots = np.full(50, 32)
    flagged = np.arange(50) % 2 == 0
    new_shots, mean, extra_exec = ad.refine(P, w, shots, np.zeros(50), flagged, 1024, np.random.default_rng(6))
    assert (new_shots[flagged] == 1024).all() and (new_shots[~flagged] == 32).all()
    assert extra_exec.sum() == flagged.sum()


@pytest.mark.parametrize("p", [0.01, 0.05, 0.2])
def test_noise_meter_is_exact_with_exact_scores(p):
    s0 = np.random.default_rng(7).uniform(0, 1, 500)
    lam2 = (1 - 4 * p / 3) ** 2
    sp = 0.5 - lam2 * (0.5 - s0)
    for est in (ad.lambda2_median, ad.lambda2_least_squares):
        assert np.isclose(ad.p_from_lambda2(est(sp, s0)), p, atol=1e-12)
