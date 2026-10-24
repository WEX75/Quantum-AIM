"""Adaptive shot allocation for readouts h = sum_q w_q z_q in [-1, 1] (w_q >= 0, sum w = 1, z_q = +-1 per shot).

  * anytime-valid confidence intervals for the mean of h: Hoeffding or empirical Bernstein (Maurer & Pontil) with a
    union bound over the batches, alpha_k = 6 alpha / (pi^2 k^2), or the predictable plug-in empirical Bernstein
    confidence sequence of Waudby-Smith & Ramdas (2024), valid at every shot without a union bound. In all three
    cases P(the interval misses the mean at ANY batch) <= alpha;
  * sequential triage: measure in batches of b shots, stop as soon as the interval excludes the threshold t_h;
  * the fixed-shot estimator (independent binomials per qubit: exact for a product state);
  * the noise meter: lambda^2 and p from a few calibration transactions whose noiseless score is known classically.

Score convention of the kit: s = (1 - <h>) / 2, fraud if s >= t, i.e. <h> <= t_h = 1 - 2 t.
"""
from __future__ import annotations

import numpy as np

H_RANGE = 2.0                 # h lies in [-1, 1]
CI_KINDS = ("hoeffding", "bernstein", "plugin-bernstein")
PLUGIN_TRUNCATION = 0.5       # c in lambda_t = min(..., c), Waudby-Smith & Ramdas (2024) Sec. 3.1
_MAX_DRAWS = 20_000_000       # uniforms drawn at once (memory bound of the shot sampler)


# ------------------------------------------------------------------ anytime-valid confidence intervals
def alpha_schedule(alpha, k):
    """Error budget of look k = 1, 2, ...: sum_k 6 alpha / (pi^2 k^2) = alpha."""
    return alpha * 6.0 / (np.pi ** 2 * np.asarray(k, dtype=float) ** 2)


def hoeffding_halfwidth(n, delta, value_range=H_RANGE):
    """Two-sided Hoeffding: P(|mean_n - mu| >= eps) <= 2 exp(-2 n eps^2 / R^2)."""
    return value_range * np.sqrt(np.log(2.0 / delta) / (2.0 * n))


def bernstein_halfwidth(n, var, delta, value_range=H_RANGE):
    """Two-sided empirical Bernstein (Maurer & Pontil 2009, Thm 4, delta / 2 per side), var = unbiased sample
    variance of the n observations, value_range = width of their support."""
    log_term = np.log(4.0 / delta)
    return np.sqrt(2.0 * var * log_term / n) + 7.0 * value_range * log_term / (3.0 * (n - 1))


def halfwidth(kind, n, sum1, sum2, delta):
    """Half-width after n observations with running sums sum1 = sum h, sum2 = sum h^2."""
    if kind == "hoeffding":
        return hoeffding_halfwidth(n, delta) * np.ones_like(sum1)
    if kind == "bernstein":
        var = np.maximum(sum2 - sum1 ** 2 / n, 0.0) / (n - 1)
        return bernstein_halfwidth(n, var, delta)
    raise ValueError(kind)


def psi_e(lam):
    return (-np.log1p(-lam) - lam) / 4.0


class AnytimeCI:
    """Anytime-valid interval for the mean of h in [-1, 1] of each of n_rows independent rows, fed batch by batch.

    "plugin-bernstein" is the predictable plug-in empirical Bernstein confidence sequence (Waudby-Smith & Ramdas 2024,
    Thm 2) on X = (h + 1) / 2 in [0, 1]:  [sum l_i X_i / sum l_i  +-  (log(2/alpha) + sum v_i psi_E(l_i)) / sum l_i],
    v_i = 4 (X_i - mu_hat_{i-1})^2. It holds for ANY predictable l_i in (0, 1) and mu_hat_{i-1} in [0, 1]; here both
    are fixed for a whole batch from the previous batches only (so they are predictable), which keeps the batch
    update in terms of the sums of h and h^2."""

    def __init__(self, kind, alpha, n_rows):
        if kind not in CI_KINDS:
            raise ValueError(kind)
        self.kind, self.alpha = kind, alpha
        self.n, self.k = np.zeros(n_rows), np.zeros(n_rows, dtype=int)
        self.s1, self.s2 = np.zeros(n_rows), np.zeros(n_rows)
        self.A, self.L, self.V, self.Q = (np.zeros(n_rows) for _ in range(4))

    def update(self, rows, a1, a2, b):
        """Add b shots with sums a1 = sum h, a2 = sum h^2 to `rows`; returns (center, half-width) in h units."""
        if self.kind == "plugin-bernstein":
            n_prev = self.n[rows]
            mu_hat = (0.5 + (self.s1[rows] + n_prev) / 2) / (n_prev + 1)
            sig2 = (0.25 + self.Q[rows]) / (n_prev + 1)
            t = n_prev + 1
            lam = np.minimum(np.sqrt(2 * np.log(2 / self.alpha) / (sig2 * t * np.log1p(t))), PLUGIN_TRUNCATION)
            sx, sx2 = (a1 + b) / 2, (a2 + 2 * a1 + b) / 4
            dev2 = np.maximum(sx2 - 2 * mu_hat * sx + b * mu_hat ** 2, 0.0)
            self.A[rows] += lam * sx
            self.L[rows] += lam * b
            self.V[rows] += psi_e(lam) * 4 * dev2
            self.Q[rows] += dev2
        self.n[rows] += b
        self.k[rows] += 1
        self.s1[rows] += a1
        self.s2[rows] += a2
        if self.kind == "plugin-bernstein":
            L = self.L[rows]
            return 2 * self.A[rows] / L - 1, 2 * (np.log(2 / self.alpha) + self.V[rows]) / L
        n = self.n[rows]
        return self.s1[rows] / n, halfwidth(self.kind, n, self.s1[rows], self.s2[rows],
                                            alpha_schedule(self.alpha, self.k[rows]))


# ------------------------------------------------------------------ shot samplers (product state)
def sample_shot_sums(prob_up, w, n_shots, rng):
    """n_shots shots on every row; prob_up (A, Q) = P(z_q = +1), qubits independent. Returns per-row sum h, sum h^2."""
    A, Q = prob_up.shape
    s1, s2 = np.zeros(A), np.zeros(A)
    step = max(1, min(n_shots, _MAX_DRAWS // max(A * Q, 1)))
    done = 0
    while done < n_shots:
        m = min(step, n_shots - done)
        z = np.where(rng.random((A, m, Q)) < prob_up[:, None, :], 1.0, -1.0)
        h = z @ w
        s1 += h.sum(1)
        s2 += (h * h).sum(1)
        done += m
    return s1, s2


def fixed_shot_mean(prob_up, w, shots, rng):
    """Mean of h over `shots` shots per row (int or (N,) array): independent binomial counts per qubit."""
    shots = np.broadcast_to(np.asarray(shots), prob_up.shape[:1])[:, None]
    z = 2.0 * rng.binomial(shots, prob_up) / shots - 1.0
    return z @ w


def triage(prob_up, w, t_h, b, alpha, caps, kind, rng, stop=True):
    """Batches of b shots per still-undecided row; a row stops at batch k when its anytime CI excludes t_h.

    caps: shot caps (multiples of b). The stopping rule does not depend on the cap, so one simulation up to the
    largest cap gives the exact result for every smaller cap. stop=False disables stopping (every row runs to the cap).
    Returns {cap: dict(shots, mean, decided, fraud)}: shots used, running mean of h at stopping time, whether the row
    stopped because its interval excluded t_h (False = it reached the cap), and the decision: the side of t_h the
    interval is on for decided rows, mean <= t_h (point estimate) for capped rows."""
    caps = sorted(caps)
    if any(c % b for c in caps) or b < 2:
        raise ValueError("caps must be multiples of b, b >= 2")
    N = len(prob_up)
    t_h = np.broadcast_to(np.asarray(t_h, dtype=float), (N,))
    ci = AnytimeCI(kind, alpha, N)
    sum1 = np.zeros(N)
    stop_batch = np.zeros(N, dtype=int)                       # 0 = has not stopped
    ci_fraud = np.zeros(N, dtype=bool)
    active = np.arange(N)
    at_cap = {}
    for k in range(1, caps[-1] // b + 1):
        if len(active):
            a1, a2 = sample_shot_sums(prob_up[active], w, b, rng)
            sum1[active] += a1
            if stop:
                center, eps = ci.update(active, a1, a2, b)
                legit, fraud = center - eps > t_h[active], center + eps < t_h[active]
                done = legit | fraud
                stop_batch[active[done]] = k
                ci_fraud[active[fraud]] = True
                active = active[~done]
        if k * b in caps:
            at_cap[k * b] = sum1.copy()                       # stopped rows are frozen: their sum is the one at stop
    out = {}
    for c in caps:
        decided = (stop_batch > 0) & (stop_batch * b <= c)
        shots = np.where(decided, stop_batch * b, c)
        mean = at_cap[c] / shots
        out[c] = dict(shots=shots, mean=mean, decided=decided, fraud=np.where(decided, ci_fraud, mean <= t_h))
    return out


def refine(prob_up, w, shots, mean, flagged, target, rng):
    """Top up the flagged rows to `target` shots in total (one more execution each); returns new shots and mean."""
    extra = np.where(flagged, np.maximum(target - shots, 0), 0)
    total = mean * shots
    for e in np.unique(extra[extra > 0]):
        rows = np.flatnonzero(extra == e)
        total[rows] += sample_shot_sums(prob_up[rows], w, int(e), rng)[0]
    new_shots = shots + extra
    return new_shots, total / new_shots, (extra > 0).astype(int)


# ------------------------------------------------------------------ noise meter
def lambda2_median(s_hat, s_0):
    """Median over calibration rows of |s_hat - 1/2| / |s_0 - 1/2| (the theorem: = lambda^2 with exact scores)."""
    d0 = np.abs(s_0 - 0.5)
    keep = d0 > 1e-9
    return float(np.median(np.abs(s_hat - 0.5)[keep] / d0[keep]))


def lambda2_least_squares(s_hat, s_0):
    """Least squares through the origin of (s_hat - 1/2) on (s_0 - 1/2). Unbiased: E[s_hat] - 1/2 = lambda^2 (s_0 - 1/2)."""
    d0 = s_0 - 0.5
    return float(np.dot(d0, s_hat - 0.5) / np.dot(d0, d0))


def p_from_lambda2(lambda2, k=2):
    """Invert lambda^k = (1 - 4p/3)^k for the per-gate depolarizing probability (k gates per qubit)."""
    lam = np.clip(lambda2, 0.0, 1.0) ** (1.0 / k)
    return 0.75 * (1.0 - lam)
