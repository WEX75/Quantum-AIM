"""BB84 quantum key distribution, simulated exactly with numpy (single photons, no multi-photon).

Bases: 0 = Z (rectilinear), 1 = X (diagonal). Measuring in the wrong basis gives a uniformly
random bit. The pipeline mirrors the real protocol:
  prepare -> channel (noise / Eve) -> measure -> sift -> estimate QBER on a public sample
  -> abort if QBER > threshold -> (asymptotic) secret key rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QBER_ABORT = 0.11  # BB84 one-way post-processing threshold (1 - 2 h(Q) = 0 at Q ~ 11%)


def h2(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def secret_key_rate(qber, leaked: float = 0.0) -> float:
    """Shor-Preskill asymptotic rate per sifted bit, minus any extra leaked information."""
    return float(max(0.0, 1 - 2 * h2(qber) - leaked))


@dataclass
class Transmission:
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_bits: np.ndarray
    eve_bits: np.ndarray | None = None      # Eve's guess of Alice's bit (nan-free, -1 = no guess)
    meta: dict = field(default_factory=dict)


def prepare(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    return rng.integers(0, 2, n), rng.integers(0, 2, n)


def measure(bits, prep_bases, meas_bases, rng) -> np.ndarray:
    same = prep_bases == meas_bases
    return np.where(same, bits, rng.integers(0, 2, len(bits)))


def channel_noise(bits, p_flip: float, rng) -> np.ndarray:
    """Depolarising-like channel seen at the detector: flips with prob p_flip."""
    if p_flip <= 0:
        return bits
    return bits ^ (rng.random(len(bits)) < p_flip)


@dataclass
class SiftResult:
    key_alice: np.ndarray
    key_bob: np.ndarray
    key_eve: np.ndarray
    qber_true: float
    qber_estimate: float
    sample_size: int
    aborted: bool
    key_rate: float
    eve_info: float  # Eve's advantage over chance on the final key (2*P(match)-1)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)} | {
            "sifted_len": int(len(self.key_alice))}


def sift_and_estimate(tx: Transmission, sample_frac: float = 0.2, rng=None,
                      threshold: float = QBER_ABORT, leaked: float = 0.0) -> SiftResult:
    rng = rng or np.random.default_rng()
    keep = tx.alice_bases == tx.bob_bases
    a, b = tx.alice_bits[keep], tx.bob_bits[keep]
    e = tx.eve_bits[keep] if tx.eve_bits is not None else -np.ones_like(a)
    n = len(a)
    errors = a != b
    k = max(1, int(sample_frac * n)) if n else 0
    sample = rng.choice(n, k, replace=False) if n else np.array([], int)
    mask = np.ones(n, bool)
    mask[sample] = False
    q_true = float(errors.mean()) if n else 0.0
    q_est = float(errors[sample].mean()) if k else 0.0
    aborted = q_est > threshold
    rest_a, rest_b, rest_e = a[mask], b[mask], e[mask]
    # Eve's advantage over random guessing on the final key: 2*P(match) - 1, in [0, 1].
    # 0 = knows nothing, 1 = knows every bit. Full intercept-resend -> ~0.5.
    match = float(np.mean(rest_e == rest_a)) if len(rest_a) else 0.5
    eve_info = max(0.0, 2 * match - 1)
    rate = 0.0 if aborted else secret_key_rate(q_est, leaked)
    return SiftResult(rest_a, rest_b, rest_e, q_true, q_est, k, aborted, rate, eve_info)


def run_bb84(n: int = 20_000, p_noise: float = 0.0, attack=None, sample_frac: float = 0.2,
             seed: int = 0, threshold: float = QBER_ABORT) -> SiftResult:
    """attack: callable(bits, bases, rng) -> (bits_on_wire, bases_on_wire, eve_guess, leaked)."""
    rng = np.random.default_rng(seed)
    bits, bases = prepare(n, rng)
    wire_bits, wire_bases, eve_guess, leaked = bits, bases, None, 0.0
    if attack is not None:
        wire_bits, wire_bases, eve_guess, leaked = attack(bits, bases, rng)
    bob_bases = rng.integers(0, 2, n)
    bob_bits = measure(wire_bits, wire_bases, bob_bases, rng)
    bob_bits = channel_noise(bob_bits, p_noise, rng)
    tx = Transmission(bits, bases, bob_bases, bob_bits, eve_guess)
    return sift_and_estimate(tx, sample_frac, rng, threshold, leaked)
