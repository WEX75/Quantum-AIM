"""Attacks on BB84 and the matching defences / detection analysis.

intercept_resend(f)   Eve measures a fraction f of photons in a random basis and resends.
                      Expected QBER contribution = f/4, Eve knows ~f/2 of the sifted key.
breidbart(f)          Intercept-resend in the intermediate (Breidbart) basis: same QBER f/4
                      but Eve guesses correctly with p = cos^2(pi/8) ~ 0.854 on attacked bits.
trojan_horse(mu, A)   Eve injects bright light into Alice's modulator and reads the back
                      reflection. Leaks Alice's (basis, bit) WITHOUT raising the QBER.
                      Defence: optical isolation of A dB reduces the returning mean photon
                      number mu_out = mu * 10^(-A/10); P(leak per pulse) = 1 - exp(-mu_out).
"""
from __future__ import annotations

import numpy as np


def intercept_resend(fraction: float = 1.0):
    def attack(bits, bases, rng):
        n = len(bits)
        hit = rng.random(n) < fraction
        eve_bases = rng.integers(0, 2, n)
        eve_meas = np.where(eve_bases == bases, bits, rng.integers(0, 2, n))
        wire_bits = np.where(hit, eve_meas, bits)
        wire_bases = np.where(hit, eve_bases, bases)
        eve_guess = np.where(hit, eve_meas, rng.integers(0, 2, n))
        return wire_bits, wire_bases, eve_guess, 0.0
    return attack


def breidbart(fraction: float = 1.0):
    """Intermediate-basis attack. Modelled at the statistics level (exact for BB84 states)."""
    p_guess = np.cos(np.pi / 8) ** 2

    def attack(bits, bases, rng):
        n = len(bits)
        hit = rng.random(n) < fraction
        correct = rng.random(n) < p_guess
        eve_guess = np.where(hit, np.where(correct, bits, 1 - bits), rng.integers(0, 2, n))
        # resent state in the Breidbart basis -> Bob errs w.p. 1/4 in the sifted key
        flip = rng.random(n) < 0.25
        wire_bits = np.where(hit & flip, 1 - bits, bits)
        return wire_bits, bases, eve_guess, 0.0
    return attack


def trojan_leak_probability(mu_in: float, isolation_db: float) -> float:
    mu_out = mu_in * 10 ** (-isolation_db / 10)
    return float(1 - np.exp(-mu_out))


def trojan_horse(mu_in: float = 1.0, isolation_db: float = 0.0):
    p_leak = trojan_leak_probability(mu_in, isolation_db)

    def attack(bits, bases, rng):
        n = len(bits)
        leak = rng.random(n) < p_leak
        eve_guess = np.where(leak, bits, rng.integers(0, 2, n))
        # no disturbance: photons are untouched, so QBER stays at the channel level.
        # privacy amplification must remove the leaked fraction explicitly.
        return bits, bases, eve_guess, p_leak
    return attack


def detection_probability(n_sample: int, attack_fraction: float = 1.0,
                          p_noise: float = 0.0) -> float:
    """P(at least one error in the public sample) for intercept-resend.

    Without noise this is the textbook 1 - (3/4)^n for a full attack. With channel noise it
    is better to look at `qber_detectable` (errors are expected anyway)."""
    q = attack_fraction / 4 * (1 - 2 * p_noise) + p_noise
    return float(1 - (1 - q) ** n_sample)


def min_sample_for_detection(confidence: float = 0.999, attack_fraction: float = 1.0) -> int:
    q = attack_fraction / 4
    return int(np.ceil(np.log(1 - confidence) / np.log(1 - q)))
