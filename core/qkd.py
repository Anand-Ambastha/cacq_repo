"""
core/qkd.py
===========
Quantum key distribution (QKD) performance model.

Computes the secure key rate (SKR) and quantum bit error rate (QBER)
from the total channel transmission efficiency.

The model follows the decoy-state BB84 framework with WCP (weak coherent
pulse) sources and one-way classical post-processing.

References
----------
- Lo, Ma & Chen (2005), PRL 94, 230504
- Lütkenhaus (2000), PRA 61, 052304
- Wang (2005), PRL 94, 230503
"""

from __future__ import annotations

import numpy as np

from config.settings import MU_PHOTON, F_EC, DT


# ─────────────────────────────────────────────────────────────────────────────
# BINARY ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    """
    Binary Shannon entropy h(p) = −p log₂(p) − (1−p) log₂(1−p).

    Parameters
    ----------
    p : probability ∈ [0, 1]

    Returns
    -------
    float : entropy (bits)
    """
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


# ─────────────────────────────────────────────────────────────────────────────
# QBER MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_qber(eta: float) -> float:
    """
    Channel QBER as a function of total transmission efficiency.

    A simplified model incorporating a baseline optical alignment error
    and a channel-loss-dependent dark-count contribution.

        QBER = min(0.5,  0.02 + 0.1·(1−η))

    Parameters
    ----------
    eta : total channel efficiency ∈ (0, 1]

    Returns
    -------
    float : QBER ∈ [0, 0.5]
    """
    return float(min(0.5, 0.02 + 0.1 * (1.0 - eta)))


# ─────────────────────────────────────────────────────────────────────────────
# SECURE KEY RATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_skr_per_pulse(
    eta: float,
    mu: float = MU_PHOTON,
    f_ec: float = F_EC,
) -> float:
    """
    Secure key rate per sent pulse (bits pulse⁻¹).

    Decoy-state BB84 with WCP source:

        Q_µ   = η · µ · exp(−µ)          (single-photon gain approximation)
        QBER  = model(η)
        SKR   = max(0, Q_µ · [1 − f_EC · h(QBER)])

    Parameters
    ----------
    eta  : total channel efficiency
    mu   : mean photon number per pulse
    f_ec : error-correction efficiency

    Returns
    -------
    float : SKR per pulse (bits), ≥ 0
    """
    Q_mu = eta * mu * np.exp(-mu)
    QBER = compute_qber(eta)
    skr  = Q_mu * (1.0 - f_ec * binary_entropy(QBER))
    return float(max(0.0, skr))


def compute_skr_over_interval(
    eta: float,
    dt_s: float = DT,
    pulse_rate_Hz: float = 1e8,
    mu: float = MU_PHOTON,
    f_ec: float = F_EC,
) -> float:
    """
    Total secure key (bits) accumulated over a time interval ``dt_s``.

    Parameters
    ----------
    eta          : channel efficiency
    dt_s         : integration interval (seconds)
    pulse_rate_Hz: clock rate of the QKD source
    mu           : mean photon number
    f_ec         : error-correction efficiency

    Returns
    -------
    float : total bits generated in ``dt_s``
    """
    skr_pulse = compute_skr_per_pulse(eta, mu, f_ec)
    return float(skr_pulse * pulse_rate_Hz * dt_s)


def loss_dB_to_eta(loss_dB: float) -> float:
    """Convert channel loss in dB to linear power transmission."""
    return float(10.0 ** (-loss_dB / 10.0))
