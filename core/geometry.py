"""
core/geometry.py
================
Gaussian beam propagation geometry for the CAQC satellite link.

Models the fraction of transmitted power captured by a circular
aperture receiver in the far field using Gaussian beam optics.

Reference
---------
Saleh & Teich (2019), *Fundamentals of Photonics*, 3rd ed., Wiley
"""

from __future__ import annotations

import numpy as np

from config.settings import D_TX, D_RX


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN BEAM PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

def rayleigh_range(lambda_m: float, w0: float = D_TX / 2.0) -> float:
    """
    Rayleigh range z_R = π·w₀² / λ.

    Parameters
    ----------
    lambda_m : wavelength (m)
    w0       : beam waist radius (m); defaults to D_TX / 2

    Returns
    -------
    float : Rayleigh range (m)
    """
    return np.pi * w0 ** 2 / lambda_m


def beam_radius_at_range(
    range_m: float,
    lambda_m: float,
    w0: float = D_TX / 2.0,
) -> float:
    """
    Gaussian beam radius w(z) at propagation distance ``range_m``.

        w(z) = w₀ · √(1 + (z / z_R)²)

    Parameters
    ----------
    range_m  : link distance (m)
    lambda_m : wavelength (m)
    w0       : beam waist radius (m)

    Returns
    -------
    float : beam radius (m)
    """
    z_R = rayleigh_range(lambda_m, w0)
    return w0 * np.sqrt(1.0 + (range_m / z_R) ** 2)


def geometric_eta(
    range_m: float,
    lambda_m: float,
    d_rx: float = D_RX,
    d_tx: float = D_TX,
) -> float:
    """
    Fraction of Gaussian beam power captured by the receiver aperture.

    Uses the approximation:

        η_geo = 1 − exp[−2·(D_rx / (2·w(z)))²]

    which accounts for finite receiver size relative to beam footprint.

    Parameters
    ----------
    range_m  : link range (m)
    lambda_m : wavelength (m)
    d_rx     : receiver aperture diameter (m)
    d_tx     : transmitter aperture diameter (m)

    Returns
    -------
    float : geometric power coupling efficiency ∈ (0, 1]
    """
    w0  = d_tx / 2.0
    w_z = beam_radius_at_range(range_m, lambda_m, w0)
    eta = 1.0 - np.exp(-2.0 * (d_rx / (2.0 * w_z)) ** 2)
    return float(np.clip(eta, 1e-12, 1.0))


def geometric_loss_dB(
    range_m: float,
    lambda_m: float,
    d_rx: float = D_RX,
    d_tx: float = D_TX,
) -> float:
    """
    Geometric beam-spreading loss in dB.

    Parameters
    ----------
    range_m  : link range (m)
    lambda_m : wavelength (m)
    d_rx     : receiver aperture diameter (m)
    d_tx     : transmitter aperture diameter (m)

    Returns
    -------
    float : geometric loss (dB, positive = loss)
    """
    eta = geometric_eta(range_m, lambda_m, d_rx, d_tx)
    return -10.0 * np.log10(eta)
