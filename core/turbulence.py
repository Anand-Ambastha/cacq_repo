"""
core/turbulence.py
==================
Atmospheric turbulence model for free-space optical propagation.

Implements the Hufnagel–Valley (HV 5/7) Cn² profile and derives:
  * Fried coherence length r₀
  * Rytov variance σ_R²
  * Turbulence-induced channel-transmission loss

References
----------
- Hufnagel & Stanley (1964), JOSA
- Andrews & Phillips (2005), *Laser Beam Propagation through Random Media*,
  SPIE Press
- Tyson (2015), *Principles of Adaptive Optics*, 4th ed., CRC Press
- Vasylyev et al., PRA 96, 043856 (2017)
"""

from __future__ import annotations

import numpy as np

from config.settings import V_WIND, A_GROUND


# ─────────────────────────────────────────────────────────────────────────────
# Cn² PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def Cn2_HV(
    h: np.ndarray,
    v_wind: float = V_WIND,
    a_ground: float = A_GROUND,
) -> np.ndarray:
    """
    Hufnagel–Valley 5/7 refractive-index structure parameter profile.

    Cn²(h) = 0.00594·(v/27)²·(10⁻⁵h)¹⁰·e^(−h/1000)
            + 2.7×10⁻¹⁶·e^(−h/1500)
            + A_ground·e^(−h/100)

    Parameters
    ----------
    h        : altitude array (metres)
    v_wind   : RMS high-altitude wind speed (m s⁻¹); default = HV 5/7 value
    a_ground : ground-layer Cn² strength (m⁻²/³)

    Returns
    -------
    np.ndarray : Cn²(h) in m⁻²/³, same shape as h
    """
    h = np.asarray(h, dtype=float)
    term1 = 0.00594 * (v_wind / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = a_ground * np.exp(-h / 100.0)
    return term1 + term2 + term3


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATED TURBULENCE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

def integrated_Cn2(
    elev_deg: float,
    v_wind: float = V_WIND,
    a_ground: float = A_GROUND,
    n_steps: int = 500,
    h_max_m: float = 20_000.0,
) -> float:
    """
    Slant-path-corrected vertical integral of Cn²:

        J = ∫₀^H Cn²(h) dh · sec(θ)

    where θ is the zenith angle.

    Parameters
    ----------
    elev_deg : satellite elevation (degrees)
    v_wind   : wind speed (m s⁻¹)
    a_ground : ground-layer strength
    n_steps  : number of quadrature points
    h_max_m  : upper integration limit (m)

    Returns
    -------
    float : J in m^(1/3)
    """
    if elev_deg <= 0:
        return 0.0

    h      = np.linspace(0.0, h_max_m, n_steps)
    Cn2    = Cn2_HV(h, v_wind, a_ground)
    J_vert = np.trapezoid(Cn2, h)

    sec_theta = 1.0 / np.sin(np.deg2rad(elev_deg))
    return float(J_vert * sec_theta)


# ─────────────────────────────────────────────────────────────────────────────
# FRIED PARAMETER
# ─────────────────────────────────────────────────────────────────────────────

def fried_parameter(
    lambda_m: float,
    J: float,
) -> float:
    """
    Fried coherence length r₀ for a slant path.

        r₀ = (0.423 · k² · J)^(−3/5)

    Parameters
    ----------
    lambda_m : wavelength (m)
    J        : slant-path Cn² integral (m^(1/3))

    Returns
    -------
    float : r₀ in metres
    """
    k = 2.0 * np.pi / lambda_m
    if J <= 0:
        return np.inf
    return float((0.423 * k ** 2 * J) ** (-3.0 / 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# RYTOV VARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def rytov_variance(
    lambda_m: float,
    J: float,
) -> float:
    """
    Plane-wave Rytov variance for weak turbulence.

        σ_R² = 1.23 · k^(7/6) · J

    Parameters
    ----------
    lambda_m : wavelength (m)
    J        : slant-path Cn² integral (m^(1/3))

    Returns
    -------
    float : σ_R² (dimensionless)
    """
    k = 2.0 * np.pi / lambda_m
    return float(1.23 * k ** (7.0 / 6.0) * J)


# ─────────────────────────────────────────────────────────────────────────────
# TURBULENCE TRANSMISSION AND LOSS
# ─────────────────────────────────────────────────────────────────────────────

def turbulence_eta(sigma_R2: float) -> float:
    """
    Turbulence-induced transmission (log-normal weak-turbulence regime).

        η_turb = exp(−σ_R²)

    Parameters
    ----------
    sigma_R2 : Rytov variance

    Returns
    -------
    float : η_turb ∈ (0, 1]
    """
    return float(np.exp(-np.clip(sigma_R2, 0.0, 50.0)))


def turbulence_loss_dB(
    lambda_m: float,
    elev_deg: float,
    v_wind: float = V_WIND,
    a_ground: float = A_GROUND,
) -> tuple[float, float, float]:
    """
    Compute turbulence loss (dB), Rytov variance, and Fried parameter r₀
    for a slant path at elevation ``elev_deg``.

    Parameters
    ----------
    lambda_m : wavelength (m)
    elev_deg : elevation angle (degrees)
    v_wind   : wind speed (m s⁻¹)
    a_ground : ground-layer strength

    Returns
    -------
    loss_dB  : turbulence loss (dB, positive = loss)
    sigma_R2 : Rytov variance
    r0       : Fried parameter (m)
    """
    J        = integrated_Cn2(elev_deg, v_wind, a_ground)
    sigma_R2 = rytov_variance(lambda_m, J)
    r0       = fried_parameter(lambda_m, J)
    eta      = turbulence_eta(sigma_R2)

    loss_dB  = -10.0 * np.log10(max(eta, 1e-30))
    return loss_dB, sigma_R2, r0
