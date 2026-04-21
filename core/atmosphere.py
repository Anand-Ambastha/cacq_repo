"""
core/atmosphere.py
==================
CAQC atmospheric-loss physics engine.

Implements wavelength-resolved channel-loss contributions from:
  * Rayleigh (molecular) scattering
  * Aerosol extinction  (Ångström scaling from AOD_550)
  * Mixed-gas absorption (O₂, CO₂, H₂O, N₂)
  * Air-mass / slant-path geometry

References
----------
- Bucholtz (1995), Applied Optics — Rayleigh coefficients
- Ångström (1929), Geografiska Annaler — aerosol spectral scaling
- Zuev (1982), Laser Beams in the Atmosphere
"""

from __future__ import annotations

import numpy as np

from config.settings import ALPHA_ANGSTROM


# ─────────────────────────────────────────────────────────────────────────────
# AIR MASS
# ─────────────────────────────────────────────────────────────────────────────

def air_mass(elev_deg: float) -> float:
    """
    Plane-parallel air-mass factor for elevation angle ``elev_deg``.

    M = 1 / sin(elev)

    Parameters
    ----------
    elev_deg : satellite elevation above horizon (degrees)

    Returns
    -------
    float : dimensionless air-mass factor (≥ 1)
    """
    return 1.0 / np.sin(np.deg2rad(elev_deg))


# ─────────────────────────────────────────────────────────────────────────────
# RAYLEIGH SCATTERING
# ─────────────────────────────────────────────────────────────────────────────

def rayleigh_optical_depth(
    lambda_m: float,
    P_hPa: float,
    T_C: float,
) -> float:
    """
    Zenith Rayleigh optical depth at wavelength ``lambda_m``.

    τ_R = 0.008569 · λ_µm⁻⁴ · (P / 1013) · (288 / (T + 273))

    Parameters
    ----------
    lambda_m : wavelength in metres
    P_hPa    : atmospheric pressure (hPa)
    T_C      : temperature (°C)

    Returns
    -------
    float : dimensionless zenith optical depth
    """
    lambda_um = lambda_m * 1e6
    return 0.008569 * lambda_um ** (-4) * (P_hPa / 1013.0) * (288.0 / (T_C + 273.0))


def rayleigh_loss_dB(
    lambda_m: float,
    P_hPa: float,
    T_C: float,
) -> float:
    """Zenith Rayleigh loss in dB (= 4.343 · τ_R)."""
    return 4.343 * rayleigh_optical_depth(lambda_m, P_hPa, T_C)


# ─────────────────────────────────────────────────────────────────────────────
# AEROSOL EXTINCTION
# ─────────────────────────────────────────────────────────────────────────────

def aod_at_wavelength(
    aod_550: float,
    lambda_m: float,
    alpha: float = ALPHA_ANGSTROM,
) -> float:
    """
    Ångström-scale AOD from 550 nm reference to target wavelength.

    AOD(λ) = AOD_550 · (λ_nm / 550)^(−α)

    Parameters
    ----------
    aod_550  : aerosol optical depth at 550 nm
    lambda_m : target wavelength (metres)
    alpha    : Ångström exponent

    Returns
    -------
    float : AOD at the target wavelength
    """
    lambda_nm = lambda_m * 1e9
    return aod_550 * (lambda_nm / 550.0) ** (-alpha)


def aerosol_loss_dB(aod_550: float, lambda_m: float) -> float:
    """Zenith aerosol loss in dB at ``lambda_m``."""
    return 4.343 * aod_at_wavelength(aod_550, lambda_m)


# ─────────────────────────────────────────────────────────────────────────────
# MIXED-GAS ABSORPTION
# ─────────────────────────────────────────────────────────────────────────────

def gas_optical_depth(P_hPa: float, RH_pct: float) -> float:
    """
    Combined zenith optical depth from O₂, CO₂, H₂O, N₂ absorption.

    Parameters
    ----------
    P_hPa  : pressure (hPa)
    RH_pct : relative humidity (0–100 %)

    Returns
    -------
    float : dimensionless zenith optical depth
    """
    tau_o2  = 0.00020 * (P_hPa / 1013.0)
    tau_co2 = 0.00005 * (P_hPa / 1013.0)
    tau_h2o = 0.00060 * (RH_pct / 100.0)
    tau_n2  = 0.00003 * (P_hPa / 1013.0)
    return tau_o2 + tau_co2 + tau_h2o + tau_n2


def gas_loss_dB(P_hPa: float, RH_pct: float) -> float:
    """Zenith gas-absorption loss in dB."""
    return 4.343 * gas_optical_depth(P_hPa, RH_pct)


# ─────────────────────────────────────────────────────────────────────────────
# SLANT-PATH TOTAL ATMOSPHERIC LOSS
# ─────────────────────────────────────────────────────────────────────────────

def slant_atm_loss_dB(
    lambda_m: float,
    elev_deg: float,
    P_hPa: float,
    T_C: float,
    RH_pct: float,
    aod_550: float,
) -> float:
    """
    Total atmospheric channel loss (dB) along a slant path.

    L_atm = M · (L_Rayleigh + L_Aerosol + L_Gas)

    where M is the air-mass factor.

    Parameters
    ----------
    lambda_m : wavelength (m)
    elev_deg : elevation angle (degrees)
    P_hPa    : surface pressure (hPa)
    T_C      : surface temperature (°C)
    RH_pct   : relative humidity (%)
    aod_550  : AOD at 550 nm

    Returns
    -------
    float : slant-path atmospheric loss (dB, positive = loss)
    """
    M = air_mass(elev_deg)
    L_R = rayleigh_loss_dB(lambda_m, P_hPa, T_C)
    L_A = aerosol_loss_dB(aod_550, lambda_m)
    L_G = gas_loss_dB(P_hPa, RH_pct)
    return M * (L_R + L_A + L_G)


# ─────────────────────────────────────────────────────────────────────────────
# ITU-R P.1814 REFERENCE MODEL (comparison baseline)
# ─────────────────────────────────────────────────────────────────────────────

def itu_loss_dB(
    distance_km: float,
    lambda_m: float,
    visibility_km: float = 23.0,
    q: float = 1.3,
) -> float:
    """
    ITU-R P.1814-inspired aerosol extinction along a slant path.

    β(λ) = (3.91 / V) · (λ_µm / 0.55)^(−q)
    L     = −10 log₁₀[ exp(−β · d) ]

    Parameters
    ----------
    distance_km   : slant-path length (km)
    lambda_m      : wavelength (m)
    visibility_km : meteorological visibility (km)
    q             : Ångström-like size exponent

    Returns
    -------
    float : loss (dB)
    """
    lambda_um = lambda_m * 1e6
    beta = (3.91 / visibility_km) * (lambda_um / 0.55) ** (-q)
    return -10.0 * np.log10(np.exp(-beta * distance_km))
