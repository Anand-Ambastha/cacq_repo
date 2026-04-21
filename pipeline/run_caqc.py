"""
pipeline/run_caqc.py
====================
Main CAQC simulation pipeline.

Entry point for the full end-to-end run:
  1. Download / load ERA5 + CAMS data for each ground station
  2. Iterate over the satellite constellation × time steps
  3. Compute per-pass loss budget (geometric + atmospheric + turbulence)
  4. Aggregate secure key yield per city
  5. Build the master results DataFrame

Usage
-----
    from pipeline.run_caqc import run_pipeline
    master_df = run_pipeline()
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from skyfield.api import load

from config.settings import (
    GROUND_STATIONS,
    LAMBDA_M, LAMBDA_M_ALT,
    D_RX, D_TX,
    MIN_ELEV, DT,
    ALPHA_ANGSTROM,
)
from core import (
    download_era5_surface, download_era5_pressure, download_cams_aod,
    process_surface, extract_wind_profile, extract_aod550,
    rayleigh_loss_dB, aerosol_loss_dB, gas_loss_dB,
    geometric_loss_dB,
    Cn2_HV, integrated_Cn2, fried_parameter, rytov_variance, turbulence_eta,
    compute_skr_per_pulse, loss_dB_to_eta, aod_at_wavelength,
    itu_loss_dB,
    generate_constellation, build_ground_stations,
)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_eta(etas: np.ndarray) -> float:
    """
    Weighted average efficiency (elevation-weighted harmonic mean proxy).

        η̄ = Σ η² / Σ η
    """
    etas = np.asarray(etas)
    if len(etas) == 0:
        return 0.0
    return float(np.sum(etas ** 2) / np.sum(etas))


def _standard_model_loss(
    geom_dB: float,
    lambda_m: float,
    M: float,
    rayleigh: float,
) -> float:
    """
    Standard atmosphere reference model loss (fixed AOD = 0.05,
    fixed turbulence term).
    """
    aod_std     = 0.05
    aer_std     = aerosol_loss_dB(aod_std, lambda_m)
    turb_std_dB = 10.0 * np.log10(1.0 + (D_RX / 0.1) ** 2)
    return geom_dB + M * (rayleigh + aer_std) + turb_std_dB


# ─────────────────────────────────────────────────────────────────────────────
# PER-CITY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_city(
    city: str,
    lat: float,
    lon: float,
    satellites: list,
    ground_stations_sf: dict,
    times,
    cache_dir: str = ".",
    verbose: bool = True,
) -> dict:
    """
    Run the full CAQC budget for one ground station.

    Parameters
    ----------
    city              : station label
    lat, lon          : decimal-degree coordinates
    satellites        : list of Skyfield EarthSatellite objects
    ground_stations_sf: Skyfield wgs84 position objects keyed by name
    times             : Skyfield time array
    cache_dir         : directory for ERA5 cache files
    verbose           : print progress

    Returns
    -------
    dict of per-city results (scalars and component losses)
    """
    if verbose:
        print(f"\n[CAQC] Processing {city} ...")

    gs = ground_stations_sf[city]

    # ── ERA5 surface ──────────────────────────────────────────────
    ds_sfc = process_surface(download_era5_surface(city, lat, lon, cache_dir))
    ds_plv = download_era5_pressure(city, lat, lon, cache_dir)

    time_coord = "time" if "time" in ds_sfc.coords else "valid_time"

    temp  = float(ds_sfc["t2m"].mean())
    rh    = float(ds_sfc["rh"].mean())
    P_hPa = float(ds_sfc["msl"].mean())

    # ── AOD ───────────────────────────────────────────────────────
    ds_aod      = download_cams_aod(city, lat, lon, cache_dir)
    AOD_550, _  = extract_aod550(ds_aod)

    AOD_785  = aod_at_wavelength(AOD_550, LAMBDA_M,     ALPHA_ANGSTROM)
    AOD_1550 = aod_at_wavelength(AOD_550, LAMBDA_M_ALT, ALPHA_ANGSTROM)

    aer_785  = aerosol_loss_dB(AOD_550, LAMBDA_M)
    aer_1550 = aerosol_loss_dB(AOD_550, LAMBDA_M_ALT)

    # ── Rayleigh ──────────────────────────────────────────────────
    ray_785  = rayleigh_loss_dB(LAMBDA_M,     P_hPa, temp)
    ray_1550 = rayleigh_loss_dB(LAMBDA_M_ALT, P_hPa, temp)

    # ── Gas absorption ────────────────────────────────────────────
    mol = gas_loss_dB(P_hPa, rh)

    # ── Wind profile ──────────────────────────────────────────────
    v_eff, v_high = extract_wind_profile(ds_plv, time_coord)

    # ── ITU-R reference (constant path = 500 km proxy) ───────────
    beta_785  = (3.91 / 23.0) * (LAMBDA_M    * 1e6 / 0.55) ** (-1.3)
    beta_1550 = (3.91 / 23.0) * (LAMBDA_M_ALT * 1e6 / 0.55) ** (-1.3)

    # ── Pass-loop accumulators ────────────────────────────────────
    losses_785,  losses_1550  = [], []
    losses_std_785, losses_std_1550 = [], []
    losses_itu_785, losses_itu_1550 = [], []
    geom_list_785, geom_list_1550   = [], []
    skr_785_list, skr_1550_list     = [], []
    elev_list, loss_elev_785        = [], []

    for sat in satellites:
        for t in times:
            topo = (sat - gs).at(t)
            elev = topo.altaz()[0].degrees
            rng  = topo.distance().km

            if elev < max(MIN_ELEV, 10.0):
                continue

            sin_e = np.sin(np.deg2rad(elev))
            if sin_e < 0.05:
                continue

            L   = rng * 1000.0          # km → m
            M   = 1.0 / sin_e

            # ── Geometry ──────────────────────────────────────
            g785  = geometric_loss_dB(L, LAMBDA_M,     D_RX, D_TX)
            g1550 = geometric_loss_dB(L, LAMBDA_M_ALT, D_RX, D_TX)
            geom_list_785.append(g785)
            geom_list_1550.append(g1550)

            # ── Atmosphere (slant) ────────────────────────────
            atm_785  = M * (ray_785  + aer_785  + mol)
            atm_1550 = M * (ray_1550 + aer_1550 + mol)

            # ── Turbulence ────────────────────────────────────
            h     = np.linspace(0.0, 20_000.0, 500)
            Cn2   = Cn2_HV(h, v_eff)
            J     = float(np.trapezoid(Cn2, h)) / sin_e

            from math import pi
            k_785  = 2.0 * pi / LAMBDA_M
            k_1550 = 2.0 * pi / LAMBDA_M_ALT

            sr785  = float(1.23 * k_785  ** (7.0 / 6.0) * J)
            sr1550 = float(1.23 * k_1550 ** (7.0 / 6.0) * J)

            eta_t785  = float(np.exp(-np.clip(sr785,  0, 50)))
            eta_t1550 = float(np.exp(-np.clip(sr1550, 0, 50)))

            turb_785  = -10.0 * np.log10(max(eta_t785,  1e-30))
            turb_1550 = -10.0 * np.log10(max(eta_t1550, 1e-30))

            # ── Fried parameters (last pass, stored once) ─────
            r0_785  = (0.423 * k_785  ** 2 * J) ** (-3.0 / 5.0)
            r0_1550 = (0.423 * k_1550 ** 2 * J) ** (-3.0 / 5.0)

            # ── Total CAQC ────────────────────────────────────
            tot785  = g785  + atm_785  + turb_785
            tot1550 = g1550 + atm_1550 + turb_1550

            losses_785.append(tot785)
            losses_1550.append(tot1550)
            elev_list.append(elev)
            loss_elev_785.append(tot785)

            # ── Standard model ────────────────────────────────
            std785  = _standard_model_loss(g785,  LAMBDA_M,     M, ray_785)
            std1550 = _standard_model_loss(g1550, LAMBDA_M_ALT, M, ray_1550)
            losses_std_785.append(std785)
            losses_std_1550.append(std1550)

            # ── ITU model ─────────────────────────────────────
            L_slant_km = rng / sin_e
            itu785  = g785  + beta_785  * L_slant_km
            itu1550 = g1550 + beta_1550 * L_slant_km
            losses_itu_785.append(itu785)
            losses_itu_1550.append(itu1550)

            # ── SKR ───────────────────────────────────────────
            eta_lin_785  = loss_dB_to_eta(tot785)
            eta_lin_1550 = loss_dB_to_eta(tot1550)
            skr_785_list.append(compute_skr_per_pulse(eta_lin_785)  * DT)
            skr_1550_list.append(compute_skr_per_pulse(eta_lin_1550) * DT)

    if not losses_785:
        if verbose:
            print(f"  [WARNING] No valid passes found for {city}!")
        return {}

    # ── Channel aggregation ───────────────────────────────────────
    def _agg(arr):
        etas = np.array([loss_dB_to_eta(x) for x in arr])
        eta_w = _weighted_eta(etas)
        return float(-10.0 * np.log10(max(eta_w, 1e-30))), eta_w

    tot785_agg,  eta785  = _agg(losses_785)
    tot1550_agg, eta1550 = _agg(losses_1550)
    std785_agg,  _       = _agg(losses_std_785)
    std1550_agg, _       = _agg(losses_std_1550)
    itu785_agg,  _       = _agg(losses_itu_785)
    itu1550_agg, _       = _agg(losses_itu_1550)

    def _err(caqc, ref):
        return abs((ref - caqc) / caqc) * 100.0 if caqc > 0 else 0.0

    return {
        "City"            : city,
        "Temp_C"          : round(temp,  3),
        "RH_%"            : round(rh,    3),
        "Pressure_hPa"    : round(P_hPa, 3),
        "AOD_550"         : round(AOD_550,    4),
        "HighWind"        : round(v_high,     3),

        "CAQC_785"        : round(tot785_agg,  4),
        "CAQC_1550"       : round(tot1550_agg, 4),

        "Standard_785_dB" : round(std785_agg,  4),
        "Standard_1550_dB": round(std1550_agg, 4),

        "ITU_785_dB"      : round(itu785_agg,  4),
        "ITU_1550_dB"     : round(itu1550_agg, 4),

        "Error_785_%"     : round(_err(tot785_agg,  std785_agg),  3),
        "Error_1550_%"    : round(_err(tot1550_agg, std1550_agg), 3),
        "Err_ITU_785_%"   : round(_err(tot785_agg,  itu785_agg),  3),
        "Err_ITU_1550_%"  : round(_err(tot1550_agg, itu1550_agg), 3),

        "Eta_785"         : eta785,
        "Eta_1550"        : eta1550,

        "SKR_785"         : round(float(np.sum(skr_785_list)),  2),
        "SKR_1550"        : round(float(np.sum(skr_1550_list)), 2),

        "Geom_785_dB"     : round(float(np.mean(geom_list_785)),  4),
        "Geom_1550_dB"    : round(float(np.mean(geom_list_1550)), 4),

        "Rayleigh_785_dB" : round(ray_785,   4),
        "Aerosol_785_dB"  : round(aer_785,   4),
        "Turbulence_785_dB": round(turb_785, 4),

        "Rayleigh_1550_dB": round(ray_1550,   4),
        "Aerosol_1550_dB" : round(aer_1550,   4),
        "Turbulence_1550_dB": round(turb_1550, 4),

        "Fried_r0_785"    : round(r0_785,  5),
        "Fried_r0_1550"   : round(r0_1550, 5),

        # Extra derived columns useful for plots / tables
        "Total_Loss_dB"          : round(tot785_agg,  4),
        "Standard_Model_Loss_dB" : round(std785_agg,  4),
        "Link_Availability_%"    : round(100.0 * len(losses_785) /
                                         max(len(losses_785) + 1, 1), 2),
        "Extra_Loss_dB"          : round(max(0.0, tot785_agg - std785_agg), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    station_overrides: dict | None = None,
    cache_dir: str = ".",
    sim_date: tuple[int, int, int] = (2022, 1, 1),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full CAQC pipeline for all (or a subset of) ground stations.

    Parameters
    ----------
    station_overrides : replace default station dict {name:(lat,lon)}
    cache_dir         : directory to cache ERA5 files
    sim_date          : (year, month, day) for satellite pass computation
    verbose           : print progress messages

    Returns
    -------
    pd.DataFrame : master results table (one row per station)
    """
    stations = station_overrides or GROUND_STATIONS
    os.makedirs(cache_dir, exist_ok=True)

    # ── Build constellation ───────────────────────────────────────
    satellites      = generate_constellation()
    ground_sf       = build_ground_stations(stations)

    # ── Build time array ──────────────────────────────────────────
    ts      = load.timescale()
    minutes = list(range(0, 24 * 60, DT // 60))
    times   = ts.utc(*sim_date, 0, minutes)

    # ── Process each station ──────────────────────────────────────
    rows = []
    for city, (lat, lon) in stations.items():
        result = process_city(
            city, lat, lon,
            satellites, ground_sf, times,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        if result:
            rows.append(result)

    if not rows:
        raise RuntimeError("No results generated — check ERA5 credentials and station list.")

    master_df = pd.DataFrame(rows).reset_index(drop=True)

    if verbose:
        print("\n[CAQC] Pipeline complete.")
        print(master_df[["City", "CAQC_785", "CAQC_1550", "SKR_785", "SKR_1550"]].to_string(index=False))

    return master_df
