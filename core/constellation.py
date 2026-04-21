"""
core/constellation.py
=====================
Synthetic LEO satellite constellation generator.

Builds a Walker-delta constellation using Two-Line Element (TLE)
format compatible with the Skyfield propagator.

The constellation parameters are read from ``config.settings.CONSTELLATION``.
"""

from __future__ import annotations

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

from config.settings import CONSTELLATION, GROUND_STATIONS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mean_motion(altitude_km: float) -> float:
    """
    Compute mean motion n in revolutions per day.

        n = (1 / 2π) · √(µ / a³)  [rad s⁻¹] → converted to rev day⁻¹
    """
    mu = 398600.4418          # km³ s⁻²
    Re = 6378.137             # km
    a  = Re + altitude_km     # semi-major axis (km)
    return np.sqrt(mu / a ** 3) * 86400.0 / (2.0 * np.pi)


def _tle_pair(
    plane: int,
    sat_idx: int,
    n_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination: float,
    eccentricity: float,
    arg_perigee: float,
) -> tuple[str, str]:
    """Generate a TLE line-1 / line-2 pair for a single satellite."""
    sat_num = plane * sats_per_plane + sat_idx + 1
    raan    = plane * (360.0 / n_planes)
    ma      = sat_idx * (360.0 / sats_per_plane)
    mm      = _mean_motion(altitude_km)
    ecc_str = f"{eccentricity:.7f}".split(".")[1]   # 7-digit mantissa only

    line1 = (
        f"1 {sat_num:05d}U 25001A   25001.00000000  .00000000 "
        f" 00000-0  00000-0 0  999{sat_num % 10}"
    )
    line2 = (
        f"2 {sat_num:05d} {inclination:8.4f} {raan:8.4f} "
        f"{ecc_str} {arg_perigee:8.4f} {ma:8.4f} {mm:.8f}    01"
    )
    return line1, line2


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_constellation(
    params: dict | None = None,
) -> list[EarthSatellite]:
    """
    Build a list of Skyfield EarthSatellite objects for the constellation.

    Parameters
    ----------
    params : constellation parameter dict (defaults to config.CONSTELLATION)

    Returns
    -------
    list[EarthSatellite]
    """
    if params is None:
        params = CONSTELLATION

    ts = load.timescale()

    n_planes       = params["n_planes"]
    sats_per_plane = params["sats_per_plane"]
    altitude_km    = params["altitude_km"]
    inclination    = params["inclination"]
    eccentricity   = params["eccentricity"]
    arg_perigee    = params["arg_perigee"]

    satellites: list[EarthSatellite] = []
    for plane in range(n_planes):
        for sat_idx in range(sats_per_plane):
            l1, l2 = _tle_pair(
                plane, sat_idx,
                n_planes, sats_per_plane,
                altitude_km, inclination,
                eccentricity, arg_perigee,
            )
            name = f"SAT-P{plane}-S{sat_idx}"
            satellites.append(EarthSatellite(l1, l2, name, ts))

    return satellites


def build_ground_stations(
    station_dict: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """
    Convert lat/lon coordinate pairs to Skyfield wgs84 GeographicPosition
    objects.

    Parameters
    ----------
    station_dict : {name: (lat, lon)} mapping

    Returns
    -------
    dict mapping station name → Skyfield GeographicPosition
    """
    if station_dict is None:
        station_dict = GROUND_STATIONS
    return {name: wgs84.latlon(lat, lon) for name, (lat, lon) in station_dict.items()}


def compute_passes(
    satellite: EarthSatellite,
    ground_station,
    times,
    min_elev_deg: float = 10.0,
) -> list[dict]:
    """
    Compute elevation and range for all time steps above ``min_elev_deg``.

    Parameters
    ----------
    satellite      : Skyfield EarthSatellite
    ground_station : Skyfield wgs84 position
    times          : Skyfield time array
    min_elev_deg   : minimum elevation cut-off (degrees)

    Returns
    -------
    list of dict with keys: elev_deg, range_km, time
    """
    passes = []
    for t in times:
        topo = (satellite - ground_station).at(t)
        elev = topo.altaz()[0].degrees
        rng  = topo.distance().km
        if elev >= min_elev_deg:
            passes.append({"elev_deg": elev, "range_km": rng, "time": t})
    return passes
