"""
core/era5.py
============
ERA5 / CAMS reanalysis download and pre-processing.

All downloads are cached as NetCDF files; if the file already exists
the download step is skipped, making repeated runs fast.
"""

from __future__ import annotations

import os
import numpy as np
import xarray as xr

from config.settings import (
    START_YEAR, END_YEAR,
    CDS_URL, CDS_KEY,
    CAMS_URL, CAMS_KEY,
)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bbox(lat: float, lon: float, pad: float = 0.25) -> list[float]:
    """Return [N, W, S, E] bounding box aligned to ERA5 0.25° grid."""
    return [
        round(lat + pad, 2),
        round(lon - pad, 2),
        round(lat - pad, 2),
        round(lon + pad, 2),
    ]


def _rename_time(ds: xr.Dataset) -> xr.Dataset:
    """Normalise CDS naming quirks (valid_time → time)."""
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError(
            f"No 'time' coordinate found. Available: {list(ds.coords)}"
        )
    return ds


def _make_cds_client(url: str, key: str):
    """Construct a cdsapi.Client with explicit credentials."""
    import cdsapi
    return cdsapi.Client(url=url, key=key)


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE (SINGLE-LEVEL) DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_era5_surface(
    city: str,
    lat: float,
    lon: float,
    cache_dir: str = ".",
) -> xr.Dataset:
    """
    Download ERA5 monthly-mean surface fields (T2m, Td2m, MSLP) for
    ``city`` over the configured year range.

    Parameters
    ----------
    city      : station label (used as file prefix)
    lat, lon  : decimal degrees
    cache_dir : directory where .nc files are cached

    Returns
    -------
    xr.Dataset with variables t2m, d2m, msl
    """
    out = os.path.join(cache_dir, f"{city}_era5_surface.nc")
    if os.path.exists(out):
        return _rename_time(xr.open_dataset(out))

    years  = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    c = _make_cds_client(CDS_URL, CDS_KEY)
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "mean_sea_level_pressure",
            ],
            "year": years,
            "month": months,
            "time": "00:00",
            "area": _bbox(lat, lon),
            "data_format": "netcdf",
        },
        out,
    )
    return _rename_time(xr.open_dataset(out))


# ─────────────────────────────────────────────────────────────────────────────
# PRESSURE-LEVEL DOWNLOAD (wind profiles)
# ─────────────────────────────────────────────────────────────────────────────

_PRESSURE_LEVELS = [
    "1000","975","950","925","900","875","850","825","800","775",
    "750","700","650","600","550","500","450","400","350","300",
    "250","225","200","175","150","125","100","70","50","30",
    "20","10","7","5","3","2","1",
]


def download_era5_pressure(
    city: str,
    lat: float,
    lon: float,
    cache_dir: str = ".",
) -> xr.Dataset:
    """
    Download ERA5 monthly-mean pressure-level fields (T, u, v, z).

    Returns
    -------
    xr.Dataset with variables t, u, v, z on pressure_level coordinates
    """
    out = os.path.join(cache_dir, f"{city}_era5_pressure.nc")
    if os.path.exists(out):
        return _rename_time(xr.open_dataset(out))

    years  = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    c = _make_cds_client(CDS_URL, CDS_KEY)
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "geopotential",
            ],
            "pressure_level": _PRESSURE_LEVELS,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": _bbox(lat, lon),
            "data_format": "netcdf",
        },
        out,
    )
    return _rename_time(xr.open_dataset(out))


# ─────────────────────────────────────────────────────────────────────────────
# CAMS AOD DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_cams_aod(
    city: str,
    lat: float,
    lon: float,
    cache_dir: str = ".",
) -> xr.Dataset:
    """
    Download CAMS EAC4 total AOD at 550 nm for a short representative
    period around the simulation date.

    Returns
    -------
    xr.Dataset with variable ``aod550`` or
    ``total_aerosol_optical_depth_550nm``
    """
    out = os.path.join(cache_dir, f"{city}_cams_aod.nc")
    if os.path.exists(out):
        return xr.open_dataset(out)

    c = _make_cds_client(CAMS_URL, CAMS_KEY)
    c.retrieve(
        "cams-global-reanalysis-eac4",
        {
            "variable"   : "total_aerosol_optical_depth_550nm",
            "date"       : "2022-01-01/2022-01-05",
            "time"       : "12:00",
            "format"     : "netcdf",
            "area"       : [lat + 1.5, lon - 1.5, lat - 1.5, lon + 1.5],
        },
        out,
    )
    return xr.open_dataset(out)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_surface(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert ERA5 surface dataset to physically meaningful quantities.

    * t2m : 2-m air temperature  (K  → °C)
    * d2m : 2-m dew-point        (K  → °C)
    * rh  : relative humidity    (0–100 %)
    * msl : mean sea-level press (Pa → hPa)
    """
    ds = ds.copy()
    ds["t2m"] = ds["t2m"] - 273.15
    ds["d2m"] = ds["d2m"] - 273.15

    # August–Magnus formula
    es = 6.11 * 10 ** ((7.5 * ds["t2m"]) / (237.3 + ds["t2m"]))
    e  = 6.11 * 10 ** ((7.5 * ds["d2m"]) / (237.3 + ds["d2m"]))
    ds["rh"] = 100.0 * (e / es)

    if ds["msl"].values.max() > 10_000:          # still in Pa
        ds["msl"] = ds["msl"] / 100.0

    return ds


def extract_wind_profile(
    ds_pressure: xr.Dataset,
    time_coord: str = "time",
) -> tuple[float, float]:
    """
    Derive effective wind speed metrics from ERA5 pressure-level data.

    Returns
    -------
    v_eff   : altitude-weighted effective wind speed (m s⁻¹)
    v_high  : mean wind speed above 10 km           (m s⁻¹)
    """
    u  = ds_pressure["u"].mean(dim=[time_coord, "latitude", "longitude"]).values
    v  = ds_pressure["v"].mean(dim=[time_coord, "latitude", "longitude"]).values
    z  = ds_pressure["z"].mean(dim=[time_coord, "latitude", "longitude"]).values
    z_m = z / 9.80665                               # geopotential height → metres

    wind    = np.sqrt(u ** 2 + v ** 2)
    weights = np.exp(-z_m / 8000.0)
    v_eff   = float(np.sum(wind * weights) / np.sum(weights))
    v_eff   = float(np.clip(v_eff, 5.0, 25.0))

    mask_hi = z_m > 10_000
    v_high  = float(np.mean(wind[mask_hi])) if mask_hi.any() else v_eff

    return v_eff, v_high


def extract_aod550(ds_aod: xr.Dataset) -> tuple[float, np.ndarray]:
    """
    Return the scalar mean AOD at 550 nm and the full spatial array.

    Handles both CDS variable naming conventions.
    """
    key = (
        "aod550"
        if "aod550" in ds_aod
        else "total_aerosol_optical_depth_550nm"
    )
    arr     = ds_aod[key].values
    aod_mean = float(ds_aod[key].mean())
    return aod_mean, arr.flatten()
