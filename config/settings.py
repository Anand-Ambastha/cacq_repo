"""
config/settings.py
==================
Global configuration for the CAQC pipeline.
All physical constants, instrument parameters, and site definitions live here.
"""

from __future__ import annotations

# ─────────────────────────────────────────────
#  WAVELENGTH CONFIGURATION
# ─────────────────────────────────────────────
WAVELENGTH_NM      = 785          # primary wavelength (nm)
WAVELENGTH_NM_ALT  = 1550         # secondary / telecom wavelength (nm)

LAMBDA_UM   = WAVELENGTH_NM / 1000.0          # µm (used in Rayleigh formula)
LAMBDA_M    = WAVELENGTH_NM * 1e-9            # metres
LAMBDA_M_ALT = WAVELENGTH_NM_ALT * 1e-9

# ─────────────────────────────────────────────
#  TEMPORAL RANGE
# ─────────────────────────────────────────────
START_YEAR = 2015
END_YEAR   = 2024

# ─────────────────────────────────────────────
#  OPTICAL / QKD SYSTEM PARAMETERS
# ─────────────────────────────────────────────
D_RX       = 0.50    # receiver aperture diameter (m)
D_TX       = 0.10    # transmitter aperture diameter (m)
THETA_DIV  = 10e-6   # full-angle beam divergence (rad)
MU_PHOTON  = 0.50    # mean photon number per pulse
F_EC       = 1.16    # error-correction efficiency factor
MIN_ELEV   = 10      # minimum satellite elevation angle (degrees)
DT         = 300     # integration time step (seconds)

# ─────────────────────────────────────────────
#  TURBULENCE MODEL PARAMETERS (Hufnagel–Valley 5/7)
# ─────────────────────────────────────────────
V_WIND    = 21       # RMS high-altitude wind speed (m/s)
A_GROUND  = 1.7e-14  # ground-layer Cn² coefficient

# ─────────────────────────────────────────────
#  AEROSOL ÅNGSTRÖM EXPONENT
# ─────────────────────────────────────────────
ALPHA_ANGSTROM = 1.3

# ─────────────────────────────────────────────
#  GROUND STATION COORDINATES
#  Format: { name: (lat_deg, lon_deg) }
# ─────────────────────────────────────────────
GROUND_STATIONS: dict[str, tuple[float, float]] = {
    "Hanle":      (32.78, 78.96),
    "Dehradun":   (30.30, 78.00),
    "MtAbu":      (24.60, 72.70),
    "Shillong":   (25.60, 91.80),
    "Kodaikanal": (10.20, 77.40),
}

# ─────────────────────────────────────────────
#  SATELLITE CONSTELLATION PARAMETERS
# ─────────────────────────────────────────────
CONSTELLATION = {
    "n_planes"    : 3,
    "sats_per_plane": 4,
    "altitude_km" : 600,
    "inclination" : 97.5,   # degrees (Sun-synchronous)
    "eccentricity": 0.0002,
    "arg_perigee" : 90.0,
}

# ─────────────────────────────────────────────
#  OUTPUT DIRECTORIES
# ─────────────────────────────────────────────
PLOT_DIR  = "outputs/plots"
TABLE_DIR = "outputs/tables"
FIG_DPI   = 600

# ─────────────────────────────────────────────
#  CDS API CONFIGURATION
#  Set via environment variable: CDSAPI_KEY
# ─────────────────────────────────────────────
import os

CDS_URL        = "https://cds.climate.copernicus.eu/api"
CDS_KEY        = os.getenv("CDSAPI_KEY", "")          # export CDSAPI_KEY=<your-key>

CAMS_URL       = "https://ads.atmosphere.copernicus.eu/api"
CAMS_KEY       = os.getenv("CDSAPI_KEY", "")

# ─────────────────────────────────────────────
#  SEASON DEFINITIONS (month ranges)
# ─────────────────────────────────────────────
SEASONS: dict[str, list[int]] = {
    "Winter"   : [12, 1, 2],
    "Pre-Monsoon": [3, 4, 5],
    "Monsoon"  : [6, 7, 8, 9],
    "Post-Monsoon": [10, 11],
}
