"""
core
====
CAQC physics engine — atmospheric, turbulence, geometry, QKD, and
constellation modules.
"""

from .atmosphere   import (
    air_mass,
    rayleigh_optical_depth, rayleigh_loss_dB,
    aerosol_loss_dB, gas_loss_dB,
    slant_atm_loss_dB,
    itu_loss_dB,
    aod_at_wavelength,
)
from .turbulence   import (
    Cn2_HV,
    integrated_Cn2,
    fried_parameter,
    rytov_variance,
    turbulence_eta,
    turbulence_loss_dB,
)
from .geometry     import geometric_eta, geometric_loss_dB
from .qkd          import (
    binary_entropy,
    compute_qber,
    compute_skr_per_pulse,
    compute_skr_over_interval,
    loss_dB_to_eta,
)
from .constellation import generate_constellation, build_ground_stations, compute_passes
from .era5          import (
    download_era5_surface,
    download_era5_pressure,
    download_cams_aod,
    process_surface,
    extract_wind_profile,
    extract_aod550,
)

__all__ = [
    "air_mass",
    "rayleigh_optical_depth", "rayleigh_loss_dB",
    "aerosol_loss_dB", "gas_loss_dB",
    "slant_atm_loss_dB", "itu_loss_dB", "aod_at_wavelength",
    "Cn2_HV", "integrated_Cn2", "fried_parameter",
    "rytov_variance", "turbulence_eta", "turbulence_loss_dB",
    "geometric_eta", "geometric_loss_dB",
    "binary_entropy", "compute_qber",
    "compute_skr_per_pulse", "compute_skr_over_interval", "loss_dB_to_eta",
    "generate_constellation", "build_ground_stations", "compute_passes",
    "download_era5_surface", "download_era5_pressure", "download_cams_aod",
    "process_surface", "extract_wind_profile", "extract_aod550",
]
