# CAQC — Climate-Aware Quantum Channel Model

> **Manuscript under review.** This repository accompanies a paper submitted for peer-reviewed publication. All rights reserved — see [LICENSE](LICENSE).

---

## Overview

CAQC (*Climate-Aware Quantum Channel*) is a physics-consistent simulation framework for evaluating satellite-to-ground **free-space quantum key distribution (QKD)** under realistic atmospheric conditions derived from ERA5 / CAMS reanalysis data.

The model computes wavelength-resolved channel loss from four physically distinct contributions:

| Component | Model |
|-----------|-------|
| Rayleigh scattering | Bucholtz (1995) formula, pressure/temperature corrected |
| Aerosol extinction | Ångström spectral scaling from CAMS AOD at 550 nm |
| Mixed-gas absorption | O₂, CO₂, H₂O, N₂ |
| Atmospheric turbulence | Hufnagel–Valley 5/7 Cn² profile + Rytov/Fried |
| Geometric spreading | Gaussian beam propagation, finite aperture coupling |

QKD performance (QBER, SKR) is computed using the decoy-state BB84 framework.

---
## Key Features

- Physics-consistent satellite-to-ground QKD channel model  
- Climate-aware attenuation using ERA5 / CAMS reanalysis data  
- Wavelength-resolved analysis (785 nm vs 1550 nm)  
- Multi-site evaluation across geographically diverse stations  
- Modular architecture for research extensibility  
- CLI and Streamlit dashboard for interactive exploration  

---
## Use Cases

- Site selection for ground-based QKD stations  
- Sensitivity analysis of atmospheric effects (AOD, turbulence, elevation)  
- Performance benchmarking under realistic climate conditions  
- Pre-deployment feasibility studies for satellite QKD missions  

---
## Repository Structure

```
caqc/
├── config/
│   └── settings.py          # All constants, system params, station coords
├── core/
│   ├── atmosphere.py        # Rayleigh, aerosol, gas, slant-path loss
│   ├── turbulence.py        # HV 5/7, Cn², Fried r₀, Rytov variance
│   ├── geometry.py          # Gaussian beam propagation, aperture coupling
│   ├── qkd.py               # BB84 decoy-state SKR / QBER model
│   ├── constellation.py     # Walker-delta LEO constellation (Skyfield TLE)
│   └── era5.py              # ERA5 / CAMS download + processing
├── pipeline/
│   └── run_caqc.py          # Full end-to-end pipeline
├── plots/
│   └── visualise.py         # Publication-quality figure generation
├── tables/
│   └── export.py            # CSV + LaTeX table export
├── dashboard/
│   └── app.py               # Interactive Streamlit dashboard
├── cli/
│   └── simulate.py          # Command-line simulation tool
├── tests/
│   └── test_core.py         # Unit tests (pytest)
├── outputs/
│   ├── plots/               # Generated figures
│   └── tables/              # Generated tables
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For ERA5 GRIB support (Ubuntu/Debian):
apt-get install -y libeccodes-dev
```

### 2. Set ERA5 credentials

```bash
export CDSAPI_KEY="your-cds-api-key-here"
```

### 3. Run the CLI demo (no ERA5 required)

```bash
# Physics demo with synthetic parameters
python -m cli.simulate --mode demo

# Single link budget
python -m cli.simulate --mode link --elev 35 --range 700 --aod 0.20

# AOD parameter sweep
python -m cli.simulate --mode sweep --param aod

# Elevation sweep
python -m cli.simulate --mode sweep --param elev
```

### 4. Launch the interactive dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides five interactive pages:
- **Link Budget** — real-time slant-path budget calculator
- **Parameter Sweep** — loss/SKR vs any atmospheric parameter
- **Turbulence Explorer** — HV 5/7 Cn² with r₀ heatmap
- **Cn² Profile** — multi-site turbulence profile viewer
- **Results Viewer** — upload pipeline output CSV for full figure suite

### 5. Run the full pipeline (requires ERA5 credentials)

```python
from pipeline.run_caqc import run_pipeline
from plots.visualise   import generate_all
from tables.export     import export_all

master_df = run_pipeline()
generate_all(master_df)
export_all(master_df)
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Physical Model Details

### Atmospheric Loss

The total slant-path atmospheric loss at wavelength λ is:

```
L_atm = M(θ) × [ L_Rayleigh(λ) + L_Aerosol(λ) + L_Gas ]
```

where `M(θ) = 1/sin(θ)` is the plane-parallel air-mass factor.

**Rayleigh:**
```
τ_R = 0.008569 × λ_µm⁻⁴ × (P/1013) × (288/(T+273))
```

**Aerosol (Ångström):**
```
AOD(λ) = AOD_550 × (λ_nm / 550)^(−α),   α = 1.3
```

### Turbulence (Hufnagel–Valley 5/7)

```
Cn²(h) = 0.00594·(v/27)²·(10⁻⁵h)¹⁰·e^(-h/1000)
        + 2.7×10⁻¹⁶·e^(-h/1500)
        + A_ground·e^(-h/100)
```

Fried parameter:  `r₀ = (0.423 k² J)^(-3/5)`

Rytov variance:   `σ_R² = 1.23 k^(7/6) J`

Turbulence transmission (weak regime):  `η_turb = exp(−σ_R²)`

### QKD (Decoy-state BB84)

```
Q_µ  = η × µ × exp(−µ)
QBER = min(0.5,  0.02 + 0.1·(1−η))
SKR  = max(0,  Q_µ · [1 − f_EC · h(QBER)])
```

---

## Ground Stations

| Station | Lat (°N) | Lon (°E) |
|---------|----------|----------|
| Hanle | 32.78 | 78.96 |
| Dehradun | 30.30 | 78.00 |
| Mt. Abu | 24.60 | 72.70 |
| Shillong | 25.60 | 91.80 |
| Kodaikanal | 10.20 | 77.40 |

### Actual Screenshots
<p align="center">
  <img src="assets\p1.png" width="45%" />
  <img src="assests\p2.png" width="45%" />
</p>

<p align="center">
  <img src="assets\summary.png" width="70%" />
</p>

### ⚙️ Simulation Backend Note

This GitHub version of the project uses a **Keplerian orbital mechanics model** for satellite trajectory propagation.

The results presented in the associated research paper were generated using the **Skyfield-based ephemeris model**, which incorporates high-precision astronomical data (e.g., JPL DE ephemerides) and accounts for perturbations.

#### Key Difference

* **Keplerian Model (GitHub)**

  * Analytical, two-body approximation
  * Lightweight and fast
  * Suitable for simulation and prototyping

* **Skyfield Model (Paper Results)**

  * Data-driven, high-precision propagation
  * Includes real orbital perturbations
  * Used for final reported results

#### Implication

Due to these differences, **minor deviations in satellite position, link geometry, and derived metrics (e.g., loss, SKR)** may occur between this implementation and the published results.

For research-grade reproducibility, refer to the Skyfield-based pipeline described in the paper.

---

## Limitations

- Pointing errors are not explicitly modeled  
- Weak turbulence regime assumed (Rytov variance ≪ 1)  
- Simplified BB84 model without finite-key effects  
- Repository uses Keplerian propagation (see backend note below)  

---

## Research Contribution

This framework enables climate-aware evaluation of satellite QKD links by integrating:

- Reanalysis-driven atmospheric modeling (ERA5, CAMS)  
- Wavelength-dependent attenuation analysis  
- Site-specific performance comparison  
- End-to-end coupling of atmospheric physics with QKD metrics  

The approach supports realistic feasibility analysis of quantum communication systems under geographically varying environmental conditions.

---

## Citation

If you use this code or model in your research, please cite the associated manuscript (details to be added upon acceptance).

---

## License

© All Rights Reserved. See [LICENSE](LICENSE) for terms.  
Academic review use only. Copying, redistribution, and derivative works are **strictly prohibited** until the official open-source release.
