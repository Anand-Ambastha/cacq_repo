"""
plots/visualise.py
==================
Publication-quality figure generation for the CAQC pipeline.

All functions accept a ``master_df`` pandas DataFrame produced by
``pipeline.run_caqc.run_pipeline()`` and save figures to ``save_dir``.

Plotting style follows journal conventions (serif font, 600 dpi,
no excessive grid lines).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config.settings import FIG_DPI, LAMBDA_M, LAMBDA_M_ALT, A_GROUND, V_WIND

matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def _ensure(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AOD vs SKR (journal scatter)
# ─────────────────────────────────────────────────────────────────────────────

def plot_aod_vs_skr(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Scatter plot of AOD (550 nm) vs SKR for both wavelengths."""
    _ensure(save_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(master_df["AOD_550"], master_df["SKR_785"],
               marker="o", s=60, label="785 nm")
    ax.scatter(master_df["AOD_550"], master_df["SKR_1550"],
               marker="s", s=60, label="1550 nm")

    for i, city in enumerate(master_df["City"]):
        ax.annotate(city,
                    (master_df["AOD_550"].iloc[i], master_df["SKR_1550"].iloc[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("AOD at 550 nm", fontsize=12)
    ax.set_ylabel("Secure Key Yield (bits / pass)", fontsize=12)
    ax.set_title("Aerosol Loading vs QKD Performance", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    _save(fig, os.path.join(save_dir, "aod_vs_skr.pdf"))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(master_df["AOD_550"], master_df["SKR_785"], marker="o", s=60, label="785 nm")
    ax2.scatter(master_df["AOD_550"], master_df["SKR_1550"], marker="s", s=60, label="1550 nm")
    for i, city in enumerate(master_df["City"]):
        ax2.annotate(city,
                     (master_df["AOD_550"].iloc[i], master_df["SKR_1550"].iloc[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.set_xlabel("AOD at 550 nm", fontsize=12)
    ax2.set_ylabel("Secure Key Yield (bits / pass)", fontsize=12)
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig2.tight_layout()
    _save(fig2, os.path.join(save_dir, "aod_vs_skr.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss decomposition (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_breakdown(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Stacked bar of Rayleigh, Aerosol, Turbulence, Geometry contributions."""
    _ensure(save_dir)
    cities = master_df["City"].tolist()
    x = np.arange(len(cities))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, master_df["Rayleigh_785_dB"],  w, label="Rayleigh")
    ax.bar(x, master_df["Aerosol_785_dB"],   w,
           bottom=master_df["Rayleigh_785_dB"],
           label="Aerosol")
    ax.bar(x, master_df["Turbulence_785_dB"], w,
           bottom=master_df["Rayleigh_785_dB"] + master_df["Aerosol_785_dB"],
           label="Turbulence")
    ax.bar(x, master_df["Geom_785_dB"], w,
           bottom=(master_df["Rayleigh_785_dB"] + master_df["Aerosol_785_dB"]
                   + master_df["Turbulence_785_dB"]),
           label="Geometric")

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right")
    ax.set_ylabel("Loss (dB)")
    ax.set_title("Loss Decomposition at 785 nm")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "loss_breakdown.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CAQC vs Standard vs ITU comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Grouped bar comparing CAQC, standard, and ITU-R models."""
    _ensure(save_dir)
    cities = master_df["City"].tolist()
    x = np.arange(len(cities))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.25, master_df["CAQC_785"],            0.25, label="CAQC")
    ax.bar(x,        master_df["ITU_785_dB"],           0.25, label="ITU-R")
    ax.bar(x + 0.25, master_df["Standard_785_dB"],      0.25, label="Standard")

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right")
    ax.set_ylabel("Channel Loss (dB)")
    ax.set_title("CAQC vs ITU-R vs Standard Model — 785 nm")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "model_comparison.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cn² profiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_cn2_profiles(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Semi-log Cn²(h) profiles coloured by city wind speed."""
    from core.turbulence import Cn2_HV
    _ensure(save_dir)
    h = np.linspace(0, 20_000, 500)

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in master_df.iterrows():
        Cn2 = Cn2_HV(h, v_wind=row["HighWind"])
        ax.semilogy(Cn2, h, label=row["City"])

    ax.set_xlabel("Cn² (m$^{-2/3}$)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("HV 5/7 Cn² Profiles")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "cn2_profiles.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dual-axis Loss vs SKR
# ─────────────────────────────────────────────────────────────────────────────

def plot_dual_axis(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Dual Y-axis: channel loss (left) and SKR (right)."""
    _ensure(save_dir)
    cities = master_df["City"].tolist()
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(cities, master_df["CAQC_785"],  "o-", label="Loss 785 nm")
    ax2.plot(cities, master_df["SKR_785"],   "s--", color="tab:orange", label="SKR 785 nm")

    ax1.set_ylabel("Channel Loss (dB)")
    ax2.set_ylabel("Secure Key Yield (bits)")
    ax1.set_title("Channel Loss vs QKD Performance")
    plt.xticks(rotation=25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "dual_axis.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Wavelength comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_wavelength_comparison(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Bar chart: 785 nm vs 1550 nm CAQC loss."""
    _ensure(save_dir)
    cities = master_df["City"].tolist()
    x = np.arange(len(cities))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - 0.2, master_df["CAQC_785"],  0.4, label="785 nm")
    ax.bar(x + 0.2, master_df["CAQC_1550"], 0.4, label="1550 nm")

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right")
    ax.set_ylabel("Channel Loss (dB)")
    ax.set_title("Wavelength-Resolved CAQC Loss")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "wavelength_comparison.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 7. AOD sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_aod_sensitivity(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Channel loss vs AOD_550 sweep for both wavelengths."""
    from core.atmosphere import aerosol_loss_dB
    _ensure(save_dir)

    aod_range = [0.05, 0.10, 0.20, 0.50, 1.00]
    alpha = 1.3

    geom785  = float(master_df["Geom_785_dB"].mean())
    geom1550 = float(master_df["Geom_1550_dB"].mean())
    ray785   = float(master_df["Rayleigh_785_dB"].mean())
    ray1550  = float(master_df["Rayleigh_1550_dB"].mean())
    turb785  = float(master_df["Turbulence_785_dB"].mean())
    turb1550 = float(master_df["Turbulence_1550_dB"].mean())
    base_aer785  = float(master_df["Aerosol_785_dB"].mean())
    base_aer1550 = float(master_df["Aerosol_1550_dB"].mean())

    losses785, losses1550 = [], []
    for aod in aod_range:
        a785  = aerosol_loss_dB(aod, LAMBDA_M)
        a1550 = aerosol_loss_dB(aod, LAMBDA_M_ALT)
        losses785.append( geom785  + (ray785  - base_aer785  + a785)  + turb785)
        losses1550.append(geom1550 + (ray1550 - base_aer1550 + a1550) + turb1550)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(aod_range, losses785,  "o-", label="785 nm")
    ax.plot(aod_range, losses1550, "s-", label="1550 nm")
    ax.set_xlabel("AOD at 550 nm")
    ax.set_ylabel("Channel Loss (dB)")
    ax.set_title("AOD Sensitivity Analysis")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "aod_sensitivity.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Link availability bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_availability(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    _ensure(save_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(master_df["City"], master_df["Link_Availability_%"], color="seagreen")
    ax.set_ylabel("Link Availability (%)")
    ax.set_title("Estimated Link Availability per Site")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "availability.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 9. 3D atmospheric scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d_atm_space(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """3-D scatter: AOD × wind × CAQC loss."""
    _ensure(save_dir)
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(master_df["AOD_550"],
               master_df["HighWind"],
               master_df["CAQC_785"])

    for _, row in master_df.iterrows():
        ax.text(row["AOD_550"], row["HighWind"], row["CAQC_785"],
                row["City"], fontsize=7)

    ax.set_xlabel("AOD")
    ax.set_ylabel("Wind (m/s)")
    ax.set_zlabel("Loss (dB)")
    ax.set_title("Atmospheric Influence Space")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "3d_atm_space.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Performance space (normalised)
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_space(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    _ensure(save_dir)
    norm_loss = master_df["CAQC_1550"] / master_df["CAQC_1550"].max()
    norm_skr  = master_df["SKR_1550"]  / master_df["SKR_1550"].max()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(norm_loss, norm_skr)
    for i, city in enumerate(master_df["City"]):
        ax.annotate(city, (norm_loss.iloc[i], norm_skr.iloc[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Normalised Loss")
    ax.set_ylabel("Normalised SKR")
    ax.set_title("QKD Performance Space (1550 nm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "performance_space.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 11. r0 sensitivity to wind speed
# ─────────────────────────────────────────────────────────────────────────────

def plot_r0_sensitivity(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Fried r₀ vs wind speed for each ground station."""
    from core.turbulence import Cn2_HV
    _ensure(save_dir)

    h           = np.linspace(0, 20_000, 500)
    wind_range  = np.linspace(2, 20, 20)
    from math import pi
    k_785 = 2.0 * pi / LAMBDA_M

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in master_df.iterrows():
        r0_list = []
        for v in wind_range:
            Cn2 = Cn2_HV(h, v_wind=v * (row["HighWind"] / 10.0))
            J   = float(np.trapezoid(Cn2, h))
            r0  = (0.423 * k_785 ** 2 * J) ** (-3.0 / 5.0) if J > 0 else 0.0
            r0_list.append(r0)
        ax.plot(wind_range, r0_list, label=row["City"])

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Fried r₀ (m)")
    ax.set_title("Turbulence Sensitivity: r₀ vs Wind")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "r0_sensitivity.png"))


# ─────────────────────────────────────────────────────────────────────────────
# BATCH — generate all publication figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(master_df: pd.DataFrame, save_dir: str = "outputs/plots") -> None:
    """Convenience wrapper: call all plot functions."""
    _ensure(save_dir)
    plot_aod_vs_skr(master_df, save_dir)
    plot_loss_breakdown(master_df, save_dir)
    plot_model_comparison(master_df, save_dir)
    plot_cn2_profiles(master_df, save_dir)
    plot_dual_axis(master_df, save_dir)
    plot_wavelength_comparison(master_df, save_dir)
    plot_aod_sensitivity(master_df, save_dir)
    plot_availability(master_df, save_dir)
    plot_3d_atm_space(master_df, save_dir)
    plot_performance_space(master_df, save_dir)
    plot_r0_sensitivity(master_df, save_dir)
    print(f"\n[plots] All figures saved to {save_dir}/")
