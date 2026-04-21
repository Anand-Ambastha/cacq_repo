"""
dashboard/app.py
================
CAQC Interactive Dashboard (Streamlit)

Run with:
    streamlit run dashboard/app.py

Features
--------
* Real-time single-pass link budget calculator
* Interactive Cn² / turbulence explorer
* AOD & elevation sweep plots
* Per-site atmospheric profile viewer
* Master results table viewer (upload CSV or use demo data)
* Download figures directly from browser
"""

from __future__ import annotations

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── allow  streamlit run dashboard/app.py  from repo root ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from config.settings import (
    LAMBDA_M, LAMBDA_M_ALT,
    D_RX, D_TX, DT,
    GROUND_STATIONS,
    V_WIND, A_GROUND,
)
from core import (
    geometric_loss_dB,
    slant_atm_loss_dB,
    turbulence_loss_dB,
    compute_skr_per_pulse,
    loss_dB_to_eta,
    Cn2_HV,
    fried_parameter, rytov_variance,
    integrated_Cn2,
    rayleigh_loss_dB,
    aerosol_loss_dB,
    gas_loss_dB,
    air_mass,
    aod_at_wavelength,
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CAQC Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🛰️ CAQC Simulator")
st.sidebar.markdown("*Climate-Aware Quantum Channel*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Link Budget", "Parameter Sweep", "Turbulence Explorer",
     "Cn² Profile", "Results Viewer"],
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Wavelengths**  \n"
    "• 785 nm (primary)  \n"
    "• 1550 nm (telecom)"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _compute_budget(elev, range_km, P, T, RH, aod, v_wind):
    L = range_km * 1000.0
    sin_e = np.sin(np.deg2rad(elev))

    g785   = geometric_loss_dB(L, LAMBDA_M)
    g1550  = geometric_loss_dB(L, LAMBDA_M_ALT)

    a785   = slant_atm_loss_dB(LAMBDA_M,     elev, P, T, RH, aod)
    a1550  = slant_atm_loss_dB(LAMBDA_M_ALT, elev, P, T, RH, aod)

    t785,  s785,  r0_785  = turbulence_loss_dB(LAMBDA_M,     elev, v_wind)
    t1550, s1550, r0_1550 = turbulence_loss_dB(LAMBDA_M_ALT, elev, v_wind)

    tot785  = g785  + a785  + t785
    tot1550 = g1550 + a1550 + t1550

    eta785  = loss_dB_to_eta(tot785)
    eta1550 = loss_dB_to_eta(tot1550)

    skr785  = compute_skr_per_pulse(eta785)  * DT
    skr1550 = compute_skr_per_pulse(eta1550) * DT

    return {
        "g785": g785,  "a785": a785,  "t785": t785,  "tot785": tot785,
        "g1550": g1550, "a1550": a1550, "t1550": t1550, "tot1550": tot1550,
        "eta785": eta785, "eta1550": eta1550,
        "skr785": skr785, "skr1550": skr1550,
        "r0_785": r0_785, "r0_1550": r0_1550,
        "sigma_785": s785, "sigma_1550": s1550,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: LINK BUDGET CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

if page == "Link Budget":
    st.title("📡 Real-Time Link Budget Calculator")
    st.markdown(
        "Adjust the sliders to compute the full CAQC atmospheric channel budget "
        "for a satellite-to-ground QKD link."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Geometry")
        elev      = st.slider("Elevation angle (°)",       10,  90, 40,  step=1)
        range_km  = st.slider("Slant range (km)",          300, 1500, 650, step=10)

        st.subheader("Turbulence")
        v_wind = st.slider("Effective wind speed (m/s)", 2.0, 25.0, 10.0, step=0.5)

    with col2:
        st.subheader("Atmosphere")
        P     = st.slider("Pressure (hPa)",          600, 1013, 870, step=5)
        T     = st.slider("Temperature (°C)",        -20,  40,   15, step=1)
        RH    = st.slider("Relative humidity (%)",     5,  100,  45, step=5)
        aod   = st.slider("AOD at 550 nm",           0.01, 1.5, 0.12, step=0.01)

    res = _compute_budget(elev, range_km, P, T, RH, aod, v_wind)

    st.divider()

    # ── Metrics row ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Loss 785 nm",  f"{res['tot785']:.2f} dB")
    m2.metric("Total Loss 1550 nm", f"{res['tot1550']:.2f} dB")
    m3.metric("SKR 785 nm",         f"{res['skr785']:.4f} bits/pass")
    m4.metric("SKR 1550 nm",        f"{res['skr1550']:.4f} bits/pass")
    m5.metric("Fried r₀ (785 nm)",  f"{res['r0_785']*100:.1f} cm")
    m6.metric("Rytov σ²  (785 nm)", f"{res['sigma_785']:.4f}")

    # ── Breakdown bars ──
    st.divider()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Geometric", "Atmospheric", "Turbulence", "Total"]
    vals785  = [res["g785"],  res["a785"],  res["t785"],  res["tot785"]]
    vals1550 = [res["g1550"], res["a1550"], res["t1550"], res["tot1550"]]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    axes[0].bar(labels, vals785,  color=colors)
    axes[0].set_title("785 nm Loss Budget")
    axes[0].set_ylabel("Loss (dB)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(labels, vals1550, color=colors)
    axes[1].set_title("1550 nm Loss Budget")
    axes[1].set_ylabel("Loss (dB)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button("⬇ Download figure", _fig_to_bytes(fig),
                       file_name="link_budget.png", mime="image/png")

    # ── Detailed table ──
    st.subheader("Detailed Results")
    detail = pd.DataFrame({
        "Component"  : ["Geometric", "Atmospheric", "Turbulence", "TOTAL"],
        "785 nm (dB)": [res["g785"], res["a785"], res["t785"], res["tot785"]],
        "1550 nm (dB)": [res["g1550"], res["a1550"], res["t1550"], res["tot1550"]],
    }).round(4)
    st.dataframe(detail, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Parameter Sweep":
    st.title("🔬 Parameter Sweep")
    st.markdown("Sweep a single channel parameter while holding all others fixed.")

    sweep_param = st.selectbox(
        "Parameter to sweep",
        ["AOD (550 nm)", "Elevation angle", "Slant range", "Wind speed",
         "Relative humidity", "Pressure"],
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fixed baseline**")
        base_elev  = st.number_input("Elevation (°)",    value=40.0, min_value=10.0, max_value=90.0)
        base_range = st.number_input("Range (km)",        value=650.0, min_value=100.0, max_value=2000.0)
        base_P     = st.number_input("Pressure (hPa)",   value=870.0, min_value=500.0, max_value=1013.0)
    with col2:
        base_T     = st.number_input("Temperature (°C)", value=15.0, min_value=-30.0, max_value=50.0)
        base_RH    = st.number_input("RH (%)",           value=45.0, min_value=5.0,  max_value=100.0)
        base_aod   = st.number_input("AOD",              value=0.12, min_value=0.01, max_value=2.0)
        base_wind  = st.number_input("Wind (m/s)",       value=10.0, min_value=1.0,  max_value=30.0)

    # build sweep axis
    param_map = {
        "AOD (550 nm)"      : ("aod",   np.linspace(0.01, 1.5, 50)),
        "Elevation angle"   : ("elev",  np.linspace(10, 90, 50)),
        "Slant range"       : ("range", np.linspace(200, 2000, 50)),
        "Wind speed"        : ("wind",  np.linspace(2, 25, 50)),
        "Relative humidity" : ("rh",    np.linspace(5, 100, 50)),
        "Pressure"          : ("P",     np.linspace(500, 1013, 50)),
    }

    key, sweep_vals = param_map[sweep_param]

    losses785, losses1550, skrs785, skrs1550 = [], [], [], []
    for val in sweep_vals:
        p = dict(elev=base_elev, range_km=base_range, P=base_P,
                 T=base_T, RH=base_RH, aod=base_aod, v_wind=base_wind)
        p[key if key != "range" else "range_km"] = val
        if p["elev"] <= 0:
            continue

        try:
            r = _compute_budget(**p)
        except Exception:
            continue

        losses785.append(r["tot785"])
        losses1550.append(r["tot1550"])
        skrs785.append(r["skr785"])
        skrs1550.append(r["skr1550"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(sweep_vals[:len(losses785)], losses785,  label="785 nm")
    ax1.plot(sweep_vals[:len(losses1550)], losses1550, label="1550 nm")
    ax1.set_xlabel(sweep_param)
    ax1.set_ylabel("Channel Loss (dB)")
    ax1.set_title(f"Loss vs {sweep_param}")
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(sweep_vals[:len(skrs785)],  skrs785,  label="785 nm")
    ax2.plot(sweep_vals[:len(skrs1550)], skrs1550, label="1550 nm")
    ax2.set_xlabel(sweep_param)
    ax2.set_ylabel("SKR (bits / pass)")
    ax2.set_title(f"SKR vs {sweep_param}")
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button("⬇ Download sweep figure", _fig_to_bytes(fig),
                       file_name="sweep.png", mime="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: TURBULENCE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Turbulence Explorer":
    st.title("🌪️ Turbulence Explorer")
    st.markdown(
        "Interactive Hufnagel–Valley 5/7 turbulence model.  "
        "Adjust wind and ground-layer strength to observe effects on "
        "Fried coherence length r₀, Rytov variance σ², and loss."
    )

    col1, col2 = st.columns(2)
    with col1:
        v_eff    = st.slider("RMS wind speed (m/s)", 2.0, 30.0, 21.0, step=0.5)
        a_ground = st.select_slider(
            "Ground-layer Cn² (A)",
            options=[1e-15, 5e-15, 1e-14, 1.7e-14, 5e-14, 1e-13, 5e-13],
            value=1.7e-14,
            format_func=lambda x: f"{x:.1e}",
        )
    with col2:
        elev_turb = st.slider("Elevation angle (°)", 10, 90, 40)
        lam_nm    = st.radio("Wavelength", [785, 1550], index=0)

    h = np.linspace(0, 20_000, 500)
    Cn2 = Cn2_HV(h, v_wind=v_eff, a_ground=a_ground)

    J = float(np.trapezoid(Cn2, h))
    from math import pi
    k = 2.0 * pi / (lam_nm * 1e-9)
    r0 = (0.423 * k**2 * J) ** (-3.0/5.0) if J > 0 else 0.0
    sigma_R2 = 1.23 * k**(7/6) * J
    eta_turb = float(np.exp(-np.clip(sigma_R2, 0, 50)))
    loss_db  = -10 * np.log10(max(eta_turb, 1e-30))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fried r₀",       f"{r0*100:.2f} cm")
    m2.metric("Rytov σ²",       f"{sigma_R2:.4f}")
    m3.metric("η_turb",         f"{eta_turb:.4f}")
    m4.metric("Turbulence loss", f"{loss_db:.3f} dB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.semilogy(Cn2, h)
    ax1.set_xlabel("Cn² (m⁻²/³)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title(f"HV 5/7 Profile  (v={v_eff:.1f} m/s)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Rytov integrand shape
    z = np.linspace(0, 500_000, 200)
    h2 = np.minimum(z * np.sin(np.deg2rad(elev_turb)), 20_000)
    Cn2_z = Cn2_HV(h2, v_eff, a_ground)
    L = 500_000
    integrand = Cn2_z * (L - z) ** (5/6)
    ax2.plot(z / 1000, integrand)
    ax2.set_xlabel("Distance along path (km)")
    ax2.set_ylabel("Cn² · (L−z)^(5/6)")
    ax2.set_title(f"Rytov Integrand  (elev={elev_turb}°)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Wind sensitivity heatmap
    st.subheader("r₀ Sensitivity Heatmap")
    wind_range  = np.linspace(2, 25, 30)
    elev_range  = np.linspace(10, 80, 30)
    R0 = np.zeros((len(elev_range), len(wind_range)))

    for i, el in enumerate(elev_range):
        for j, vw in enumerate(wind_range):
            J_ij = integrated_Cn2(el, vw, a_ground)
            R0[i, j] = fried_parameter(lam_nm * 1e-9, J_ij) * 100  # cm

    fig2, ax = plt.subplots(figsize=(7, 4))
    c = ax.contourf(wind_range, elev_range, R0, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax, label="r₀ (cm)")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(f"Fried r₀ — {lam_nm} nm")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Cn² PROFILE VIEWER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Cn² Profile":
    st.title("📈 Cn² Profile Viewer")
    st.markdown(
        "Compare HV 5/7 turbulence profiles across user-defined sites or "
        "the default ground-station configuration."
    )

    use_defaults = st.checkbox("Use default ground stations (demo wind speeds)", value=True)

    if use_defaults:
        site_winds = {
            "Hanle":      8.0,
            "Dehradun":  12.0,
            "MtAbu":     10.0,
            "Shillong":  14.0,
            "Kodaikanal": 9.0,
        }
    else:
        n_sites = st.number_input("Number of sites", 1, 8, 3, step=1)
        site_winds = {}
        cols = st.columns(int(n_sites))
        for i in range(int(n_sites)):
            with cols[i]:
                name = st.text_input(f"Site {i+1} name", f"Site{i+1}", key=f"name{i}")
                wind = st.number_input(f"Wind (m/s)", 2.0, 30.0, 10.0, key=f"wind{i}")
                site_winds[name] = wind

    h = np.linspace(0, 20_000, 500)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, vw in site_winds.items():
        Cn2 = Cn2_HV(h, v_wind=vw)
        ax.semilogy(Cn2, h, label=f"{name} ({vw:.0f} m/s)")

    ax.set_xlabel("Cn² (m⁻²/³)", fontsize=11)
    ax.set_ylabel("Altitude (m)", fontsize=11)
    ax.set_title("HV 5/7 Cn² Profiles")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button("⬇ Download", _fig_to_bytes(fig),
                       file_name="cn2_profiles.png", mime="image/png")

    # 3-D time-evolution
    if st.checkbox("Show 3-D turbulence time-evolution"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        h3 = np.linspace(0, 20_000, 100)
        t_steps = 50
        Cn2_3D = np.zeros((t_steps, 100))
        for ti in range(t_steps):
            vw = 5 + 10 * np.sin(2 * np.pi * ti / t_steps)
            Cn2_3D[ti] = Cn2_HV(h3, v_wind=vw)

        H, T = np.meshgrid(h3, np.arange(t_steps))
        fig3 = plt.figure(figsize=(8, 5))
        ax3  = fig3.add_subplot(111, projection="3d")
        ax3.plot_surface(H, T, Cn2_3D, cmap="plasma", linewidth=0)
        ax3.set_xlabel("Altitude (m)")
        ax3.set_ylabel("Time step")
        ax3.set_zlabel("Cn²")
        ax3.set_title("Diurnal Turbulence Evolution")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: RESULTS VIEWER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Results Viewer":
    st.title("📊 Results Viewer")
    st.markdown(
        "Upload a **master_results.csv** (produced by the full pipeline) "
        "or load the **built-in demo dataset** to explore all figures."
    )

    uploaded = st.file_uploader("Upload master_results.csv", type=["csv"])

    if uploaded is not None:
        master_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(master_df)} rows from uploaded file.")
    else:
        st.info("No file uploaded — using synthetic demo data.")
        # Synthetic demo data (no ERA5 required)
        master_df = pd.DataFrame({
            "City"              : list(GROUND_STATIONS.keys()),
            "Temp_C"            : [5.2, 22.1, 18.3, 20.5, 15.4],
            "Pressure_hPa"      : [640, 920, 850, 900, 840],
            "RH_%"              : [22, 65, 40, 72, 55],
            "AOD_550"           : [0.053, 0.218, 0.142, 0.302, 0.097],
            "HighWind"          : [8.1, 12.3, 9.7, 14.2, 8.9],
            "CAQC_785"          : [28.4, 35.2, 31.7, 38.1, 29.8],
            "CAQC_1550"         : [24.1, 30.5, 27.3, 33.6, 25.4],
            "Standard_785_dB"   : [25.1, 32.0, 28.5, 34.9, 26.7],
            "ITU_785_dB"        : [26.3, 33.1, 29.8, 36.2, 27.9],
            "Error_785_%"       : [11.3, 9.1, 10.1, 8.4, 10.4],
            "Err_ITU_785_%"     : [7.4, 6.0, 6.0, 5.0, 6.4],
            "SKR_785"           : [0.0821, 0.0203, 0.0441, 0.0112, 0.0631],
            "SKR_1550"          : [0.1402, 0.0452, 0.0823, 0.0284, 0.1051],
            "Geom_785_dB"       : [18.2, 21.4, 19.8, 22.7, 18.9],
            "Geom_1550_dB"      : [16.1, 19.2, 17.7, 20.5, 16.8],
            "Rayleigh_785_dB"   : [1.2, 1.9, 1.6, 1.8, 1.5],
            "Aerosol_785_dB"    : [0.7, 2.9, 1.9, 4.0, 1.3],
            "Turbulence_785_dB" : [8.3, 8.9, 8.4, 9.6, 8.1],
            "Rayleigh_1550_dB"  : [0.3, 0.5, 0.4, 0.5, 0.4],
            "Aerosol_1550_dB"   : [0.3, 1.2, 0.8, 1.7, 0.5],
            "Turbulence_1550_dB": [7.4, 7.9, 7.4, 8.5, 7.2],
            "Fried_r0_785"      : [0.042, 0.031, 0.037, 0.027, 0.040],
            "Fried_r0_1550"     : [0.071, 0.053, 0.063, 0.047, 0.068],
            "Total_Loss_dB"     : [28.4, 35.2, 31.7, 38.1, 29.8],
            "Standard_Model_Loss_dB": [25.1, 32.0, 28.5, 34.9, 26.7],
            "Link_Availability_%": [94.2, 87.3, 91.1, 84.5, 93.0],
            "Extra_Loss_dB"     : [3.3, 3.2, 3.2, 3.2, 3.1],
            "Eta_785"           : [3.6e-3, 3.0e-4, 6.8e-4, 1.5e-4, 1.0e-3],
            "Eta_1550"          : [3.9e-3, 8.9e-4, 1.9e-3, 4.4e-4, 2.9e-3],
        })

    # ── Raw table ──
    with st.expander("📋 Raw results table", expanded=False):
        st.dataframe(master_df.round(3), use_container_width=True)
        csv_bytes = master_df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv_bytes,
                           file_name="master_results.csv", mime="text/csv")

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Loss Breakdown", "Model Comparison",
        "Wavelength", "AOD vs SKR", "Performance Space",
    ])

    cities = master_df["City"].tolist()
    x = np.arange(len(cities))

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, master_df["Rayleigh_785_dB"], 0.5, label="Rayleigh")
        ax.bar(x, master_df["Aerosol_785_dB"],  0.5,
               bottom=master_df["Rayleigh_785_dB"], label="Aerosol")
        ax.bar(x, master_df["Turbulence_785_dB"], 0.5,
               bottom=master_df["Rayleigh_785_dB"] + master_df["Aerosol_785_dB"],
               label="Turbulence")
        ax.bar(x, master_df["Geom_785_dB"], 0.5,
               bottom=(master_df["Rayleigh_785_dB"] + master_df["Aerosol_785_dB"]
                       + master_df["Turbulence_785_dB"]),
               label="Geometric")
        ax.set_xticks(x); ax.set_xticklabels(cities, rotation=25)
        ax.set_ylabel("Loss (dB)"); ax.set_title("Loss Decomposition — 785 nm")
        ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download", _fig_to_bytes(fig), "loss_breakdown.png", "image/png")

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 0.25, master_df["CAQC_785"],         0.25, label="CAQC")
        ax.bar(x,        master_df["ITU_785_dB"],        0.25, label="ITU-R")
        ax.bar(x + 0.25, master_df["Standard_785_dB"],   0.25, label="Standard")
        ax.set_xticks(x); ax.set_xticklabels(cities, rotation=25)
        ax.set_ylabel("Loss (dB)"); ax.set_title("CAQC vs ITU-R vs Standard — 785 nm")
        ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download", _fig_to_bytes(fig), "model_comparison.png", "image/png")

    with tab3:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - 0.2, master_df["CAQC_785"],  0.4, label="785 nm")
        ax.bar(x + 0.2, master_df["CAQC_1550"], 0.4, label="1550 nm")
        ax.set_xticks(x); ax.set_xticklabels(cities, rotation=25)
        ax.set_ylabel("Loss (dB)"); ax.set_title("Wavelength Comparison")
        ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download", _fig_to_bytes(fig), "wavelength.png", "image/png")

    with tab4:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(master_df["AOD_550"], master_df["SKR_785"],  marker="o", s=80, label="785 nm")
        ax.scatter(master_df["AOD_550"], master_df["SKR_1550"], marker="s", s=80, label="1550 nm")
        for i, city in enumerate(master_df["City"]):
            ax.annotate(city,
                        (master_df["AOD_550"].iloc[i], master_df["SKR_1550"].iloc[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("AOD at 550 nm"); ax.set_ylabel("SKR (bits/pass)")
        ax.set_title("Aerosol Loading vs QKD Performance")
        ax.legend(frameon=False); ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download", _fig_to_bytes(fig), "aod_vs_skr.png", "image/png")

    with tab5:
        norm_loss = master_df["CAQC_1550"] / master_df["CAQC_1550"].max()
        norm_skr  = master_df["SKR_1550"]  / master_df["SKR_1550"].max()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(norm_loss, norm_skr, s=80, zorder=3)
        for i, city in enumerate(master_df["City"]):
            ax.annotate(city, (norm_loss.iloc[i], norm_skr.iloc[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel("Normalised Loss"); ax.set_ylabel("Normalised SKR")
        ax.set_title("QKD Performance Space — 1550 nm")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇ Download", _fig_to_bytes(fig), "performance_space.png", "image/png")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "CAQC — Climate-Aware Quantum Channel Model  |  "
    "© Research Work — All Rights Reserved  |  "
    "Manuscript under review"
)
