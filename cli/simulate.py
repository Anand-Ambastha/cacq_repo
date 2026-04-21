"""
cli/simulate.py
===============
Command-line interface for the CAQC pipeline.

Usage examples
--------------
# Full pipeline (requires ERA5 credentials)
    python -m cli.simulate --mode full

# Physics-only demo with synthetic parameters (no ERA5 required)
    python -m cli.simulate --mode demo

# Single-point link budget
    python -m cli.simulate --mode link --elev 30 --range 800 --aod 0.15

# AOD sweep
    python -m cli.simulate --mode sweep --param aod

# Elevation sweep
    python -m cli.simulate --mode sweep --param elev
"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import pandas as pd

# ── allow running as  python -m cli.simulate  from repo root ──────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.settings import (
    LAMBDA_M, LAMBDA_M_ALT,
    D_RX, D_TX, MIN_ELEV, DT,
    GROUND_STATIONS,
)
from core import (
    geometric_loss_dB,
    slant_atm_loss_dB,
    turbulence_loss_dB,
    compute_skr_per_pulse,
    loss_dB_to_eta,
    Cn2_HV,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_BAR = "─" * 60


def _header(title: str) -> None:
    print(f"\n{_BAR}")
    print(f"  CAQC  |  {title}")
    print(_BAR)


def _link_budget(
    elev_deg: float,
    range_km: float,
    P_hPa: float,
    T_C: float,
    RH_pct: float,
    aod_550: float,
    v_wind: float,
    label: str = "",
) -> dict:
    """Compute and print a full link budget for one geometry."""
    L = range_km * 1000.0

    g785  = geometric_loss_dB(L, LAMBDA_M,     D_RX, D_TX)
    g1550 = geometric_loss_dB(L, LAMBDA_M_ALT, D_RX, D_TX)

    a785  = slant_atm_loss_dB(LAMBDA_M,     elev_deg, P_hPa, T_C, RH_pct, aod_550)
    a1550 = slant_atm_loss_dB(LAMBDA_M_ALT, elev_deg, P_hPa, T_C, RH_pct, aod_550)

    t785,  s785,  r0_785  = turbulence_loss_dB(LAMBDA_M,     elev_deg, v_wind)
    t1550, s1550, r0_1550 = turbulence_loss_dB(LAMBDA_M_ALT, elev_deg, v_wind)

    tot785  = g785  + a785  + t785
    tot1550 = g1550 + a1550 + t1550

    eta785  = loss_dB_to_eta(tot785)
    eta1550 = loss_dB_to_eta(tot1550)

    skr785  = compute_skr_per_pulse(eta785)  * DT
    skr1550 = compute_skr_per_pulse(eta1550) * DT

    tag = f" [{label}]" if label else ""
    print(f"\n  Geometry{tag}:  elev={elev_deg:.1f}°   range={range_km:.0f} km")
    print(f"  Atmosphere:   P={P_hPa:.1f} hPa  T={T_C:.1f}°C  RH={RH_pct:.1f}%  AOD={aod_550:.3f}")
    print(f"  Wind:         {v_wind:.1f} m/s   r₀(785)={r0_785*100:.1f} cm")
    print()
    print(f"  {'Component':<22}  {'785 nm':>10}  {'1550 nm':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
    print(f"  {'Geometric':<22}  {g785:>10.3f}  {g1550:>10.3f}  dB")
    print(f"  {'Atmospheric':<22}  {a785:>10.3f}  {a1550:>10.3f}  dB")
    print(f"  {'Turbulence':<22}  {t785:>10.3f}  {t1550:>10.3f}  dB")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
    print(f"  {'TOTAL CAQC':<22}  {tot785:>10.3f}  {tot1550:>10.3f}  dB")
    print(f"  {'η (linear)':<22}  {eta785:>10.4e}  {eta1550:>10.4e}")
    print(f"  {'SKR (bits/pass)':<22}  {skr785:>10.4f}  {skr1550:>10.4f}")
    print(f"  {'Rytov σ_R²':<22}  {s785:>10.4f}  {s1550:>10.4f}")

    return {
        "elev_deg": elev_deg, "range_km": range_km,
        "geom_785": g785,  "atm_785": a785,  "turb_785": t785,  "total_785": tot785,
        "geom_1550": g1550, "atm_1550": a1550, "turb_1550": t1550, "total_1550": tot1550,
        "skr_785": skr785, "skr_1550": skr1550,
        "r0_785": r0_785,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────

def mode_demo() -> None:
    """Synthetic demo — no ERA5 required."""
    _header("Physics Demo (synthetic parameters)")

    demo_sites = {
        "Hanle":      dict(elev=45, range_km=600, P=640, T=5,  RH=20, aod=0.05, v=8),
        "Dehradun":   dict(elev=35, range_km=750, P=920, T=22, RH=65, aod=0.22, v=12),
        "MtAbu":      dict(elev=40, range_km=680, P=850, T=18, RH=40, aod=0.14, v=10),
        "Shillong":   dict(elev=30, range_km=820, P=900, T=20, RH=72, aod=0.30, v=14),
        "Kodaikanal": dict(elev=50, range_km=550, P=840, T=15, RH=55, aod=0.10, v=9),
    }

    rows = []
    for name, p in demo_sites.items():
        res = _link_budget(
            p["elev"], p["range_km"],
            p["P"], p["T"], p["RH"], p["aod"], p["v"],
            label=name,
        )
        res["city"] = name
        rows.append(res)

    df = pd.DataFrame(rows)
    print(f"\n{'─'*60}")
    print("  SUMMARY TABLE")
    print(f"{'─'*60}")
    cols = ["city", "total_785", "total_1550", "skr_785", "skr_1550"]
    print(df[cols].to_string(index=False, float_format="%.3f"))


def mode_link(args: argparse.Namespace) -> None:
    """Single link budget from CLI arguments."""
    _header("Single Link Budget")
    _link_budget(
        elev_deg=args.elev,
        range_km=args.range,
        P_hPa=args.pressure,
        T_C=args.temp,
        RH_pct=args.rh,
        aod_550=args.aod,
        v_wind=args.wind,
    )


def mode_sweep(args: argparse.Namespace) -> None:
    """Parameter sweep over AOD or elevation."""
    param = args.param
    _header(f"Parameter Sweep: {param}")

    # Fixed baseline
    base = dict(elev=40.0, range_km=650.0, P=870.0, T=15.0, RH=45.0, aod=0.12, v=10.0)

    if param == "aod":
        sweep_vals = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
        key        = "aod"
        x_label    = "AOD_550"
    elif param == "elev":
        sweep_vals = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
        key        = "elev"
        x_label    = "Elevation (°)"
    elif param == "range":
        sweep_vals = [300, 500, 700, 900, 1200, 1500]
        key        = "range_km"
        x_label    = "Range (km)"
    else:
        print(f"Unknown sweep param '{param}'. Choose: aod | elev | range")
        return

    print(f"\n  {x_label:<14} {'Loss 785':>10} {'Loss 1550':>10} {'SKR 785':>12} {'SKR 1550':>12}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for val in sweep_vals:
        p = dict(base)
        p[key] = val
        if p["elev"] <= 0:
            continue

        L = p["range_km"] * 1000.0
        g785  = geometric_loss_dB(L, LAMBDA_M)
        g1550 = geometric_loss_dB(L, LAMBDA_M_ALT)
        a785  = slant_atm_loss_dB(LAMBDA_M,     p["elev"], p["P"], p["T"], p["RH"], p["aod"])
        a1550 = slant_atm_loss_dB(LAMBDA_M_ALT, p["elev"], p["P"], p["T"], p["RH"], p["aod"])
        t785,  _, _ = turbulence_loss_dB(LAMBDA_M,     p["elev"], p["v"])
        t1550, _, _ = turbulence_loss_dB(LAMBDA_M_ALT, p["elev"], p["v"])
        tot785  = g785  + a785  + t785
        tot1550 = g1550 + a1550 + t1550
        skr785  = compute_skr_per_pulse(loss_dB_to_eta(tot785))  * DT
        skr1550 = compute_skr_per_pulse(loss_dB_to_eta(tot1550)) * DT

        print(f"  {val:<14.2f} {tot785:>10.3f} {tot1550:>10.3f} {skr785:>12.4f} {skr1550:>12.4f}")


def mode_full(args: argparse.Namespace) -> None:
    """Full pipeline — requires ERA5 credentials."""
    _header("Full CAQC Pipeline")
    try:
        from pipeline.run_caqc import run_pipeline
        from plots.visualise   import generate_all
        from tables.export     import export_all
    except ImportError as e:
        print(f"Import error: {e}")
        return

    master_df = run_pipeline(cache_dir=args.cache_dir, verbose=True)
    generate_all(master_df, save_dir=args.plot_dir)
    export_all(master_df,   save_dir=args.table_dir)
    print("\n[CLI] Full run complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caqc",
        description="CAQC — Climate-Aware Quantum Channel simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.simulate --mode demo
  python -m cli.simulate --mode link --elev 35 --range 700 --aod 0.20
  python -m cli.simulate --mode sweep --param aod
  python -m cli.simulate --mode sweep --param elev
  python -m cli.simulate --mode full
""",
    )

    p.add_argument("--mode", choices=["demo", "link", "sweep", "full"],
                   default="demo",
                   help="Simulation mode (default: demo)")

    # Link-budget arguments
    p.add_argument("--elev",     type=float, default=40.0,  help="Elevation angle (°)")
    p.add_argument("--range",    type=float, default=650.0, help="Slant range (km)")
    p.add_argument("--aod",      type=float, default=0.12,  help="AOD at 550 nm")
    p.add_argument("--pressure", type=float, default=870.0, help="Surface pressure (hPa)")
    p.add_argument("--temp",     type=float, default=15.0,  help="Temperature (°C)")
    p.add_argument("--rh",       type=float, default=45.0,  help="Relative humidity (%)")
    p.add_argument("--wind",     type=float, default=10.0,  help="Wind speed (m/s)")

    # Sweep
    p.add_argument("--param", choices=["aod", "elev", "range"], default="aod",
                   help="Parameter to sweep (default: aod)")

    # Full pipeline paths
    p.add_argument("--cache-dir",  default=".",            help="ERA5 cache directory")
    p.add_argument("--plot-dir",   default="outputs/plots", help="Output plots directory")
    p.add_argument("--table-dir",  default="outputs/tables", help="Output tables directory")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "demo"  : lambda: mode_demo(),
        "link"  : lambda: mode_link(args),
        "sweep" : lambda: mode_sweep(args),
        "full"  : lambda: mode_full(args),
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
