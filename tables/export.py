"""
tables/export.py
================
Generate publication-ready tables (CSV + LaTeX) from the master results
DataFrame produced by ``pipeline.run_caqc.run_pipeline()``.
"""

from __future__ import annotations

import os
import pandas as pd


def _save(df: pd.DataFrame, name: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}.csv")
    tex_path = os.path.join(save_dir, f"{name}.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"  [table] {name}.csv / .tex → {save_dir}/")


def export_all(master_df: pd.DataFrame, save_dir: str = "outputs/tables") -> None:
    """Export all five publication tables."""
    df = master_df.copy().round(3)

    df["Reduction_%"] = (
        (df["CAQC_785"] - df["CAQC_1550"]) / df["CAQC_785"] * 100
    ).round(3)
    df["SKR_Improvement_%"] = (
        (df["SKR_1550"] - df["SKR_785"]) / df["SKR_785"].replace(0, float("nan")) * 100
    ).round(3)

    # Table 1 — Atmospheric Parameters
    t1 = df[["City", "Temp_C", "Pressure_hPa", "RH_%",
              "AOD_550", "HighWind", "Fried_r0_785"]]
    _save(t1, "table1_atmosphere", save_dir)

    # Table 2 — Loss Budget 785 nm
    t2 = df[["City", "Geom_785_dB", "Geom_1550_dB",
              "Rayleigh_785_dB", "Aerosol_785_dB",
              "Turbulence_785_dB", "CAQC_785"]]
    _save(t2, "table2_loss_budget_785", save_dir)

    # Table 3 — Wavelength Comparison
    t3 = df[["City", "CAQC_785", "CAQC_1550", "Reduction_%"]]
    _save(t3, "table3_wavelength", save_dir)

    # Table 4 — Model Comparison + Error
    t4 = df[["City", "CAQC_785", "Standard_785_dB",
              "ITU_785_dB", "Error_785_%", "Err_ITU_785_%"]]
    _save(t4, "table4_model_comparison", save_dir)

    # Table 5 — QKD Performance
    t5 = df[["City", "SKR_785", "SKR_1550", "SKR_Improvement_%",
              "Eta_785", "Eta_1550"]]
    _save(t5, "table5_qkd_performance", save_dir)

    print(f"\n[tables] All 5 tables exported to {save_dir}/")
