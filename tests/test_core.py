"""
tests/test_core.py
==================
Unit tests for the CAQC core physics engine.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.atmosphere import (
    air_mass,
    rayleigh_optical_depth,
    rayleigh_loss_dB,
    aerosol_loss_dB,
    gas_optical_depth,
    slant_atm_loss_dB,
    itu_loss_dB,
    aod_at_wavelength,
)
from core.turbulence import (
    Cn2_HV,
    integrated_Cn2,
    fried_parameter,
    rytov_variance,
    turbulence_eta,
    turbulence_loss_dB,
)
from core.geometry import (
    rayleigh_range,
    beam_radius_at_range,
    geometric_eta,
    geometric_loss_dB,
)
from core.qkd import (
    binary_entropy,
    compute_qber,
    compute_skr_per_pulse,
    loss_dB_to_eta,
)
from config.settings import LAMBDA_M, LAMBDA_M_ALT


# ─────────────────────────────────────────────────────────────────────────────
# ATMOSPHERE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAtmosphere:

    def test_air_mass_zenith(self):
        """At 90° (zenith), air mass = 1."""
        assert math.isclose(air_mass(90.0), 1.0, rel_tol=1e-6)

    def test_air_mass_low_elev(self):
        """At 30°, air mass ≈ 2."""
        assert math.isclose(air_mass(30.0), 2.0, rel_tol=1e-4)

    def test_rayleigh_optical_depth_positive(self):
        tau = rayleigh_optical_depth(LAMBDA_M, 1013.0, 15.0)
        assert tau > 0

    def test_rayleigh_wavelength_scaling(self):
        """Rayleigh scales as λ⁻⁴ — longer wavelength gives lower τ."""
        tau_785  = rayleigh_optical_depth(LAMBDA_M,     1013, 15)
        tau_1550 = rayleigh_optical_depth(LAMBDA_M_ALT, 1013, 15)
        assert tau_1550 < tau_785

    def test_rayleigh_loss_positive(self):
        assert rayleigh_loss_dB(LAMBDA_M, 870, 15) > 0

    def test_aerosol_angstrom_scaling(self):
        """AOD(1550) < AOD(785) for α > 0."""
        aod785  = aod_at_wavelength(0.1, LAMBDA_M)
        aod1550 = aod_at_wavelength(0.1, LAMBDA_M_ALT)
        assert aod1550 < aod785

    def test_gas_optical_depth_positive(self):
        tau = gas_optical_depth(870, 50)
        assert tau > 0

    def test_gas_humidity_dependence(self):
        """Higher RH → higher gas optical depth."""
        tau_low  = gas_optical_depth(870, 10)
        tau_high = gas_optical_depth(870, 90)
        assert tau_high > tau_low

    def test_slant_loss_increases_with_lower_elev(self):
        """Lower elevation → longer path → higher loss."""
        loss_hi = slant_atm_loss_dB(LAMBDA_M, 70, 870, 15, 45, 0.1)
        loss_lo = slant_atm_loss_dB(LAMBDA_M, 20, 870, 15, 45, 0.1)
        assert loss_lo > loss_hi

    def test_itu_loss_positive(self):
        assert itu_loss_dB(500, LAMBDA_M) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TURBULENCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTurbulence:

    def test_cn2_profile_positive(self):
        h = np.linspace(0, 20_000, 100)
        assert np.all(Cn2_HV(h) > 0)

    def test_cn2_decreases_with_altitude(self):
        """Above the ground layer, Cn² should generally decrease."""
        Cn2 = Cn2_HV(np.array([1000.0, 10_000.0]))
        # not strictly monotone, but high-altitude should be lower
        assert Cn2[1] < Cn2[0]

    def test_integrated_cn2_positive(self):
        J = integrated_Cn2(45.0)
        assert J > 0

    def test_fried_positive(self):
        J = integrated_Cn2(45.0)
        r0 = fried_parameter(LAMBDA_M, J)
        assert r0 > 0

    def test_fried_wavelength_scaling(self):
        """r₀ increases with wavelength (λ^(6/5) scaling)."""
        J = integrated_Cn2(45.0)
        r0_785  = fried_parameter(LAMBDA_M,     J)
        r0_1550 = fried_parameter(LAMBDA_M_ALT, J)
        assert r0_1550 > r0_785

    def test_rytov_variance_positive(self):
        J = integrated_Cn2(45.0)
        assert rytov_variance(LAMBDA_M, J) > 0

    def test_turbulence_eta_range(self):
        for sigma in [0.1, 1.0, 5.0]:
            eta = turbulence_eta(sigma)
            assert 0 < eta <= 1

    def test_turbulence_loss_returns_triple(self):
        result = turbulence_loss_dB(LAMBDA_M, 45.0)
        assert len(result) == 3
        loss_dB, sigma, r0 = result
        assert loss_dB >= 0
        assert sigma >= 0
        assert r0 > 0

    def test_turbulence_loss_increases_lower_elev(self):
        loss_hi, _, _ = turbulence_loss_dB(LAMBDA_M, 70.0)
        loss_lo, _, _ = turbulence_loss_dB(LAMBDA_M, 15.0)
        assert loss_lo > loss_hi


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometry:

    def test_rayleigh_range_positive(self):
        assert rayleigh_range(LAMBDA_M) > 0

    def test_beam_radius_increases_with_range(self):
        w1 = beam_radius_at_range(100_000, LAMBDA_M)
        w2 = beam_radius_at_range(500_000, LAMBDA_M)
        assert w2 > w1

    def test_geometric_eta_in_range(self):
        eta = geometric_eta(500_000, LAMBDA_M)
        assert 0 < eta <= 1

    def test_geometric_eta_decreases_with_range(self):
        eta_near = geometric_eta(200_000, LAMBDA_M)
        eta_far  = geometric_eta(900_000, LAMBDA_M)
        assert eta_far < eta_near

    def test_geometric_loss_positive(self):
        assert geometric_loss_dB(500_000, LAMBDA_M) > 0

    def test_geometric_loss_increases_with_range(self):
        l1 = geometric_loss_dB(300_000, LAMBDA_M)
        l2 = geometric_loss_dB(800_000, LAMBDA_M)
        assert l2 > l1


# ─────────────────────────────────────────────────────────────────────────────
# QKD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestQKD:

    def test_binary_entropy_half(self):
        """h(0.5) = 1.0 bit."""
        assert math.isclose(binary_entropy(0.5), 1.0, rel_tol=1e-6)

    def test_binary_entropy_extremes(self):
        """h(0) = h(1) = 0."""
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_qber_positive(self):
        assert compute_qber(0.5) >= 0

    def test_qber_high_loss(self):
        """Near-zero transmission gives QBER → 0.5."""
        assert compute_qber(1e-10) == pytest.approx(0.5, abs=0.01)

    def test_qber_good_channel(self):
        """η = 1 gives QBER = 0.02 (optical alignment floor)."""
        assert compute_qber(1.0) == pytest.approx(0.02, abs=0.001)

    def test_skr_positive(self):
        """Good channel should give positive SKR."""
        assert compute_skr_per_pulse(0.5) > 0

    def test_skr_bad_channel(self):
        """Very low efficiency gives SKR → 0."""
        assert compute_skr_per_pulse(1e-12) == pytest.approx(0.0, abs=1e-15)

    def test_loss_to_eta_roundtrip(self):
        """10 dB loss → η = 0.1."""
        assert math.isclose(loss_dB_to_eta(10.0), 0.1, rel_tol=1e-6)

    def test_loss_to_eta_zero(self):
        """0 dB loss → η = 1."""
        assert math.isclose(loss_dB_to_eta(0.0), 1.0, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end sanity check without ERA5."""

    def test_full_link_budget(self):
        """Compute a complete link budget and verify physical ranges."""
        elev_deg = 45.0
        range_km = 600.0
        L = range_km * 1000.0

        g785 = geometric_loss_dB(L, LAMBDA_M)
        a785 = slant_atm_loss_dB(LAMBDA_M, elev_deg, 870, 15, 45, 0.1)
        t785, _, _ = turbulence_loss_dB(LAMBDA_M, elev_deg)

        total = g785 + a785 + t785

        assert 10 < total < 100, f"Total loss {total:.1f} dB outside plausible range"

        eta = loss_dB_to_eta(total)
        skr = compute_skr_per_pulse(eta)

        assert 0 <= skr < 1.0

    def test_1550_lower_loss_than_785(self):
        """1550 nm should have lower atmospheric loss than 785 nm."""
        loss785  = slant_atm_loss_dB(LAMBDA_M,     45, 870, 15, 45, 0.1)
        loss1550 = slant_atm_loss_dB(LAMBDA_M_ALT, 45, 870, 15, 45, 0.1)
        assert loss1550 < loss785
