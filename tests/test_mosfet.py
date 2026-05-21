"""Tests for MOSFET Level-1 and Level-3 models."""

import numpy as np
import pytest
from src.device.mosfet import (
    MOSFETLevel1, MOSFETLevel3,
    MOSFETParamsLevel1, MOSFETParamsLevel3,
)
from src.device.curves import generate_iv_curves, classify_region, OperatingRegion


class TestMOSFETLevel1:
    def test_zero_current_below_threshold(self, model_l1):
        """Ids should be ~0 when Vgs < VTH."""
        ids = model_l1.ids(0.3, 1.0)  # Vgs=0.3 < VTH=0.45
        assert ids == pytest.approx(0.0, abs=1e-15)

    def test_current_increases_with_vgs(self, model_l1):
        """Ids should increase monotonically with Vgs."""
        i1 = model_l1.ids(0.6, 1.0)
        i2 = model_l1.ids(0.8, 1.0)
        i3 = model_l1.ids(1.0, 1.0)
        assert i1 < i2 < i3

    def test_current_increases_with_vds_linear(self, model_l1):
        """Ids should increase with Vds in linear region."""
        i1 = model_l1.ids(1.5, 0.1)
        i2 = model_l1.ids(1.5, 0.3)
        assert i1 < i2

    def test_saturation_flatness(self, model_l1):
        """Current should be nearly flat in saturation (with CLM slope)."""
        i1 = model_l1.ids(1.0, 0.5)
        i2 = model_l1.ids(1.0, 1.0)
        i3 = model_l1.ids(1.0, 2.0)
        # Should be close but increasing slightly due to LAMBDA
        assert i1 > 0
        assert i2 >= i1 * 0.9
        assert i3 >= i2  # Small increase from CLM

    def test_vectorized_operation(self, model_l1):
        vgs = np.linspace(0.5, 2.5, 50)
        ids = model_l1.ids(vgs, 1.0)
        assert ids.shape == vgs.shape
        assert np.all(np.diff(ids) >= -1e-15)  # Monotonic increasing

    def test_gm_positive_above_threshold(self, model_l1):
        gm = model_l1.gm(1.0, 1.0)
        assert gm > 0

    def test_gds_positive(self, model_l1):
        gds = model_l1.gds(1.0, 1.0)
        assert gds >= 0  # Non-negative output conductance

    def test_w_scaling(self, params_l1):
        """Ids should scale linearly with W."""
        p_wide = MOSFETParamsLevel1(W=20e-6, L=0.18e-6, VTH0=params_l1.VTH0,
                                    GAMMA=params_l1.GAMMA, PHI=params_l1.PHI,
                                    KP=params_l1.KP, LAMBDA=params_l1.LAMBDA)
        m1 = MOSFETLevel1(params_l1)
        m2 = MOSFETLevel1(p_wide)
        i1 = m1.ids(1.0, 1.0)
        i2 = m2.ids(1.0, 1.0)
        assert i2 == pytest.approx(2 * i1, rel=0.01)

    def test_l_scaling(self, params_l1):
        """Ids should scale inversely with L."""
        p_short = MOSFETParamsLevel1(W=10e-6, L=0.09e-6, VTH0=params_l1.VTH0,
                                     GAMMA=params_l1.GAMMA, PHI=params_l1.PHI,
                                     KP=params_l1.KP, LAMBDA=params_l1.LAMBDA)
        m1 = MOSFETLevel1(params_l1)
        m2 = MOSFETLevel1(p_short)
        i1 = m1.ids(1.0, 1.0)
        i2 = m2.ids(1.0, 1.0)
        assert i2 == pytest.approx(2 * i1, rel=0.01)


class TestMOSFETLevel3:
    def test_subthreshold_current(self, model_l3):
        """Level-3 should have non-zero subthreshold current."""
        ids = model_l3.ids(0.3, 1.0)  # Below VTH=0.45
        assert ids > 1e-12  # Subthreshold leakage
        assert ids < 1e-6   # Still small

    def test_above_threshold_larger(self, model_l3):
        """Above-threshold current should be much larger than subthreshold."""
        i_sub = model_l3.ids(0.3, 1.0)
        i_above = model_l3.ids(1.0, 1.0)
        assert i_above > 1000 * i_sub

    def test_velocity_saturation_effect(self):
        """Lower VSAT should reduce saturation current."""
        p_base = MOSFETParamsLevel3(U0=400, VSAT=1e7)
        p_vsat = MOSFETParamsLevel3(U0=400, VSAT=5e6)
        i_base = MOSFETLevel3(p_base).ids(1.5, 2.5)
        i_vsat = MOSFETLevel3(p_vsat).ids(1.5, 2.5)
        assert i_vsat < i_base  # Lower v_sat → lower current at high Vds

    def test_dibl_effect(self):
        """Higher ETA0 (DIBL) should increase current at high Vds."""
        p_base = MOSFETParamsLevel3(ETA0=0.01)
        p_dibl = MOSFETParamsLevel3(ETA0=0.15)
        i_base = MOSFETLevel3(p_base).ids(1.0, 2.5)
        i_dibl = MOSFETLevel3(p_dibl).ids(1.0, 2.5)
        assert i_dibl > i_base  # More DIBL → lower VTH → more current

    def test_rs_rd_degrade_current(self):
        """Parasitic resistance should reduce effective current."""
        p_base = MOSFETParamsLevel3(RD=0, RS=0)
        p_r = MOSFETParamsLevel3(RD=100, RS=100)
        i_base = MOSFETLevel3(p_base).ids(1.5, 1.0)
        i_r = MOSFETLevel3(p_r).ids(1.5, 1.0)
        assert i_r < i_base

    def test_vectorized_operation(self, model_l3, vgs_sweep):
        ids = model_l3.ids(vgs_sweep, 1.0)
        assert ids.shape == vgs_sweep.shape
        assert ids[-1] > ids[50]  # Higher at Vgs=2.5 than Vgs=1.0

    def test_gm_positive(self, model_l3):
        gm = model_l3.gm(1.0, 1.0)
        assert gm > 0

    def test_upgrade_from_level1(self, params_l1):
        model_l3 = MOSFETLevel3.from_level1(params_l1)
        # Should have similar on-state current
        i_l1 = MOSFETLevel1(params_l1).ids(1.5, 1.0)
        i_l3 = model_l3.ids(1.5, 1.0)
        assert i_l3 > 0
        # Level-3 with v_sat will be lower, but within order of magnitude
        assert i_l3 > i_l1 * 0.1


class TestIVCurves:
    def test_generate_iv_curves(self, model_l3):
        id_vg, id_vd = generate_iv_curves(model_l3)
        assert id_vg.ids.shape[0] == len(id_vg.vds_values)
        assert id_vg.ids.shape[1] == len(id_vg.vgs)
        assert id_vd.ids.shape[0] == len(id_vd.vgs_values)
        assert id_vd.ids.shape[1] == len(id_vd.vds)

    def test_operating_region_classification(self, model_l1):
        # Vgs=0V → cutoff
        r = classify_region(model_l1, np.array([0.0]), np.array([1.0]))
        assert r[0] == OperatingRegion.CUTOFF

        # Vgs=1.0, Vds=0.1 → linear
        r = classify_region(model_l1, np.array([1.0]), np.array([0.1]))
        assert r[0] == OperatingRegion.LINEAR

        # Vgs=1.0, Vds=2.0 → saturation
        r = classify_region(model_l1, np.array([1.0]), np.array([2.0]))
        assert r[0] == OperatingRegion.SATURATION
