"""Tests for physical constants, mobility models, and threshold voltage."""

import numpy as np
import pytest
from src.device.physical import (
    PhysicalConstants, MobilityModel, ThresholdVoltage,
    thermal_voltage, _pc,
)


class TestPhysicalConstants:
    def test_thermal_voltage_300K(self):
        vt = thermal_voltage()
        assert 0.025 < vt < 0.027  # ~0.02585 V at 300 K

    def test_thermal_voltage_400K(self):
        vt = thermal_voltage(400)
        assert vt > thermal_voltage(300)

    def test_dielectric_constants(self):
        pc = PhysicalConstants()
        assert pc.eps_sio2 > pc.eps_0  # SiO2 has higher permittivity
        assert pc.eps_si > pc.eps_sio2  # Si has even higher

    def test_intrinsic_carrier(self):
        pc = PhysicalConstants()
        assert 1e15 < pc.n_i < 1e17  # ~1.45e16 at 300K


class TestMobilityModel:
    def test_zero_field_mobility(self):
        mm = MobilityModel(u0=400)
        mu = mm.effective_mobility(0.0, 0.0, 0.0)
        assert mu == pytest.approx(400, rel=0.01)

    def test_vertical_field_degradation(self):
        mm = MobilityModel(u0=400, theta=0.1)
        mu_high = mm.effective_mobility(2.0, 0.45)   # Vgs-Vth = 1.55V
        mu_low = mm.effective_mobility(0.5, 0.45)     # Vgs-Vth = 0.05V
        assert mu_high < mu_low  # Degradation at higher Vgs

    def test_mobility_never_negative(self):
        mm = MobilityModel(u0=400)
        mu = mm.effective_mobility(10.0, 0.45, 5.0)
        assert mu >= 1.0  # Clipped to minimum


class TestThresholdVoltage:
    def test_phi_f_computation(self):
        tv = ThresholdVoltage()
        phi_f = tv.compute_phi_f(1e22)  # ~1e16 cm⁻³
        assert 0.2 < phi_f < 0.5  # ~0.35V for typical doping

    def test_c_ox_computation(self):
        tv = ThresholdVoltage()
        c_ox = tv.compute_c_ox(4e-9)  # 4 nm oxide
        assert 8e-3 < c_ox < 9e-3  # ~8.63e-3 F/m²

    def test_vth_body_effect(self):
        tv = ThresholdVoltage()
        vth0 = tv.vth0_ideal(phi_f=0.35, gamma=0.4, v_sb=0)
        vth1 = tv.vth0_ideal(phi_f=0.35, gamma=0.4, v_sb=1.0)
        assert vth1 > vth0  # Body effect increases VTH

    def test_dibl_reduces_vth(self):
        tv = ThresholdVoltage()
        vth_low_vds = tv.vth_with_dibl(0.5, vds=0.1, eta0=0.05)
        vth_high_vds = tv.vth_with_dibl(0.5, vds=2.0, eta0=0.05)
        assert vth_high_vds < vth_low_vds  # DIBL lowers VTH at high Vds
