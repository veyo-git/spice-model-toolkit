"""MOSFET SPICE model implementations: Level-1 (Shichman-Hodges) and Level-3.

Level-1: Long-channel square-law model (~10 parameters).
Level-3: Semi-empirical short-channel model (~20 parameters) with:
  - Velocity saturation (VSAT / KAPPA)
  - Mobility degradation (THETA)
  - DIBL (ETA0)
  - Subthreshold conduction (N0)
  - Channel-length modulation
  - Source/drain parasitic resistance (RD, RS)

All equations follow the SPICE3 reference formulation.
"""

import numpy as np
from dataclasses import dataclass, field
from src.device.physical import (
    PhysicalConstants, MobilityModel, ThresholdVoltage,
    thermal_voltage, _pc,
)


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MOSFETParamsLevel1:
    """MOSFET Level-1 (Shichman-Hodges) model parameters.

    Defaults represent a typical 0.18 µm NMOS device.
    """

    # Geometry
    W: float = 10e-6          # Gate width (m)
    L: float = 0.18e-6        # Gate length (m)

    # Threshold
    VTH0: float = 0.45        # Zero-bias threshold voltage (V)

    # Body effect
    GAMMA: float = 0.4        # Body-effect coefficient (√V)
    PHI: float = 0.65         # Surface potential = 2·φ_f (V)

    # Transconductance
    KP: float = 200e-6         # μ_n·C_ox (A/V²)

    # Channel-length modulation
    LAMBDA: float = 0.05       # Channel-length modulation (V⁻¹)

    # Substrate doping (for derived parameters)
    NSUB: float = 1e22         # Substrate doping (m⁻³) ≈ 1e16 cm⁻³

    # Oxide
    TOX: float = 4e-9          # Gate oxide thickness (m)

    # Temperature
    T_nom: float = 300.15      # Nominal temperature (K)


@dataclass
class MOSFETParamsLevel3:
    """MOSFET Level-3 (semi-empirical short-channel) model parameters."""

    # Geometry
    W: float = 10e-6
    L: float = 0.18e-6

    # Threshold
    VTH0: float = 0.45
    GAMMA: float = 0.4
    PHI: float = 0.65

    # Transconductance
    U0: float = 400.0          # Low-field mobility (cm²/V·s)
    KP: float = None           # If None, computed from U0 and TOX

    # Mobility degradation
    THETA: float = 0.05        # Vertical-field mobility degradation (V⁻¹)

    # Velocity saturation
    VSAT: float = 1.0e7        # Saturation velocity (cm/s)
    KAPPA: float = 0.5         # Lateral-field factor for v_sat (dimensionless)

    # Channel-length modulation
    LAMBDA: float = 0.05

    # DIBL
    ETA0: float = 0.05         # DIBL coefficient

    # Subthreshold
    N0: float = 1.5            # Subthreshold slope factor

    # Parasitic resistance
    RD: float = 20.0           # Drain parasitic resistance (Ω)
    RS: float = 20.0           # Source parasitic resistance (Ω)

    # Substrate
    NSUB: float = 1e22         # ≈ 1e16 cm⁻³

    # Oxide
    TOX: float = 4e-9

    # Temperature
    T_nom: float = 300.15

    def __post_init__(self):
        if self.KP is None:
            c_ox = 3.9 * 8.854187817e-12 / self.TOX
            # U0 is in cm²/V·s, convert to m²/V·s
            self.KP = self.U0 * 1e-4 * c_ox


# ---------------------------------------------------------------------------
# MOSFET Level-1 (Shichman-Hodges)
# ---------------------------------------------------------------------------

class MOSFETLevel1:
    """MOSFET Level-1: Shichman-Hodges long-channel square-law model.

    Operating regions:
      - Cutoff:    V_GS < V_TH  →  I_DS = 0
      - Linear:    V_DS < V_GS - V_TH  →  square-law with V_DS term
      - Saturation: V_DS >= V_GS - V_TH  →  constant current + λ modulation
    """

    def __init__(self, params=None):
        self.params = params or MOSFETParamsLevel1()
        self._pc = PhysicalConstants()
        self._tv = ThresholdVoltage()

    @property
    def p(self):
        return self.params

    def ids(self, vgs, vds, vsb=0.0):
        """Drain current for given bias points.

        Parameters
        ----------
        vgs, vds : float or ndarray
            Gate-source and drain-source voltages (V).
        vsb : float or ndarray
            Source-body voltage (V). Default 0.

        Returns
        -------
        ids : float or ndarray
            Drain current (A).
        """
        scalar_input = np.ndim(vgs) == 0 and np.ndim(vds) == 0
        vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
        vds = np.atleast_1d(np.asarray(vds, dtype=float))
        vsb = np.atleast_1d(np.asarray(vsb, dtype=float))

        # Broadcast vds to match vgs shape (needed when vgs is array, vds scalar)
        if vds.shape != vgs.shape:
            vds = np.full_like(vgs, vds[0])

        vth = self._vth(vsb)
        v_gt = vgs - vth  # gate overdrive

        kp = self.p.KP
        w_over_l = self.p.W / self.p.L
        beta = kp * w_over_l

        # Region masks
        linear = (v_gt > 0) & (vds < v_gt)
        sat = (v_gt > 0) & (vds >= v_gt)

        ids = np.zeros_like(vgs)

        # Linear region: I_DS = β · [(V_GS - V_TH) · V_DS - V_DS²/2]
        ids[linear] = (
            beta
            * (v_gt[linear] * vds[linear] - 0.5 * vds[linear] ** 2)
            * (1.0 + self.p.LAMBDA * vds[linear])
        )

        # Saturation region: I_DS = β/2 · (V_GS - V_TH)² · (1 + λ·V_DS)
        ids[sat] = (
            0.5 * beta * v_gt[sat] ** 2
            * (1.0 + self.p.LAMBDA * vds[sat])
        )

        if scalar_input:
            return float(ids[0])
        return ids

    def _vth(self, vsb=0.0):
        """Threshold voltage with body effect."""
        phi = self.p.PHI
        gamma = self.p.GAMMA
        sqrt_phi = np.sqrt(np.maximum(phi + vsb, 0.0))
        return self.p.VTH0 + gamma * (sqrt_phi - np.sqrt(phi))

    def gm(self, vgs, vds, vsb=0.0):
        """Transconductance g_m = ∂I_DS / ∂V_GS (S)."""
        scalar_input = np.ndim(vgs) == 0 and np.ndim(vds) == 0
        vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
        vds = np.atleast_1d(np.asarray(vds, dtype=float))
        vth = self._vth(vsb)
        v_gt = vgs - vth
        beta = self.p.KP * self.p.W / self.p.L

        gm = np.zeros_like(vgs)
        linear = (v_gt > 0) & (vds < v_gt)
        sat = (v_gt > 0) & (vds >= v_gt)

        gm[linear] = beta * vds[linear] * (1.0 + self.p.LAMBDA * vds[linear])
        gm[sat] = beta * v_gt[sat] * (1.0 + self.p.LAMBDA * vds[sat])

        if scalar_input:
            return float(gm[0])
        return gm

    def gds(self, vgs, vds, vsb=0.0):
        """Output conductance g_ds = ∂I_DS / ∂V_DS (S)."""
        scalar_input = np.ndim(vgs) == 0 and np.ndim(vds) == 0
        vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
        vds = np.atleast_1d(np.asarray(vds, dtype=float))
        vth = self._vth(vsb)
        v_gt = vgs - vth
        beta = self.p.KP * self.p.W / self.p.L

        gds = np.zeros_like(vgs)
        linear = (v_gt > 0) & (vds < v_gt)
        sat = (v_gt > 0) & (vds >= v_gt)

        gds[linear] = beta * (
            v_gt[linear] - vds[linear]
            + self.p.LAMBDA * (2 * v_gt[linear] * vds[linear] - 1.5 * vds[linear] ** 2)
        )
        gds[sat] = 0.5 * beta * v_gt[sat] ** 2 * self.p.LAMBDA

        if scalar_input:
            return float(gds[0])
        return gds


# ---------------------------------------------------------------------------
# MOSFET Level-3 (semi-empirical short-channel)
# ---------------------------------------------------------------------------

class MOSFETLevel3:
    """MOSFET Level-3: Semi-empirical short-channel model.

    Extends Level-1 with:
      - Subthreshold current (exponential tail below VTH)
      - Velocity saturation (VSAT, KAPPA)
      - Mobility degradation (THETA)
      - DIBL (ETA0)
      - Source/drain resistance (RD, RS) via iterative voltage correction
    """

    def __init__(self, params=None):
        self.params = params or MOSFETParamsLevel3()
        self.mobility = MobilityModel(
            u0=self.params.U0,
            vsat=self.params.VSAT,
            theta=self.params.THETA,
        )
        self._tv = ThresholdVoltage()

    @property
    def p(self):
        return self.params

    def ids(self, vgs, vds, vsb=0.0):
        """Drain current including subthreshold, DIBL, v_sat, and R_S/R_D.

        Parameters
        ----------
        vgs, vds : float or ndarray
            Applied terminal voltages (V). Internal node voltages are
            corrected for parasitic source/drain resistance.
        vsb : float
            Source-body voltage (V).

        Returns
        -------
        ids : float or ndarray
            Drain current (A).
        """
        vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
        vds = np.atleast_1d(np.asarray(vds, dtype=float))
        scalar_in = vgs.ndim == 0

        # --- Internal node correction for R_S, R_D ---
        # Solve: V_GS_int = V_GS_app - I_DS * R_S
        #        V_DS_int = V_DS_app - I_DS * (R_S + R_D)
        # Simple iterative correction (2-3 iterations are enough)
        ids_guess = np.zeros_like(vgs)
        vgs_int = vgs.copy()
        vds_int = vds.copy()
        for _ in range(3):
            ids_guess = self._ids_core(vgs_int, vds_int, vsb)
            vgs_int = vgs - ids_guess * self.p.RS
            vds_int = vds - ids_guess * (self.p.RS + self.p.RD)
            vgs_int = np.maximum(vgs_int, -0.1)  # prevent runaway
            vds_int = np.maximum(vds_int, 0.0)

        ids_val = self._ids_core(vgs_int, vds_int, vsb)
        if scalar_in:
            return float(ids_val[0])
        return ids_val

    def _ids_core(self, vgs, vds, vsb=0.0):
        """Core drain current without parasitic resistance.

        Combines subthreshold and above-threshold currents with a
        smooth transition via the BSIM-style effective V_GS approach.
        """
        vth = self._vth_effective(vds, vsb)
        n0 = self.p.N0
        vt = thermal_voltage()

        # Subthreshold current (exponential)
        v_eff = vgs - vth
        # Smooth effective gate voltage (EKV-style interpolation)
        x = v_eff / (n0 * vt)
        # Avoid overflow in exp
        x_clipped = np.clip(x, -50, 50)
        v_eff_smooth = n0 * vt * np.log(1.0 + np.exp(x_clipped))

        # Above-threshold current with mobility degradation and v_sat
        ids_above = self._ids_above_threshold(v_eff_smooth, vds, vth)

        return ids_above

    def _ids_above_threshold(self, v_eff, vds, vth):
        """Above-threshold drain current with velocity saturation."""
        kp = self.p.KP
        w = self.p.W
        l = self.p.L
        beta = kp * w / l
        theta = self.p.THETA
        kappa = self.p.KAPPA
        vsat = self.p.VSAT

        # Velocity saturation factor
        # I_DS saturation voltage modified by v_sat:
        # V_DSsat = (V_GS - V_TH) / (1 + KAPPA)  (simplified level-3)
        # More precisely:
        # V_DSsat = V_eff / (1 + (beta*V_eff)/(W*vsat*C_ox) )
        c_ox = 3.9 * 8.854187817e-12 / self.p.TOX
        denom = 1.0 + (self.p.U0 * 1e-4) * v_eff / (vsat * 1e-2 * l * (1.0 + theta * v_eff))
        denom = np.maximum(denom, 0.1)
        v_dsat = np.maximum(v_eff / denom, 0.0)

        # Effective drain voltage (smooth transition to saturation)
        # V_DS_eff = V_DS / (1 + (V_DS / V_DSsat)^m)^(1/m)
        m = 3.0  # smoothing exponent
        ratio = np.where(v_dsat > 1e-12, vds / v_dsat, 1e6)
        vds_eff = vds / (1.0 + ratio ** m) ** (1.0 / m)

        # Drain current with mobility degradation
        mu_factor = 1.0 / (1.0 + theta * v_eff)
        ids = beta * mu_factor * v_eff * vds_eff * (1.0 - 0.5 * vds_eff / np.maximum(v_eff, 1e-12))

        # Channel-length modulation
        ids = ids * (1.0 + self.p.LAMBDA * vds)

        return np.maximum(ids, 0.0)

    def _vth_effective(self, vds, vsb=0.0):
        """Effective threshold including body effect and DIBL."""
        phi = self.p.PHI
        gamma = self.p.GAMMA
        # Body effect
        sqrt_term = np.sqrt(np.maximum(phi + vsb, 0.0))
        vth0_body = self.p.VTH0 + gamma * (sqrt_term - np.sqrt(phi))
        # DIBL
        delta_vds = np.maximum(vds - 0.1, 0.0)
        vth = vth0_body - self.p.ETA0 * delta_vds
        return np.maximum(vth, 0.05)

    def gm(self, vgs, vds, vsb=0.0):
        """Transconductance via numerical differentiation."""
        eps = 1e-6
        ids_p = self.ids(vgs + eps, vds, vsb)
        ids_m = self.ids(vgs - eps, vds, vsb)
        return (ids_p - ids_m) / (2 * eps)

    def gds(self, vgs, vds, vsb=0.0):
        """Output conductance via numerical differentiation."""
        eps = 1e-6
        ids_p = self.ids(vgs, vds + eps, vsb)
        ids_m = self.ids(vgs, vds - np.where(vds > eps, eps, 0), vsb)
        return (ids_p - ids_m) / (2 * eps)

    @classmethod
    def from_level1(cls, params_l1, **kwargs):
        """Upgrade a Level-1 parameter set to Level-3."""
        p3 = MOSFETParamsLevel3(
            W=params_l1.W,
            L=params_l1.L,
            VTH0=params_l1.VTH0,
            GAMMA=params_l1.GAMMA,
            PHI=params_l1.PHI,
            NSUB=params_l1.NSUB,
            TOX=params_l1.TOX,
            **kwargs,
        )
        return cls(p3)
