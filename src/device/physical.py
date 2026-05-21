"""Physical constants, mobility models, and threshold voltage calculations."""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants (SI units)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants used in semiconductor device physics."""

    q: float = 1.602176634e-19       # Elementary charge (C)
    k_B: float = 1.380649e-23        # Boltzmann constant (J/K)
    eps_0: float = 8.854187817e-12   # Vacuum permittivity (F/m)
    eps_si: float = 11.7 * 8.854187817e-12  # Si permittivity
    eps_sio2: float = 3.9 * 8.854187817e-12  # SiO2 permittivity
    n_i: float = 1.45e16             # Intrinsic carrier concentration (m⁻³) at 300K
    T_nom: float = 300.15            # Nominal temperature (K) — ~27°C

    @property
    def V_T(self) -> float:
        """Thermal voltage at nominal temperature (V)."""
        return self.k_B * self.T_nom / self.q


# Global instance for convenience
_pc = PhysicalConstants()


def thermal_voltage(T=None):
    """Return kT/q in volts. If T is None, uses nominal 300.15 K."""
    if T is None:
        return _pc.V_T
    return _pc.k_B * T / _pc.q


# ---------------------------------------------------------------------------
# Mobility models
# ---------------------------------------------------------------------------

class MobilityModel:
    """Field-dependent carrier mobility models.

    Parameters
    ----------
    u0 : float
        Low-field mobility (cm²/V·s).
    vsat : float
        Saturation velocity (cm/s).
    theta : float
        Mobility degradation factor (1/V) for vertical field.
    """

    def __init__(self, u0=400.0, vsat=8e6, theta=0.05):
        self.u0 = u0
        self.vsat = vsat
        self.theta = theta

    def effective_mobility(
        self, vgs, vth, vds=0.0, tox=2e-9, eeff_model="simple"
    ):
        """Effective mobility including vertical and lateral field degradation.

        The simple model (level-1 compatible):
            mu_eff = u0 / (1 + theta * (vgs - vth))

        The advanced model adds lateral-field (velocity saturation) term:
            mu_eff = mu_vert / (1 + (mu_vert * vds) / (vsat * Leff))

        Parameters
        ----------
        vgs, vds : float or ndarray
            Gate-source and drain-source voltages (V).
        vth : float
            Threshold voltage (V).
        tox : float
            Oxide thickness (m).
        eeff_model : str
            "simple" or "advanced".

        Returns
        -------
        mu_eff : float or ndarray
            Effective mobility (cm²/V·s).
        """
        # Convert u0 from cm²/V·s to m²/V·s for SI consistency, then back
        v_od = np.maximum(vgs - vth, 0.0)

        # Vertical field degradation (surface roughness + phonon scattering)
        if eeff_model == "simple":
            mu_v = self.u0 / (1.0 + self.theta * v_od)
        else:
            # Effective vertical field ~ (vgs + vth) / (2 * tox)
            e_eff = (vgs + vth) / (2.0 * tox) * 1e-6  # MV/cm
            mu_v = self.u0 / (1.0 + (e_eff / 0.8) ** 1.6)

        # Lateral field (velocity saturation)
        if np.any(vds > 0):
            # Leff is set in the MOSFET model; use a default of 0.18 um here
            leff = 0.18e-6
            denom = 1.0 + (mu_v * vds) / (self.vsat * leff)
            denom = np.maximum(denom, 1.0)
            mu_eff = mu_v / denom
        else:
            mu_eff = mu_v

        return np.clip(mu_eff, 1.0, self.u0)


# ---------------------------------------------------------------------------
# Threshold voltage
# ---------------------------------------------------------------------------

class ThresholdVoltage:
    """MOSFET threshold voltage model (body-effect and DIBL included)."""

    def __init__(self, pc=None):
        self.pc = pc or _pc

    def vth0_ideal(self, phi_f, gamma, v_sb=0):
        """Ideal threshold: VTH0 + GAMMA * (√(2φ_f + V_SB) - √(2φ_f)).

        Parameters
        ----------
        phi_f : float
            Fermi potential (V). φ_f = V_T * ln(N_sub / n_i).
        gamma : float
            Body-effect coefficient (√V). γ = √(2q·ε_si·N_sub) / C_ox.
        v_sb : float or ndarray
            Source-body bias (V). Usually 0 for common-source.

        Returns
        -------
        vth : float or ndarray
            Threshold voltage (V).
        """
        sqrt_term = np.sqrt(np.maximum(2.0 * phi_f + v_sb, 0.0))
        return 2.0 * phi_f + gamma * (sqrt_term - np.sqrt(2.0 * phi_f))

    def vth_with_dibl(self, vth0, vds, eta0=0.05, vds_ref=0.1):
        """Apply DIBL correction: VTH = VTH0 - ETA0 * (V_DS - V_DS_ref).

        Parameters
        ----------
        vth0 : float or ndarray
            Base threshold voltage (V).
        vds : float or ndarray
            Drain-source voltage (V).
        eta0 : float
            DIBL coefficient (dimensionless), typically 0.02–0.10.
        vds_ref : float
            Reference V_DS at which VTH0 is defined (V).

        Returns
        -------
        vth : float or ndarray
            DIBL-corrected threshold voltage (V).
        """
        delta = np.maximum(vds - vds_ref, 0.0)
        return np.maximum(vth0 - eta0 * delta, 0.05)  # floor at 50 mV

    def compute_phi_f(self, n_sub):
        """Compute Fermi potential φ_f = V_T * ln(N_sub / n_i)."""
        return self.pc.V_T * np.log(n_sub / self.pc.n_i)

    def compute_gamma(self, n_sub, c_ox):
        """Compute body-effect coefficient γ = √(2q·ε_si·N_sub) / C_ox."""
        return np.sqrt(2.0 * self.pc.q * self.pc.eps_si * n_sub) / c_ox

    def compute_c_ox(self, tox):
        """Gate oxide capacitance per unit area C_ox = ε_SiO2 / tox (F/m²)."""
        return self.pc.eps_sio2 / tox
