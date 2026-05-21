"""Multi-curve weighted objective function for SPICE parameter extraction.

Design principle:
  - Id-Vg error evaluated in log10 space (covers 6+ decades from pA to mA)
  - Id-Vd error evaluated in linear space (focus on on-state accuracy)
  - Gm peak error added as a shape-sensitive term
  - Physics penalty terms keep parameters in physically meaningful ranges
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from src.device.mosfet import (
    MOSFETLevel1, MOSFETLevel3,
    MOSFETParamsLevel1, MOSFETParamsLevel3,
)


@dataclass
class CurveData:
    """Container for one set of measured/simulated I-V data."""

    vgs: np.ndarray       # Gate voltages (V)
    vds: np.ndarray       # Drain voltages (V)
    ids: np.ndarray       # Drain current (A) — shape (len(vds), len(vgs)) or (len(vgs),)
    ids_std: Optional[np.ndarray] = None  # Measurement std for weighted fitting

    @property
    def has_2d_sweep(self):
        return self.ids.ndim == 2


@dataclass
class ExtractionObjective:
    """Multi-curve weighted objective for parameter extraction.

    Total error = w_vg * E_IdVg + w_vd * E_IdVd + w_gm * E_GmPeak + penalties

    Parameters
    ----------
    target_idvg : CurveData
        Target transfer characteristics.
    target_idvd : CurveData, optional
        Target output characteristics.
    weights : tuple
        (w_vg, w_vd, w_gm) weights for each error component.
    log_floor : float
        Floor value for log10 to avoid log(0) issues (A).
    """

    target_idvg: CurveData
    target_idvd: Optional[CurveData] = None
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.3)
    log_floor: float = 1e-12

    # Physics bounds (VTH0 > 0, U0 in [50, 800], TOX > 0.5 nm, etc.)
    physics_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.physics_bounds:
            self.physics_bounds = {
                "VTH0": (0.05, 1.5),
                "U0": (50.0, 800.0),
                "TOX": (0.5e-9, 20e-9),
                "THETA": (0.0, 0.5),
                "VSAT": (5e6, 2e7),
                "ETA0": (0.0, 0.3),
                "GAMMA": (0.0, 2.0),
                "PHI": (0.3, 1.2),
                "LAMBDA": (0.0, 0.5),
            }

    def __call__(self, param_vector, model_cls=MOSFETLevel3):
        """Evaluate total error for a given parameter vector.

        Parameters
        ----------
        param_vector : ndarray (D,)
            Model parameters in canonical order (see PARAM_ORDER).
        model_cls : type
            MOSFETLevel1 or MOSFETLevel3.

        Returns
        -------
        error : float
        """
        params = vector_to_params(param_vector, model_cls)
        model = model_cls(params) if model_cls is MOSFETLevel3 else MOSFETLevel1(params)

        w_vg, w_vd, w_gm = self.weights
        total = 0.0

        # 1. Id-Vg error (log-space RMSE)
        e_vg = self._idvg_error(model)
        total += w_vg * e_vg

        # 2. Id-Vd error (linear-space RMSE), optional
        if self.target_idvd is not None:
            e_vd = self._idvd_error(model)
            total += w_vd * e_vd
        else:
            # Use the highest Vds from Id-Vg as a proxy
            pass

        # 3. Gm peak error
        if self.target_idvd is not None:
            e_gm = self._gm_peak_error(model)
            total += w_gm * e_gm

        # 4. Physics penalty terms
        penalties = self._physics_penalty(param_vector, model_cls)
        total += penalties

        return float(total)

    def _idvg_error(self, model):
        """Log-space RMSE for Id-Vg curves.

        CurveData layout for Id-Vg: vgs = Vgs axis (1D), vds = Vds bias list,
        ids = (n_vds, n_vgs) array.
        """
        tg = self.target_idvg
        ids_sim = np.zeros_like(tg.ids)
        for i, vds_val in enumerate(tg.vds):
            ids_sim[i] = model.ids(tg.vgs, vds_val)

        # Log-space: floor small values and take log10
        ids_tgt_log = np.log10(np.maximum(tg.ids, self.log_floor))
        ids_sim_log = np.log10(np.maximum(ids_sim, self.log_floor))

        return float(np.sqrt(np.mean((ids_tgt_log - ids_sim_log) ** 2)))

    def _idvd_error(self, model):
        """Linear-space RMSE for Id-Vd curves.

        CurveData layout for Id-Vd: vgs = Vgs bias list, vds = Vds axis (1D),
        ids = (n_vgs, n_vds) array.
        """
        tg = self.target_idvd
        ids_sim = np.zeros_like(tg.ids)
        for i, vgs_val in enumerate(tg.vgs):
            ids_sim[i] = model.ids(vgs_val, tg.vds)

        # Normalize by max current to keep scale ~1
        i_max = max(np.max(tg.ids), 1e-12)
        return float(np.sqrt(np.mean(((tg.ids - ids_sim) / i_max) ** 2)))

    def _gm_peak_error(self, model):
        """Relative error in Gm peak value."""
        tg = self.target_idvg
        # Use the highest Vds for Gm
        vds_high = tg.vds[-1]
        ids_sim = model.ids(tg.vgs, vds_high)
        gm_sim = np.gradient(ids_sim, tg.vgs)
        gm_tgt = np.gradient(tg.ids[-1], tg.vgs)

        gm_max_sim = np.max(gm_sim)
        gm_max_tgt = np.max(gm_tgt)
        if gm_max_tgt < 1e-15:
            return 0.0
        return float(abs(gm_max_sim - gm_max_tgt) / gm_max_tgt)

    def _physics_penalty(self, x, model_cls):
        """Penalty for violating physics bounds."""
        penalty = 0.0
        param_names = _get_param_order(model_cls)
        for i, name in enumerate(param_names):
            if name in self.physics_bounds:
                lo, hi = self.physics_bounds[name]
                if x[i] < lo:
                    penalty += (lo - x[i]) ** 2 * 10.0
                elif x[i] > hi:
                    penalty += (x[i] - hi) ** 2 * 10.0
        return penalty


# ---------------------------------------------------------------------------
# Parameter vector <-> dataclass conversion
# ---------------------------------------------------------------------------

def _get_param_order(model_cls=MOSFETLevel3):
    """Return canonical parameter order for a given model/params class."""
    if model_cls in (MOSFETLevel1, MOSFETParamsLevel1):
        return ["W", "L", "VTH0", "GAMMA", "PHI", "KP", "LAMBDA"]
    else:
        return [
            "W", "L", "VTH0", "GAMMA", "PHI", "U0", "THETA",
            "VSAT", "KAPPA", "LAMBDA", "ETA0", "N0", "RD", "RS",
        ]


def params_to_vector(params):
    """Convert a MOSFET parameter dataclass to a NumPy vector."""
    order = _get_param_order(type(params))
    return np.array([getattr(params, name) for name in order], dtype=float)


def vector_to_params(x, model_cls=MOSFETLevel3):
    """Convert a NumPy vector to a MOSFET parameter dataclass.

    Parameters
    ----------
    x : ndarray
        Parameter vector.
    model_cls : type
        MOSFETLevel1 or MOSFETLevel3 (the model class, not params class).

    Returns
    -------
    params : MOSFETParamsLevel1 or MOSFETParamsLevel3
    """
    order = _get_param_order(model_cls)
    kw = {name: float(x[i]) for i, name in enumerate(order)}

    if model_cls is MOSFETLevel1:
        kw.setdefault("NSUB", 1e22)
        kw.setdefault("TOX", 4e-9)
        kw.setdefault("T_nom", 300.15)
        filtered = {k: kw[k] for k in ["W", "L", "VTH0", "GAMMA", "PHI", "KP", "LAMBDA", "NSUB", "TOX", "T_nom"] if k in kw}
        return MOSFETParamsLevel1(**filtered)
    else:
        kw.setdefault("NSUB", 1e22)
        kw.setdefault("TOX", 4e-9)
        kw.setdefault("T_nom", 300.15)
        kw["KP"] = None  # auto-compute from U0
        return MOSFETParamsLevel3(**kw)


# For backward compatibility / convenience aliases
PARAM_ORDER_L1 = _get_param_order(MOSFETLevel1)
PARAM_ORDER_L3 = _get_param_order(MOSFETLevel3)


# ---------------------------------------------------------------------------
# Reduced optimization with fixed parameters
# ---------------------------------------------------------------------------

def build_reduced_objective(
    objective: ExtractionObjective,
    model_cls,
    fixed_params: dict,
):
    """Create a reduced objective where some parameters are held fixed.

    Parameters
    ----------
    objective : ExtractionObjective
        The full objective function.
    model_cls : type
        MOSFETLevel1 or MOSFETLevel3.
    fixed_params : dict
        Dict of param_name → fixed_value. These params are not optimized.

    Returns
    -------
    reduced_obj : callable
        Function of reduced parameter vector → error.
    reduced_bounds : list of (lo, hi)
        Bounds for the free parameters only.
    free_names : list of str
        Names of free parameters.
    """
    full_order = _get_param_order(model_cls)
    # Only consider fixed params that are in the optimization order
    effective_fixed = {k: v for k, v in fixed_params.items() if k in full_order}
    fixed_set = set(effective_fixed.keys())
    free_names = [name for name in full_order if name not in fixed_set]
    free_indices = [i for i, name in enumerate(full_order) if name not in fixed_set]

    # Get full bounds
    if model_cls is MOSFETLevel1:
        full_bounds, _ = _get_default_bounds_l1()
    else:
        full_bounds, _ = _get_default_bounds_l3()

    reduced_bounds = [full_bounds[i] for i in free_indices]

    def reduced_obj(x_reduced, *_args):
        # Build full parameter vector
        x_full = np.zeros(len(full_order))
        # Fill fixed params (only those in the optimization order)
        for name, val in effective_fixed.items():
            idx = full_order.index(name)
            x_full[idx] = val
        # Fill free params
        for j, idx in enumerate(free_indices):
            x_full[idx] = x_reduced[j]
        return objective(x_full, model_cls)

    return reduced_obj, reduced_bounds, free_names


def _get_default_bounds_l1():
    return [
        (5e-6, 50e-6),     # W
        (0.1e-6, 1e-6),    # L
        (0.1, 1.0),        # VTH0
        (0.1, 2.0),        # GAMMA
        (0.3, 1.2),        # PHI
        (5e-6, 5e-4),      # KP
        (0.0, 0.2),        # LAMBDA
    ], PARAM_ORDER_L1


def _get_default_bounds_l3():
    return [
        (5e-6, 50e-6),     # W
        (0.1e-6, 1e-6),    # L
        (0.1, 1.0),        # VTH0
        (0.1, 2.0),        # GAMMA
        (0.3, 1.2),        # PHI
        (50.0, 800.0),     # U0
        (0.0, 0.3),        # THETA
        (5e6, 2e7),        # VSAT
        (0.1, 2.0),        # KAPPA
        (0.0, 0.2),        # LAMBDA
        (0.0, 0.2),        # ETA0
        (1.0, 3.0),        # N0
        (0.0, 200.0),      # RD
        (0.0, 200.0),      # RS
    ], PARAM_ORDER_L3
