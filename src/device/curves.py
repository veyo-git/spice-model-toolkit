"""I-V curve generators and operating region analysis."""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple


class OperatingRegion(IntEnum):
    CUTOFF = 0
    LINEAR = 1
    SATURATION = 2
    SUBTHRESHOLD = 3


@dataclass
class IdVgSweep:
    """Transfer characteristics: Id vs Vgs at fixed Vds values.

    Attributes
    ----------
    vgs : ndarray (N,)
        Gate-source voltage sweep points (V).
    ids : ndarray (M, N)
        Drain current for each (Vds, Vgs) pair (A).
    vds_values : ndarray (M,)
        Drain-source voltages used.
    """

    vgs: np.ndarray
    ids: np.ndarray
    vds_values: np.ndarray


@dataclass
class IdVdSweep:
    """Output characteristics: Id vs Vds at fixed Vgs values.

    Attributes
    ----------
    vds : ndarray (N,)
        Drain-source voltage sweep points (V).
    ids : ndarray (M, N)
        Drain current for each (Vgs, Vds) pair (A).
    vgs_values : ndarray (M,)
        Gate-source voltages used.
    """

    vds: np.ndarray
    ids: np.ndarray
    vgs_values: np.ndarray


def generate_iv_curves(
    model,
    vgs_range=(-0.5, 2.5, 101),
    vds_range=(0.0, 2.5, 101),
    vsb=0.0,
):
    """Generate complete I-V characteristics for a MOSFET model.

    Parameters
    ----------
    model : MOSFETLevel1 or MOSFETLevel3
        The MOSFET model instance.
    vgs_range : tuple
        (V_GS_min, V_GS_max, n_points) for transfer curve.
    vds_range : tuple
        (V_DS_min, V_DS_max, n_points) for output curve.
    vsb : float
        Source-body bias (V).

    Returns
    -------
    id_vg : IdVgSweep
    id_vd : IdVdSweep
    """
    # --- Transfer: Id-Vg at multiple Vds ---
    vgs = np.linspace(*vgs_range)
    vds_vg = np.array([0.05, 0.3, 1.0, 2.5])
    # Filter to those within range
    vds_vg = vds_vg[(vds_vg >= vds_range[0]) & (vds_vg <= vds_range[1])]

    if len(vds_vg) == 0:
        vds_vg = np.array([0.1])

    ids_vg = np.zeros((len(vds_vg), len(vgs)))
    for i, vds_val in enumerate(vds_vg):
        ids_vg[i] = model.ids(vgs, vds_val, vsb)

    id_vg = IdVgSweep(vgs=vgs, ids=ids_vg, vds_values=vds_vg)

    # --- Output: Id-Vd at multiple Vgs ---
    vds = np.linspace(*vds_range)
    n_vgs = 6
    vgs_min = max(vgs_range[0], 0.0)
    vgs_max = vgs_range[1]
    vgs_vd = np.linspace(vgs_min + 0.2, vgs_max, n_vgs)

    ids_vd = np.zeros((len(vgs_vd), len(vds)))
    for i, vgs_val in enumerate(vgs_vd):
        ids_vd[i] = model.ids(vgs_val, vds, vsb)

    id_vd = IdVdSweep(vds=vds, ids=ids_vd, vgs_values=vgs_vd)

    return id_vg, id_vd


def classify_region(model, vgs, vds, vsb=0.0):
    """Classify operating region for given bias points.

    Parameters
    ----------
    model : MOSFET model instance
    vgs, vds : ndarray
        Bias voltages.

    Returns
    -------
    region : ndarray of OperatingRegion
    """
    vgs = np.asarray(vgs)
    vds = np.asarray(vds)

    if hasattr(model, "params"):
        vth0 = model.params.VTH0
    else:
        vth0 = 0.45

    v_gt = vgs - vth0
    region = np.full_like(vgs, OperatingRegion.CUTOFF, dtype=int)
    region[(v_gt > 0) & (vds < v_gt)] = OperatingRegion.LINEAR
    region[(v_gt > 0) & (vds >= v_gt)] = OperatingRegion.SATURATION
    # Subthreshold region (vgs < vth0 but > vth0 - 0.15)
    sub = (v_gt <= 0) & (vgs > vth0 - 0.15)
    region[sub] = OperatingRegion.SUBTHRESHOLD

    return region


def extract_vth_linear(model, vgs, vds=0.05, vsb=0.0):
    """Extract threshold voltage via linear extrapolation method.

    Finds V_TH as the x-intercept of max-slope tangent line on Id-Vg
    at low V_DS (linear region).

    Parameters
    ----------
    model : MOSFET model instance
    vgs : ndarray
        Gate voltage sweep (V).
    vds : float
        Small drain voltage for linear region (V).
    vsb : float
        Source-body bias (V).

    Returns
    -------
    vth_extracted : float
    """
    ids = model.ids(vgs, vds, vsb)
    gm = np.gradient(ids, vgs)
    idx_max_gm = np.argmax(gm)
    # Tangent line: I_D = gm_max * (V_GS - V_TH)
    gm_max = gm[idx_max_gm]
    ids_at_max = ids[idx_max_gm]
    vgs_at_max = vgs[idx_max_gm]
    vth_extracted = vgs_at_max - ids_at_max / gm_max
    return float(vth_extracted)
