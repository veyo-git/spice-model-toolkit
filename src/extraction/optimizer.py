"""Global and local optimizers for SPICE parameter extraction.

Strategy: Two-stage extraction
  1. Global search: Differential Evolution (DE) or Particle Swarm (PSO)
  2. Local refinement: Levenberg-Marquardt (LM)

All optimizers operate on parameter vectors and use the ExtractionObjective
as their cost function.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from src.device.mosfet import MOSFETLevel1, MOSFETLevel3, MOSFETParamsLevel1, MOSFETParamsLevel3
from src.extraction.objective import (
    ExtractionObjective, params_to_vector, vector_to_params,
    _get_param_order, PARAM_ORDER_L1, PARAM_ORDER_L3,
)


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

def get_default_bounds(model_cls=MOSFETLevel3):
    """Get default parameter bounds for optimization.

    Returns
    -------
    bounds : list of (lo, hi)
    param_names : list of str
    """
    if model_cls is MOSFETLevel1:
        return [
            (5e-6, 50e-6),     # W
            (0.1e-6, 1e-6),    # L
            (0.1, 1.0),        # VTH0
            (0.1, 2.0),        # GAMMA
            (0.3, 1.2),        # PHI
            (5e-6, 5e-4),      # KP
            (0.0, 0.2),        # LAMBDA
        ], PARAM_ORDER_L1
    else:
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


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

@dataclass
class DEOptions:
    """Differential Evolution hyperparameters."""

    max_iter: int = 1000
    pop_size: int = 30   # 10x–15x dimensionality recommended
    mutation: tuple = (0.5, 1.5)
    recombination: float = 0.7
    seed: int = 42
    tol: float = 1e-6
    workers: int = 1     # Use -1 for all cores


class DifferentialEvolution:
    """Differential Evolution global optimizer for parameter extraction.

    DE is robust for high-dimensional, multi-modal problems and does not
    require gradient information.
    """

    def __init__(self, objective: ExtractionObjective, model_cls=MOSFETLevel3):
        self.objective = objective
        self.model_cls = model_cls

    def optimize(
        self,
        bounds=None,
        options: DEOptions = None,
        callback=None,
        verbose=True,
    ):
        """Run DE optimization.

        Parameters
        ----------
        bounds : list of tuples, optional
            Parameter bounds. If None, uses default bounds.
        options : DEOptions, optional
        callback : callable, optional
            Called after each iteration.
        verbose : bool

        Returns
        -------
        result : dict
            Keys: x (optimal params), fun (final cost), nfev, nit, success, message
        """
        opts = options or DEOptions()
        if bounds is None:
            bounds, _ = get_default_bounds(self.model_cls)

        def cost(x):
            return self.objective(x, self.model_cls)

        result = differential_evolution(
            cost,
            bounds,
            maxiter=opts.max_iter,
            popsize=opts.pop_size,
            mutation=opts.mutation,
            recombination=opts.recombination,
            seed=opts.seed,
            tol=opts.tol,
            workers=opts.workers,
            polish=False,  # We do our own polish with LM
            disp=verbose,
            callback=callback,
        )

        return {
            "x": result.x,
            "fun": result.fun,
            "nfev": result.nfev,
            "nit": result.nit,
            "success": result.success,
            "message": result.message,
        }


# ---------------------------------------------------------------------------
# Particle Swarm Optimization (custom implementation)
# ---------------------------------------------------------------------------

class ParticleSwarm:
    """Particle Swarm Optimization for continuous parameter extraction.

    PSO often converges faster than DE for well-behaved cost surfaces.
    """

    def __init__(self, objective: ExtractionObjective, model_cls=MOSFETLevel3):
        self.objective = objective
        self.model_cls = model_cls

    def optimize(
        self,
        bounds=None,
        n_particles=30,
        max_iter=200,
        w_start=0.9,
        w_end=0.4,
        c1=2.0,
        c2=2.0,
        seed=42,
        verbose=True,
    ):
        """Run PSO optimization.

        Parameters
        ----------
        bounds : list of tuples, optional
        n_particles : int
            Swarm size.
        max_iter : int
            Maximum iterations.
        w_start, w_end : float
            Inertia weight schedule (linear decay).
        c1, c2 : float
            Cognitive and social acceleration coefficients.
        seed : int
        verbose : bool

        Returns
        -------
        result : dict
        """
        rng = np.random.RandomState(seed)
        if bounds is None:
            bounds, _ = get_default_bounds(self.model_cls)

        n_dim = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initialize particles
        pos = lo + rng.rand(n_particles, n_dim) * (hi - lo)
        vel = np.zeros((n_particles, n_dim))
        vel_max = 0.2 * (hi - lo)

        # Evaluate initial cost
        cost = np.array([self.objective(p, self.model_cls) for p in pos])
        p_best_pos = pos.copy()
        p_best_cost = cost.copy()
        g_best_idx = np.argmin(cost)
        g_best_pos = pos[g_best_idx].copy()
        g_best_cost = cost[g_best_idx]

        nfev = n_particles
        history = [g_best_cost]

        for it in range(max_iter):
            w = w_start - (w_start - w_end) * it / max_iter

            r1 = rng.rand(n_particles, n_dim)
            r2 = rng.rand(n_particles, n_dim)

            vel = (
                w * vel
                + c1 * r1 * (p_best_pos - pos)
                + c2 * r2 * (g_best_pos - pos)
            )
            vel = np.clip(vel, -vel_max, vel_max)

            pos = pos + vel
            pos = np.clip(pos, lo, hi)

            cost = np.array([self.objective(p, self.model_cls) for p in pos])
            nfev += n_particles

            improved = cost < p_best_cost
            p_best_pos[improved] = pos[improved]
            p_best_cost[improved] = cost[improved]

            new_g_best = np.argmin(p_best_cost)
            if p_best_cost[new_g_best] < g_best_cost:
                g_best_cost = p_best_cost[new_g_best]
                g_best_pos = p_best_pos[new_g_best].copy()

            history.append(g_best_cost)

            if verbose and (it + 1) % 20 == 0:
                print(f"  PSO iter {it+1:4d}/{max_iter} | cost = {g_best_cost:.6e}")

        return {
            "x": g_best_pos,
            "fun": g_best_cost,
            "nfev": nfev,
            "nit": max_iter,
            "history": history,
            "success": True,
        }


# ---------------------------------------------------------------------------
# Levenberg-Marquardt local refinement
# ---------------------------------------------------------------------------

class LevenbergMarquardt:
    """Levenberg-Marquardt local optimizer for fine-tuning extracted parameters.

    Uses scipy.optimize.least_squares with the 'lm' method. Since the cost
    function is a scalar RMSE, we construct a residual vector by flattening
    the Id-Vg and Id-Vd error surfaces.
    """

    def __init__(self, objective: ExtractionObjective, model_cls=MOSFETLevel3):
        self.objective = objective
        self.model_cls = model_cls

    def refine(
        self,
        x0,
        bounds=None,
        max_nfev=500,
        verbose=True,
    ):
        """Refine parameters starting from x0 using Nelder-Mead (no gradient needed).

        Works with both full and reduced parameter vectors.

        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector (e.g., from DE/PSO output).
        bounds : list of tuples, optional
        max_nfev : int
        verbose : bool

        Returns
        -------
        result : dict
        """
        if bounds is None:
            bounds, _ = get_default_bounds(self.model_cls)

        def cost(x):
            # Call objective directly — works with both full and wrapped objectives
            try:
                return self.objective(x, self.model_cls)
            except TypeError:
                return self.objective(x)

        result = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": max_nfev, "ftol": 1e-8},
        )

        return {
            "x": result.x,
            "fun": result.fun,
            "nfev": result.nfev,
            "success": result.success,
            "message": result.message,
        }


# ---------------------------------------------------------------------------
# Two-stage extraction
# ---------------------------------------------------------------------------

def two_stage_extraction(
    objective: ExtractionObjective,
    model_cls=MOSFETLevel3,
    stage1="de",
    stage1_options=None,
    fixed_params=None,
    verbose=True,
):
    """Two-stage parameter extraction: global search + local refinement.

    Parameters
    ----------
    objective : ExtractionObjective
    model_cls : type
        MOSFETLevel1 or MOSFETLevel3.
    stage1 : str
        "de" or "pso".
    stage1_options : dict, optional
        Keyword arguments for the stage-1 optimizer.
    fixed_params : dict, optional
        Parameters to hold fixed during optimization, e.g. {"W": 10e-6, "L": 0.18e-6}.
    verbose : bool

    Returns
    -------
    result : dict
        Keys: x_global, x_refined, cost_global, cost_refined, nfev_total, stage1_info,
        param_names (full parameter names), x_refined_full (full parameter vector).
    """
    fixed_params = fixed_params or {}
    bounds, full_param_names = get_default_bounds(model_cls)

    if fixed_params:
        from src.extraction.objective import build_reduced_objective, _get_param_order
        wrapped_obj, bounds, param_names = build_reduced_objective(
            objective, model_cls, fixed_params
        )
        full_order = _get_param_order(model_cls)
        free_indices = [i for i, name in enumerate(full_order) if name not in fixed_params]
    else:
        wrapped_obj = objective
        param_names = full_param_names

    # Stage 1: Global search
    if verbose:
        print(f"Stage 1: Global search ({stage1.upper()}) on {len(param_names)} parameters")
        print(f"  Parameters: {param_names}")

    if stage1 == "de":
        de_opts = DEOptions(**(stage1_options or {}))
        if stage1_options and "max_iter" in stage1_options:
            de_opts.max_iter = stage1_options["max_iter"]
        optimizer = DifferentialEvolution(wrapped_obj, model_cls)
        r1 = optimizer.optimize(bounds, de_opts, verbose=verbose)
    elif stage1 == "pso":
        opts = stage1_options or {}
        optimizer = ParticleSwarm(wrapped_obj, model_cls)
        r1 = optimizer.optimize(bounds, verbose=verbose, **opts)
    else:
        raise ValueError(f"Unknown stage1 optimizer: {stage1}")

    if verbose:
        print(f"  Stage 1 complete: cost = {r1['fun']:.6e}")
        print()

    # Stage 2: Local refinement
    if verbose:
        print("Stage 2: LM local refinement")

    lm = LevenbergMarquardt(wrapped_obj, model_cls)
    r2 = lm.refine(r1["x"], bounds, verbose=verbose)

    # Reconstruct full parameter vector if needed
    x_refined_full = r2["x"].copy()
    if fixed_params:
        from src.extraction.objective import _get_param_order
        full_order = _get_param_order(model_cls)
        x_full = np.zeros(len(full_order))
        for name, val in fixed_params.items():
            if name in full_order:
                idx = full_order.index(name)
                x_full[idx] = val
        for j, idx in enumerate(free_indices):
            x_full[idx] = r2["x"][j]
        x_refined_full = x_full
        param_names = full_param_names

    if verbose:
        print(f"  Stage 2 complete: cost = {r2['fun']:.6e}")
        print(f"  Total NFev: {r1['nfev'] + r2['nfev']}")
        print()
        _print_extracted_params(x_refined_full, param_names)

    return {
        "x_global": r1["x"],
        "x_refined": x_refined_full,
        "cost_global": r1["fun"],
        "cost_refined": r2["fun"],
        "nfev_total": r1["nfev"] + r2["nfev"],
        "stage1_info": r1,
        "stage2_info": r2,
        "param_names": param_names,
    }


def _print_extracted_params(x, param_names):
    """Pretty-print extracted parameters."""
    print("  Extracted parameters:")
    for name, val in zip(param_names, x):
        if abs(val) < 1e-6:
            print(f"    {name:12s} = {val:.4e}")
        elif abs(val) < 1:
            print(f"    {name:12s} = {val:.6f}")
        elif abs(val) < 1000:
            print(f"    {name:12s} = {val:.4f}")
        else:
            print(f"    {name:12s} = {val:.4e}")
