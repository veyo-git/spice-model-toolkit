"""Parameter sensitivity analysis for SPICE model extraction.

Methods:
  - Sobol': Global variance-based sensitivity indices (first-order + total-effect)
  - Morris: Screening method for factor ranking (elementary effects)

These help identify which parameters most influence the I-V curves,
guiding extraction strategy (fix insensitive params, focus on sensitive ones).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
import warnings


@dataclass
class ParameterBounds:
    """Parameter bounds for sensitivity analysis."""

    names: List[str]
    bounds: List[Tuple[float, float]]

    @property
    def n_params(self):
        return len(self.names)

    def sample(self, n, seed=None):
        """Uniform random sampling within bounds."""
        rng = np.random.RandomState(seed)
        samples = np.zeros((n, self.n_params))
        for i, (lo, hi) in enumerate(self.bounds):
            samples[:, i] = rng.uniform(lo, hi, n)
        return samples


# ---------------------------------------------------------------------------
# Sobol' sensitivity analysis
# ---------------------------------------------------------------------------

class SobolAnalyzer:
    """Sobol' global sensitivity analysis.

    Decomposes output variance into contributions from each parameter
    and their interactions. Requires SALib package.

    References
    ----------
    Saltelli, A. et al. (2010). "Variance based sensitivity analysis of
    model output." Computer Physics Communications.
    """

    def __init__(self, objective_fn: Callable, param_bounds: ParameterBounds):
        """
        Parameters
        ----------
        objective_fn : callable
            Function f(X) -> float where X is a 1D array of parameters.
        param_bounds : ParameterBounds
        """
        self.objective_fn = objective_fn
        self.param_bounds = param_bounds

    def analyze(self, n_base=512, seed=42, verbose=True):
        """Run Sobol' analysis.

        Parameters
        ----------
        n_base : int
            Base sample size. Total evaluations = N*(2D+2) for Saltelli method.
        seed : int
        verbose : bool

        Returns
        -------
        result : dict
            Keys: S1 (first-order), ST (total-effect), param_names, conf
        """
        try:
            from SALib.sample import sobol as sobol_sample
            from SALib.analyze import sobol as sobol_analyze
        except ImportError:
            warnings.warn("SALib not installed. Install with: pip install SALib")
            return self._fallback_analysis(n_base, seed)

        D = self.param_bounds.n_params
        problem = {
            "num_vars": D,
            "names": self.param_bounds.names,
            "bounds": self.param_bounds.bounds,
        }

        if verbose:
            print(f"Generating Sobol' samples (N={n_base}, D={D})...")
        param_values = sobol_sample.sample(problem, n_base, seed=seed)

        if verbose:
            print(f"Evaluating {len(param_values)} samples...")
        Y = np.array([self.objective_fn(pv) for pv in param_values])

        if verbose:
            print("Computing Sobol' indices...")
        Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

        return {
            "S1": Si["S1"],
            "ST": Si["ST"],
            "S1_conf": Si["S1_conf"],
            "ST_conf": Si["ST_conf"],
            "param_names": self.param_bounds.names,
            "n_samples": len(param_values),
        }

    def _fallback_analysis(self, n_base, seed):
        """Simple one-at-a-time sensitivity as fallback when SALib unavailable."""
        D = self.param_bounds.n_params
        rng = np.random.RandomState(seed)

        # Baseline at center of bounds
        x0 = np.array([(lo + hi) / 2 for lo, hi in self.param_bounds.bounds])
        y0 = self.objective_fn(x0)

        S1 = np.zeros(D)
        delta = 0.05  # ±5% perturbation

        for i in range(D):
            lo, hi = self.param_bounds.bounds[i]
            pert = delta * (hi - lo)
            x_hi = x0.copy()
            x_hi[i] = min(x0[i] + pert, hi)
            x_lo = x0.copy()
            x_lo[i] = max(x0[i] - pert, lo)
            y_hi = self.objective_fn(x_hi)
            y_lo = self.objective_fn(x_lo)
            S1[i] = abs(y_hi - y_lo) / (2 * pert / (hi - lo))

        # Normalize
        total = np.sum(S1)
        if total > 1e-12:
            S1 = S1 / total

        return {
            "S1": S1,
            "ST": S1,  # Approximate
            "S1_conf": np.zeros(D),
            "ST_conf": np.zeros(D),
            "param_names": self.param_bounds.names,
            "n_samples": 2 * D + 1,
            "note": "One-at-a-time OAT fallback (SALib not available)",
        }


# ---------------------------------------------------------------------------
# Morris screening
# ---------------------------------------------------------------------------

class MorrisAnalyzer:
    """Morris method for parameter screening.

    Computes μ* (mean absolute elementary effect) and σ (standard deviation)
    to classify parameters as:
      - Negligible (low μ*)
      - Linear/additive (high μ*, low σ)
      - Nonlinear/interactive (high μ*, high σ)

    References
    ----------
    Morris, M. D. (1991). "Factorial sampling plans for preliminary
    computational experiments." Technometrics.
    """

    def __init__(self, objective_fn: Callable, param_bounds: ParameterBounds):
        self.objective_fn = objective_fn
        self.param_bounds = param_bounds

    def analyze(self, n_trajectories=10, n_levels=4, seed=42, verbose=True):
        """Run Morris analysis.

        Parameters
        ----------
        n_trajectories : int
            Number of Morris trajectories (typically 10–50).
        n_levels : int
            Number of grid levels (typically 4–8).
        seed : int
        verbose : bool

        Returns
        -------
        result : dict
            Keys: mu_star, sigma, mu, param_names
        """
        try:
            from SALib.sample import morris as morris_sample
            from SALib.analyze import morris as morris_analyze
        except ImportError:
            warnings.warn("SALib not installed. Using simple OAT screening.")
            return self._fallback_analysis(n_trajectories, seed)

        D = self.param_bounds.n_params
        problem = {
            "num_vars": D,
            "names": self.param_bounds.names,
            "bounds": self.param_bounds.bounds,
        }

        if verbose:
            print(f"Generating Morris trajectories (N={n_trajectories}, p={n_levels})...")
        param_values = morris_sample.sample(
            problem, n_trajectories, n_levels, seed=seed
        )

        if verbose:
            print(f"Evaluating {len(param_values)} samples...")
        Y = np.array([self.objective_fn(pv) for pv in param_values])

        if verbose:
            print("Computing Morris indices...")
        Si = morris_analyze.analyze(problem, param_values, Y, print_to_console=False)

        return {
            "mu_star": Si["mu_star"],
            "mu": Si["mu"],
            "sigma": Si["sigma"],
            "param_names": self.param_bounds.names,
            "n_samples": len(param_values),
        }

    def _fallback_analysis(self, n_trajectories, seed):
        """OAT fallback."""
        D = self.param_bounds.n_params
        rng = np.random.RandomState(seed)
        x0 = np.array([(lo + hi) / 2 for lo, hi in self.param_bounds.bounds])
        y0 = self.objective_fn(x0)

        mu_star = np.zeros(D)
        sigma = np.zeros(D)

        for i in range(D):
            effects = []
            for _ in range(n_trajectories):
                lo, hi = self.param_bounds.bounds[i]
                # Random step within bounds
                x1 = x0.copy()
                x1[i] = rng.uniform(lo, hi)
                y1 = self.objective_fn(x1)
                dy = y1 - y0
                dx = max(abs(x1[i] - x0[i]), 1e-12)
                effects.append(abs(dy / dx))
            mu_star[i] = np.mean(effects)
            sigma[i] = np.std(effects)

        return {
            "mu_star": mu_star,
            "mu": mu_star,
            "sigma": sigma,
            "param_names": self.param_bounds.names,
            "n_samples": (D + 1) + n_trajectories * D,
            "note": "OAT fallback (SALib not available)",
        }
