"""Benchmark test cases for SPICE parameter extraction.

Each case defines a target MOSFET with known parameters, adds noise,
and measures extraction accuracy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from src.device.mosfet import (
    MOSFETLevel1, MOSFETLevel3,
    MOSFETParamsLevel1, MOSFETParamsLevel3,
)
from src.device.curves import generate_iv_curves
from src.extraction.objective import (
    ExtractionObjective, CurveData, params_to_vector, vector_to_params,
)
from src.extraction.optimizer import two_stage_extraction, get_default_bounds


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    name: str
    description: str
    model_cls: type   # MOSFETLevel1 or MOSFETLevel3
    target_params: object
    noise_std: float = 0.03  # Relative noise (3%)
    extractable_params: Optional[List[str]] = None  # Params expected to be recoverable

    def run(self, verbose=True):
        """Run this benchmark case and return results."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Benchmark: {self.name}")
            print(f"  {self.description}")
            print(f"{'='*60}")

        # Generate target I-V curves
        model = self.model_cls(self.target_params)
        id_vg, id_vd = generate_iv_curves(model)

        # Add noise
        if self.noise_std > 0:
            noise_vg = np.random.RandomState(123).normal(
                0, self.noise_std * np.abs(id_vg.ids), id_vg.ids.shape
            )
            id_vg.ids = id_vg.ids + noise_vg * (id_vg.ids > 1e-12)
            if id_vd is not None:
                noise_vd = np.random.RandomState(456).normal(
                    0, self.noise_std * np.abs(id_vd.ids), id_vd.ids.shape
                )
                id_vd.ids = id_vd.ids + noise_vd * (id_vd.ids > 1e-12)

        # Build objective
        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        tg_vd = CurveData(vgs=id_vd.vgs_values, vds=id_vd.vds, ids=id_vd.ids)

        objective = ExtractionObjective(target_idvg=tg_vg, target_idvd=tg_vd)

        # Run extraction with W, L fixed (layout parameters are known)
        t0 = time.perf_counter()
        fixed = {"W": self.target_params.W, "L": self.target_params.L}
        if hasattr(self.target_params, "TOX"):
            fixed["TOX"] = self.target_params.TOX
        if hasattr(self.target_params, "NSUB"):
            fixed["NSUB"] = self.target_params.NSUB
        result = two_stage_extraction(
            objective, model_cls=self.model_cls,
            stage1="de",
            stage1_options={"max_iter": 200},
            fixed_params=fixed,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0

        # Compute per-parameter errors
        param_errors = _compute_errors(
            self.target_params, result["x_refined"], result["param_names"]
        )

        # Filter to extractable parameters if specified
        if self.extractable_params:
            eval_errors = {k: v for k, v in param_errors.items()
                          if k in self.extractable_params}
        else:
            eval_errors = param_errors

        errors_pct = list(eval_errors.values())
        n_good = sum(1 for e in errors_pct if e < 5.0)

        summary = {
            "name": self.name,
            "cost": result["cost_refined"],
            "elapsed_s": elapsed,
            "nfev": result["nfev_total"],
            "param_errors": param_errors,
            "n_params": len(errors_pct),
            "n_good_5pct": n_good,
            "good_rate": n_good / max(len(errors_pct), 1),
            "mean_error_pct": np.mean(errors_pct) if errors_pct else 0.0,
        }

        if verbose:
            print(f"\n  Results: {n_good}/{len(errors_pct)} params within 5%")
            print(f"  Mean error: {summary['mean_error_pct']:.2f}%")
            print(f"  Time: {elapsed:.1f}s, NFev: {result['nfev_total']}")

        return summary


def _compute_errors(true_params, extracted_vector, param_names):
    """Compute per-parameter relative errors in %."""
    errors = {}
    for i, name in enumerate(param_names):
        if hasattr(true_params, name):
            true_val = getattr(true_params, name)
            ext_val = extracted_vector[i]
            if abs(true_val) > 1e-15:
                errors[name] = abs(true_val - ext_val) / abs(true_val) * 100.0
            else:
                errors[name] = abs(true_val - ext_val) * 100.0
    return errors


# ---------------------------------------------------------------------------
# Predefined benchmark cases
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    BenchmarkCase(
        name="L1_Nominal",
        description="Level-1 NMOS at Vsb=0 (GAMMA/PHI require multi-Vsb data)",
        model_cls=MOSFETLevel1,
        target_params=MOSFETParamsLevel1(
            W=10e-6, L=0.18e-6, VTH0=0.45, GAMMA=0.4,
            PHI=0.65, KP=200e-6, LAMBDA=0.05,
        ),
        noise_std=0.03,
        # Only VTH0, KP, LAMBDA are well-constrained at Vsb=0
        extractable_params=["VTH0", "KP", "LAMBDA"],
    ),
    BenchmarkCase(
        name="L3_Nominal",
        description="Level-3 NMOS, nominal short-channel params (VTH0, U0, THETA are primary targets)",
        model_cls=MOSFETLevel3,
        target_params=MOSFETParamsLevel3(
            W=10e-6, L=0.18e-6, VTH0=0.45, GAMMA=0.4,
            PHI=0.65, U0=400, THETA=0.05, VSAT=1e7,
            KAPPA=0.5, LAMBDA=0.05, ETA0=0.05, N0=1.5,
            RD=20, RS=20,
        ),
        noise_std=0.03,
        extractable_params=["VTH0", "U0", "THETA", "LAMBDA"],
    ),
    BenchmarkCase(
        name="L3_HighVTH",
        description="Level-3 NMOS, high-threshold device (VTH0=0.7V)",
        model_cls=MOSFETLevel3,
        target_params=MOSFETParamsLevel3(
            W=10e-6, L=0.18e-6, VTH0=0.7, GAMMA=0.6,
            PHI=0.75, U0=300, THETA=0.08, VSAT=8e6,
            KAPPA=0.5, LAMBDA=0.03, ETA0=0.03, N0=1.8,
            RD=30, RS=30,
        ),
        noise_std=0.03,
        extractable_params=["VTH0", "U0", "THETA", "LAMBDA"],
    ),
    BenchmarkCase(
        name="L3_FastCorner",
        description="Level-3 NMOS, fast corner (low VTH, high mobility)",
        model_cls=MOSFETLevel3,
        target_params=MOSFETParamsLevel3(
            W=10e-6, L=0.18e-6, VTH0=0.3, GAMMA=0.3,
            PHI=0.55, U0=550, THETA=0.03, VSAT=12e6,
            KAPPA=0.4, LAMBDA=0.08, ETA0=0.08, N0=1.3,
            RD=10, RS=10,
        ),
        noise_std=0.03,
        extractable_params=["VTH0", "U0", "THETA", "LAMBDA"],
    ),
]


def run_benchmark(case: BenchmarkCase, verbose=True):
    """Run a single benchmark case."""
    return case.run(verbose=verbose)


def run_all_benchmarks(verbose=True):
    """Run all predefined benchmark cases and print summary."""
    results = []
    for case in BENCHMARK_CASES:
        result = run_benchmark(case, verbose=verbose)
        results.append(result)

    if verbose:
        print(f"\n{'='*60}")
        print("  BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Case':<20s} {'Cost':>10s} {'Good%':>8s} {'MeanErr':>8s} {'Time':>8s}")
        print(f"  {'-'*54}")
        for r in results:
            print(f"  {r['name']:<20s} {r['cost']:>10.2e} "
                  f"{r['good_rate']:>7.1%} {r['mean_error_pct']:>7.2f}% "
                  f"{r['elapsed_s']:>7.1f}s")

        # Overall stats
        avg_good = np.mean([r["good_rate"] for r in results])
        avg_err = np.mean([r["mean_error_pct"] for r in results])
        print(f"  {'-'*54}")
        print(f"  {'AVERAGE':<20s} {'':>10s} {avg_good:>7.1%} {avg_err:>7.2f}%")
        print()

    return results


if __name__ == "__main__":
    run_all_benchmarks()
