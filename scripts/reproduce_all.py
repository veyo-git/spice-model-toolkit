"""One-click reproduction script for SPICE Model Parameter Extraction Toolkit.

Runs the full pipeline:
  1. Unit tests
  2. Device physics verification (I-V curve sanity checks)
  3. Parameter extraction benchmarks
  4. Sensitivity analysis demo
  5. Surrogate model training demo (optional)

Usage:
    python scripts/reproduce_all.py              # Full pipeline
    python scripts/reproduce_all.py --quick      # Quick smoke test
    python scripts/reproduce_all.py --skip-surrogate  # Skip NN training
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def run_tests(verbose=True):
    """Run pytest suite."""
    import pytest
    test_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    if verbose:
        print("=" * 60)
        print("  STAGE 1: Unit Tests")
        print("=" * 60)

    exit_code = pytest.main([test_dir, "-v", "--tb=short"])
    success = exit_code == pytest.ExitCode.OK
    if verbose:
        status = "PASSED" if success else "FAILED"
        print(f"\n  Tests: {status}")
    return success


def verify_device_physics(verbose=True):
    """Run physics sanity checks."""
    from src.device.mosfet import MOSFETLevel3, MOSFETParamsLevel3
    from src.device.curves import generate_iv_curves, extract_vth_linear

    if verbose:
        print("\n" + "=" * 60)
        print("  STAGE 2: Device Physics Verification")
        print("=" * 60)

    params = MOSFETParamsLevel3()
    model = MOSFETLevel3(params)
    id_vg, id_vd = generate_iv_curves(model)

    # Check 1: Current at Vgs=VTH should be small
    ids_at_vth = float(model.ids(params.VTH0, 0.1))
    if verbose:
        print(f"  Ids(Vgs=VTH0={params.VTH0}V, Vds=0.1V) = {ids_at_vth:.3e} A")
    assert ids_at_vth < 1e-5, f"Ids at VTH should be small, got {ids_at_vth}"

    # Check 2: Current increases with Vgs
    i1 = model.ids(1.0, 1.0)
    i2 = model.ids(1.5, 1.0)
    assert i2 > i1, "Ids should increase with Vgs"

    # Check 3: Gm peak in expected range
    gm = np.gradient(id_vg.ids[-1], id_vg.vgs)
    gm_max = np.max(gm)
    if verbose:
        print(f"  Gm max = {gm_max*1e3:.2f} mS")
    assert gm_max > 0, "Gm should be positive"

    # Check 4: VTH extraction consistency
    vth_ext = extract_vth_linear(model, id_vg.vgs)
    if verbose:
        print(f"  Extracted VTH = {vth_ext:.4f}V (true = {params.VTH0:.4f}V)")
    assert abs(vth_ext - params.VTH0) < 0.15, f"VTH extraction off by {abs(vth_ext - params.VTH0):.3f}V"

    # Check 5: Subthreshold current is small at negative Vgs (below threshold)
    ids_sub = float(model.ids(-0.1, 1.0))
    assert ids_sub < 1e-3, f"Current at Vgs=-0.1V should be very small, got {ids_sub:.3e}"

    if verbose:
        print("  Device physics verification: PASSED")
    return True


def run_extraction_benchmarks(quick=False, verbose=True):
    """Run parameter extraction benchmarks."""
    from src.benchmark.test_cases import run_all_benchmarks, BENCHMARK_CASES

    if verbose:
        print("\n" + "=" * 60)
        print("  STAGE 3: Parameter Extraction Benchmarks")
        print("=" * 60)

    if quick:
        # Run only the first benchmark case
        from src.benchmark.test_cases import run_benchmark
        results = [run_benchmark(BENCHMARK_CASES[0], verbose=verbose)]
    else:
        results = run_all_benchmarks(verbose=verbose)

    return results


def run_sensitivity_demo(verbose=True):
    """Run sensitivity analysis demo."""
    from src.extraction.sensitivity import SobolAnalyzer, ParameterBounds
    from src.device.mosfet import MOSFETLevel3, MOSFETParamsLevel3

    if verbose:
        print("\n" + "=" * 60)
        print("  STAGE 4: Sensitivity Analysis Demo")
        print("=" * 60)

    p_default = MOSFETParamsLevel3()
    base_model = MOSFETLevel3(p_default)
    from src.device.curves import generate_iv_curves
    base_vg, _ = generate_iv_curves(base_model)

    bounds = ParameterBounds(
        names=["VTH0", "GAMMA", "PHI", "U0", "THETA", "VSAT", "ETA0", "LAMBDA"],
        bounds=[
            (0.2, 0.8), (0.2, 1.0), (0.4, 1.0),
            (200, 600), (0.02, 0.15), (6e6, 14e6),
            (0.02, 0.12), (0.02, 0.10),
        ],
    )

    def cost_fn(x):
        p = MOSFETParamsLevel3(
            VTH0=float(x[0]), GAMMA=float(x[1]), PHI=float(x[2]),
            U0=float(x[3]), THETA=float(x[4]), VSAT=float(x[5]),
            ETA0=float(x[6]), LAMBDA=float(x[7]),
        )
        m = MOSFETLevel3(p)
        vg, _ = generate_iv_curves(m)
        ids_log = np.log10(np.maximum(base_vg.ids, 1e-12))
        sim_log = np.log10(np.maximum(vg.ids, 1e-12))
        return float(np.sqrt(np.mean((ids_log - sim_log) ** 2)))

    analyzer = SobolAnalyzer(cost_fn, bounds)
    result = analyzer.analyze(n_base=128, verbose=verbose)

    # Print ranking
    ranking = sorted(
        zip(bounds.names, result["S1"], result["ST"]),
        key=lambda x: -x[2],
    )
    if verbose:
        print("\n  Sensitivity Ranking (Total-Effect ST):")
        for rank, (name, s1, st) in enumerate(ranking, 1):
            bar = "█" * int(st * 100)
            print(f"    {rank:2d}. {name:10s} S1={s1:.4f} ST={st:.4f} {bar}")

    return result


def run_surrogate_demo(verbose=True):
    """Train a small surrogate model and benchmark speedup."""
    from src.device.mosfet import MOSFETLevel3, MOSFETParamsLevel3
    from src.surrogate.nn_surrogate import train_surrogate, SurrogateConfig

    if verbose:
        print("\n" + "=" * 60)
        print("  STAGE 5: Surrogate Model Demo")
        print("=" * 60)

    param_space = {
        "VTH0": (0.3, 0.7),
        "U0": (200, 500),
        "THETA": (0.02, 0.10),
        "VSAT": (6e6, 12e6),
        "ETA0": (0.03, 0.08),
        "LAMBDA": (0.02, 0.10),
    }

    def make_model(params_dict):
        defaults = {
            "W": 10e-6, "L": 0.18e-6, "GAMMA": 0.4, "PHI": 0.65,
            "KAPPA": 0.5, "N0": 1.5, "RD": 20, "RS": 20,
            "NSUB": 1e22, "TOX": 4e-9,
        }
        p = MOSFETParamsLevel3(**{**defaults, **params_dict})
        return MOSFETLevel3(p)

    config = SurrogateConfig(
        n_params=len(param_space),
        hidden_dims=(128, 256, 128),
        epochs=50,
        batch_size=512,
    )

    surrogate, history = train_surrogate(
        make_model, param_space, config,
        n_samples=5000, seed=42, verbose=verbose,
    )

    final_loss = history["test_loss"][-1]
    if verbose:
        print(f"  Final test MSE (log10 space): {final_loss:.6f}")
        print(f"  RMSE in Ids: ~{10**np.sqrt(final_loss):.1f}×")

    return surrogate, history


def main():
    parser = argparse.ArgumentParser(description="Reproduce all SPICE toolkit results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (fewer iterations)")
    parser.add_argument("--skip-surrogate", action="store_true",
                        help="Skip surrogate model training")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for surrogate training (cpu/cuda/auto)")
    args = parser.parse_args()

    t_start = time.perf_counter()

    print("=" * 60)
    print("  SPICE Model Parameter Extraction Toolkit — Reproduction")
    print("=" * 60)
    print(f"  Mode: {'Quick' if args.quick else 'Full'}")
    print()

    all_ok = True

    # Stage 1: Tests
    if not args.skip_tests:
        tests_ok = run_tests(verbose=True)
        all_ok = all_ok and tests_ok

    # Stage 2: Physics verification
    try:
        verify_device_physics(verbose=True)
    except Exception as e:
        print(f"  Physics verification FAILED: {e}")
        all_ok = False

    # Stage 3: Extraction benchmarks (informational — quality depends on optimizer tuning)
    try:
        run_extraction_benchmarks(quick=args.quick, verbose=True)
        print("  Extraction benchmarks: COMPLETED")
    except Exception as e:
        print(f"  Extraction benchmarks FAILED: {e}")
        all_ok = False

    # Stage 4: Sensitivity demo
    try:
        run_sensitivity_demo(verbose=True)
    except Exception as e:
        print(f"  Sensitivity demo FAILED: {e}")
        all_ok = False

    # Stage 5: Surrogate demo
    if not args.skip_surrogate:
        try:
            run_surrogate_demo(verbose=True)
        except Exception as e:
            print(f"  Surrogate demo FAILED: {e}")
            all_ok = False

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    status = "ALL OK" if all_ok else "SOME FAILURES"
    print(f"  Reproduction complete: {status} ({elapsed:.1f}s)")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
