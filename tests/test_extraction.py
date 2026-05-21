"""Tests for parameter extraction: objective, optimizers, sensitivity."""

import numpy as np
import pytest
from src.device.mosfet import MOSFETLevel3, MOSFETParamsLevel3
from src.device.curves import generate_iv_curves
from src.extraction.objective import (
    ExtractionObjective, CurveData, params_to_vector, vector_to_params,
    PARAM_ORDER_L1, PARAM_ORDER_L3,
)
from src.extraction.optimizer import (
    DifferentialEvolution, ParticleSwarm, LevenbergMarquardt,
    two_stage_extraction, get_default_bounds, DEOptions,
)
from src.extraction.sensitivity import (
    SobolAnalyzer, MorrisAnalyzer, ParameterBounds,
)


class TestObjective:
    def test_objective_at_true_params_zero(self):
        """Objective should be near zero when evaluated at true params."""
        params = MOSFETParamsLevel3()
        model = MOSFETLevel3(params)
        id_vg, id_vd = generate_iv_curves(model)

        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        tg_vd = CurveData(vgs=id_vd.vgs_values, vds=id_vd.vds, ids=id_vd.ids)

        obj = ExtractionObjective(target_idvg=tg_vg, target_idvd=tg_vd)
        x_true = params_to_vector(params)

        error = obj(x_true, MOSFETLevel3)
        assert error < 0.01  # Should be near zero (only numerical noise)

    def test_objective_increases_with_wrong_params(self):
        """Error should increase when parameters are wrong."""
        params = MOSFETParamsLevel3(VTH0=0.45)
        model = MOSFETLevel3(params)
        id_vg, id_vd = generate_iv_curves(model)

        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        tg_vd = CurveData(vgs=id_vd.vgs_values, vds=id_vd.vds, ids=id_vd.ids)

        obj = ExtractionObjective(target_idvg=tg_vg, target_idvd=tg_vd)

        # Correct params
        x_true = params_to_vector(params)
        err_true = obj(x_true, MOSFETLevel3)

        # Wrong VTH0
        x_wrong = x_true.copy()
        x_wrong[2] = 0.9  # VTH0=0.9 instead of 0.45
        err_wrong = obj(x_wrong, MOSFETLevel3)

        assert err_wrong > err_true

    def test_physics_penalty_active(self):
        """Negative VTH0 should be penalized."""
        params = MOSFETParamsLevel3()
        model = MOSFETLevel3(params)
        id_vg, id_vd = generate_iv_curves(model)

        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        obj = ExtractionObjective(target_idvg=tg_vg)

        x_bad = np.array([10e-6, 0.18e-6, -0.5, 0.4, 0.65, 400, 0.05,
                          1e7, 0.5, 0.05, 0.05, 1.5, 20, 20])
        err = obj(x_bad, MOSFETLevel3)
        # With negative VTH0, penalty term should be non-zero
        assert err > 0.1

    def test_vector_params_roundtrip_l1(self):
        from src.device.mosfet import MOSFETParamsLevel1, MOSFETLevel1
        p1 = MOSFETParamsLevel1()
        x1 = params_to_vector(p1)
        p1_back = vector_to_params(x1, MOSFETLevel1)
        assert p1_back.VTH0 == pytest.approx(p1.VTH0)
        assert p1_back.KP == pytest.approx(p1.KP)

    def test_vector_params_roundtrip_l3(self):
        p3 = MOSFETParamsLevel3()
        x3 = params_to_vector(p3)
        p3_back = vector_to_params(x3, MOSFETLevel3)
        assert p3_back.VTH0 == pytest.approx(p3.VTH0)
        assert p3_back.U0 == pytest.approx(p3.U0)
        assert p3_back.VSAT == pytest.approx(p3.VSAT)


class TestOptimizers:
    @pytest.fixture
    def objective_l1(self):
        """Simple Level-1 optimization problem."""
        params = MOSFETParamsLevel3()
        from src.device.mosfet import MOSFETLevel1, MOSFETParamsLevel1
        p1 = MOSFETParamsLevel1()
        m1 = MOSFETLevel1(p1)
        id_vg, id_vd = generate_iv_curves(m1)

        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        return ExtractionObjective(target_idvg=tg_vg)

    def test_de_optimizer_runs(self, objective_l1):
        bounds, _ = get_default_bounds(MOSFETLevel3)
        from src.device.mosfet import MOSFETLevel1
        opt = DifferentialEvolution(objective_l1, MOSFETLevel1)
        result = opt.optimize(bounds[:7], DEOptions(max_iter=20, pop_size=10), verbose=False)
        assert "x" in result
        assert result["fun"] < 10.0
        assert len(result["x"]) == 7

    def test_pso_optimizer_runs(self, objective_l1):
        bounds, _ = get_default_bounds(MOSFETLevel3)
        from src.device.mosfet import MOSFETLevel1
        opt = ParticleSwarm(objective_l1, MOSFETLevel1)
        result = opt.optimize(bounds[:7], n_particles=10, max_iter=20, verbose=False)
        assert "x" in result
        assert result["fun"] < 10.0
        assert "history" in result

    def test_two_stage_extraction_runs(self):
        """Full two-stage extraction should complete without error."""
        from src.device.mosfet import MOSFETLevel1, MOSFETParamsLevel1
        p1 = MOSFETParamsLevel1()
        m1 = MOSFETLevel1(p1)
        id_vg, id_vd = generate_iv_curves(m1)

        tg_vg = CurveData(vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids)
        tg_vd = CurveData(vgs=id_vd.vgs_values, vds=id_vd.vds, ids=id_vd.ids)

        obj = ExtractionObjective(target_idvg=tg_vg, target_idvd=tg_vd)
        result = two_stage_extraction(
            obj, model_cls=MOSFETLevel1,
            stage1="de",
            stage1_options={"max_iter": 30, "pop_size": 10},
            verbose=False,
        )
        assert result["cost_refined"] < 5.0
        assert result["nfev_total"] > 0


class TestSensitivity:
    def test_parameter_bounds_sample(self):
        bounds = ParameterBounds(
            names=["VTH0", "U0"],
            bounds=[(0.1, 1.0), (100, 600)],
        )
        samples = bounds.sample(100, seed=42)
        assert samples.shape == (100, 2)
        assert np.all(samples[:, 0] >= 0.1) and np.all(samples[:, 0] <= 1.0)
        assert np.all(samples[:, 1] >= 100) and np.all(samples[:, 1] <= 600)

    def test_sobol_fallback_runs(self):
        def dummy_cost(x):
            return float(x[0]**2 + 0.5 * x[1]**2 + x[0] * 0.3)

        bounds = ParameterBounds(
            names=["a", "b"],
            bounds=[(0.5, 2.0), (0.5, 2.0)],  # Avoid symmetric center
        )
        analyzer = SobolAnalyzer(dummy_cost, bounds)
        result = analyzer.analyze(n_base=64, verbose=False)
        assert "S1" in result
        assert "ST" in result
        # First param has stronger influence (coefficient 1.0 + linear term)
        assert result["S1"][0] > result["S1"][1] or result["ST"][0] > result["ST"][1]

    def test_morris_fallback_runs(self):
        def dummy_cost(x):
            return float(np.sum(x**2))

        bounds = ParameterBounds(
            names=["a", "b", "c"],
            bounds=[(-1, 1), (-1, 1), (-1, 1)],
        )
        analyzer = MorrisAnalyzer(dummy_cost, bounds)
        result = analyzer.analyze(n_trajectories=5, verbose=False)
        assert "mu_star" in result
        assert len(result["mu_star"]) == 3
