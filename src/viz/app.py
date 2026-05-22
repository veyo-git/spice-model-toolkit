"""Gradio interactive dashboard for SPICE model parameter extraction.

Three tabs:
  1. Device Simulator — Adjust MOSFET parameters, see I-V curves in real time
  2. Parameter Extraction — Run optimization and compare extracted vs target
  3. Sensitivity Analysis — View Sobol'/Morris results
"""

import numpy as np
import gradio as gr
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.device.mosfet import MOSFETLevel3, MOSFETParamsLevel3
from src.device.curves import generate_iv_curves
from src.extraction.objective import ExtractionObjective, CurveData
from src.extraction.optimizer import two_stage_extraction
from src.extraction.sensitivity import SobolAnalyzer, ParameterBounds
from src.viz.plots import (
    plot_iv_curves, plot_gm, plot_optimization_trace,
    plot_sensitivity_heatmap, plot_extraction_comparison,
)


# ---------------------------------------------------------------------------
# Tab 1: Device Simulator
# ---------------------------------------------------------------------------

def simulate_device(W, L, VTH0, GAMMA, PHI, U0, THETA, VSAT, ETA0, LAMBDA, N0, RD, RS,
                    vgs_min, vgs_max, vds_min, vds_max):
    """Generate I-V curves from user-specified parameters."""
    try:
        params = MOSFETParamsLevel3(
            W=W * 1e-3, L=L * 1e-9, VTH0=VTH0, GAMMA=GAMMA, PHI=PHI,
            U0=U0, THETA=THETA, VSAT=VSAT * 1e6, ETA0=ETA0,
            LAMBDA=LAMBDA, N0=N0, RD=RD, RS=RS,
        )
        model = MOSFETLevel3(params)
        id_vg, id_vd = generate_iv_curves(
            model,
            vgs_range=(vgs_min, vgs_max, 101),
            vds_range=(vds_min, vds_max, 101),
        )
        fig = plot_iv_curves(id_vg, id_vd, "MOSFET I-V (Parameter Sweep)")
        return fig
    except Exception as e:
        return None


def build_simulator_tab():
    """Build the device simulator tab UI."""
    with gr.Column():
        gr.Markdown("## Device Simulator: Real-time I-V Curve Generation")
        gr.Markdown("Adjust MOSFET parameters and see I-V curves update instantly.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Geometry")
                W = gr.Slider(1, 50, 10, step=0.1, label="Width W (mm)")
                L = gr.Slider(18, 500, 180, step=1, label="Length L (nm)")

                gr.Markdown("### Threshold")
                VTH0 = gr.Slider(0.1, 1.0, 0.45, step=0.01, label="VTH0 (V)")
                GAMMA = gr.Slider(0.0, 2.0, 0.4, step=0.01, label="GAMMA (√V)")
                PHI = gr.Slider(0.3, 1.2, 0.65, step=0.01, label="PHI (V)")

                gr.Markdown("### Transport")
                U0 = gr.Slider(50, 800, 400, step=1, label="U0 (cm²/V·s)")
                THETA = gr.Slider(0.0, 0.3, 0.05, step=0.01, label="THETA (V⁻¹)")
                VSAT = gr.Slider(5, 20, 10, step=0.1, label="VSAT (×10⁶ cm/s)")

                gr.Markdown("### Short-Channel Effects")
                ETA0 = gr.Slider(0.0, 0.2, 0.05, step=0.01, label="ETA0 (DIBL)")
                LAMBDA = gr.Slider(0.0, 0.2, 0.05, step=0.01, label="LAMBDA (CLM)")
                N0 = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="N0 (Subthreshold)")

                gr.Markdown("### Parasitics")
                RD = gr.Slider(0, 200, 20, step=1, label="RD (Ω)")
                RS = gr.Slider(0, 200, 20, step=1, label="RS (Ω)")

            with gr.Column(scale=2):
                gr.Markdown("### Sweep Settings")
                with gr.Row():
                    vgs_min = gr.Slider(-0.5, 0.0, -0.3, step=0.05, label="Vgs min (V)")
                    vgs_max = gr.Slider(0.5, 3.0, 2.5, step=0.05, label="Vgs max (V)")
                    vds_min = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Vds min (V)")
                    vds_max = gr.Slider(1.0, 5.0, 2.5, step=0.05, label="Vds max (V)")

                plot_output = gr.Plot(label="I-V Curves", value=None)

    inputs = [W, L, VTH0, GAMMA, PHI, U0, THETA, VSAT, ETA0, LAMBDA, N0, RD, RS,
              vgs_min, vgs_max, vds_min, vds_max]

    for inp in inputs:
        inp.change(fn=simulate_device, inputs=inputs, outputs=plot_output)

    return plot_output


# ---------------------------------------------------------------------------
# Tab 2: Parameter Extraction
# ---------------------------------------------------------------------------

_extraction_state = {"target_params": None, "target_curves": None, "result": None}


def generate_target(n_params_to_vary=5):
    """Generate synthetic target I-V data with known (hidden) parameters."""
    p = MOSFETParamsLevel3()
    # Vary some parameters randomly
    rng = np.random.RandomState(42)
    p.VTH0 = rng.uniform(0.3, 0.7)
    p.U0 = rng.uniform(200, 500)
    p.THETA = rng.uniform(0.02, 0.10)
    p.VSAT = rng.uniform(6e6, 12e6)
    p.ETA0 = rng.uniform(0.03, 0.08)

    model = MOSFETLevel3(p)
    id_vg, id_vd = generate_iv_curves(model)

    _extraction_state["target_params"] = p
    _extraction_state["target_curves"] = (id_vg, id_vd)

    # Also compute Gm
    gm = np.gradient(id_vg.ids[-1], id_vg.vgs)

    info = (
        f"Target generated with {n_params_to_vary} varied parameters.\n\n"
        f"VTH0 = {p.VTH0:.4f} V\n"
        f"U0   = {p.U0:.1f} cm²/V·s\n"
        f"THETA = {p.THETA:.4f} V⁻¹\n"
        f"VSAT = {p.VSAT/1e6:.1f} ×10⁶ cm/s\n"
        f"ETA0 = {p.ETA0:.4f}\n"
    )
    return plot_iv_curves(id_vg, id_vd, "Target MOSFET"), info


def run_extraction(method="de"):
    """Run two-stage extraction on the current target."""
    if _extraction_state["target_curves"] is None:
        return None, "Please generate target data first."

    id_vg, id_vd = _extraction_state["target_curves"]

    # Build CurveData containers
    tg_vg = CurveData(
        vgs=id_vg.vgs, vds=id_vg.vds_values, ids=id_vg.ids,
    )
    tg_vd = CurveData(
        vgs=id_vd.vgs_values, vds=id_vd.vds, ids=id_vd.ids,
    )

    objective = ExtractionObjective(target_idvg=tg_vg, target_idvd=tg_vd)

    result = two_stage_extraction(
        objective,
        model_cls=MOSFETLevel3,
        stage1=method,
        stage1_options={"max_iter": 50},
        verbose=False,
    )

    _extraction_state["result"] = result

    # Generate extracted curves
    from src.extraction.objective import vector_to_params
    ext_params = vector_to_params(result["x_refined"], MOSFETLevel3)
    ext_model = MOSFETLevel3(ext_params)
    ext_vg, ext_vd = generate_iv_curves(ext_model)

    fig = plot_extraction_comparison(id_vg, ext_vg)

    # Build report text
    p_tgt = _extraction_state["target_params"]
    report = "### Extraction Results\n\n"
    report += f"Final cost: {result['cost_refined']:.6e}\n"
    report += f"Total NFev: {result['nfev_total']}\n\n"
    report += "| Parameter | True | Extracted | Error |\n"
    report += "|-----------|------|-----------|-------|\n"

    param_names = result["param_names"]
    x_ref = result["x_refined"]
    for i, name in enumerate(param_names):
        if hasattr(p_tgt, name):
            true_val = getattr(p_tgt, name)
            ext_val = x_ref[i]
            err = abs(true_val - ext_val) / max(abs(true_val), 1e-12) * 100
            report += f"| {name} | {true_val:.4g} | {ext_val:.4g} | {err:.2f}% |\n"

    return fig, report


def build_extraction_tab():
    """Build the parameter extraction tab UI."""
    with gr.Column():
        gr.Markdown("## Parameter Extraction")

        with gr.Row():
            with gr.Column(scale=1):
                btn_target = gr.Button("Generate Target Data", variant="primary")
                method = gr.Radio(["de", "pso"], value="de", label="Global Method")
                btn_extract = gr.Button("Run Extraction", variant="secondary")
                info_text = gr.Markdown("")

            with gr.Column(scale=2):
                plot_out = gr.Plot(label="Visualization")

        report_out = gr.Markdown("")

    btn_target.click(fn=generate_target, outputs=[plot_out, info_text])
    btn_extract.click(fn=run_extraction, inputs=[method],
                      outputs=[plot_out, report_out])

    return plot_out


# ---------------------------------------------------------------------------
# Tab 3: Sensitivity Analysis
# ---------------------------------------------------------------------------

_sensitivity_cache = {}


def run_sensitivity_analysis():
    """Run Sobol' sensitivity analysis on the MOSFET model."""
    # Define sensitivity objective: average Id-Vg RMSE
    p_default = MOSFETParamsLevel3()
    base_model = MOSFETLevel3(p_default)
    base_vg, base_vd = generate_iv_curves(base_model)

    bounds = ParameterBounds(
        names=["VTH0", "GAMMA", "PHI", "U0", "THETA", "VSAT", "ETA0", "LAMBDA", "N0", "RD", "RS"],
        bounds=[
            (0.2, 0.8), (0.2, 1.0), (0.4, 1.0),
            (200, 600), (0.02, 0.15), (6e6, 14e6),
            (0.02, 0.12), (0.02, 0.10), (1.2, 2.5),
            (5, 100), (5, 100),
        ],
    )

    def cost_fn(x):
        p = MOSFETParamsLevel3(
            VTH0=float(x[0]), GAMMA=float(x[1]), PHI=float(x[2]),
            U0=float(x[3]), THETA=float(x[4]), VSAT=float(x[5]),
            ETA0=float(x[6]), LAMBDA=float(x[7]), N0=float(x[8]),
            RD=float(x[9]), RS=float(x[10]),
        )
        m = MOSFETLevel3(p)
        vg, vd = generate_iv_curves(m)
        # RMSE in log space vs baseline
        ids_log = np.log10(np.maximum(base_vg.ids, 1e-12))
        sim_log = np.log10(np.maximum(vg.ids, 1e-12))
        return float(np.sqrt(np.mean((ids_log - sim_log) ** 2)))

    analyzer = SobolAnalyzer(cost_fn, bounds)
    result = analyzer.analyze(n_base=256, verbose=False)

    _sensitivity_cache["result"] = result
    fig = plot_sensitivity_heatmap(result)

    # Build ranking text
    names = result["param_names"]
    s1 = result["S1"]
    st = result["ST"]
    ranking = sorted(zip(names, s1, st), key=lambda x: -x[2])

    text = "### Sensitivity Ranking (Total-Effect ST)\n\n"
    text += "| Rank | Parameter | S1 | ST |\n"
    text += "|------|-----------|----|----|\n"
    for rank, (name, s1_val, st_val) in enumerate(ranking, 1):
        text += f"| {rank} | {name} | {s1_val:.4f} | {st_val:.4f} |\n"

    return fig, text


def build_sensitivity_tab():
    """Build the sensitivity analysis tab UI."""
    with gr.Column():
        gr.Markdown("## Sensitivity Analysis")
        gr.Markdown("Sobol' variance-based sensitivity analysis identifies "
                    "which parameters most strongly influence I-V curves.")

        btn_run = gr.Button("Run Sobol' Analysis", variant="primary")
        plot_out = gr.Plot(label="Sensitivity Indices")
        text_out = gr.Markdown("")

    btn_run.click(fn=run_sensitivity_analysis, outputs=[plot_out, text_out])

    return plot_out


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def create_app():
    """Create the Gradio Blocks app."""
    with gr.Blocks(title="SPICE Model Toolkit") as app:
        gr.Markdown(
            """# SPICE Model Parameter Extraction Toolkit

            Semiconductor device SPICE model simulation, parameter extraction,
            and sensitivity analysis dashboard.
            """
        )

        with gr.Tab("Device Simulator"):
            build_simulator_tab()

        with gr.Tab("Parameter Extraction"):
            build_extraction_tab()

        with gr.Tab("Sensitivity Analysis"):
            build_sensitivity_tab()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, theme=gr.themes.Soft())
