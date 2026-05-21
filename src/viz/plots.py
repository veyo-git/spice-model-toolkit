"""Publication-quality Matplotlib plots for MOSFET I-V and extraction analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
from io import BytesIO
import base64


# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
})


def figure_to_bytes(fig, fmt="png", dpi=150):
    """Convert a Matplotlib figure to base64-encoded bytes for Gradio."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    if fmt == "base64":
        return base64.b64encode(buf.read()).decode("utf-8")
    return buf


# ---------------------------------------------------------------------------
# I-V Curve plots
# ---------------------------------------------------------------------------

def plot_iv_curves(
    id_vg,
    id_vd=None,
    title="MOSFET I-V Characteristics",
    figsize=(14, 5),
):
    """Plot transfer (Id-Vg) and output (Id-Vd) characteristics.

    Parameters
    ----------
    id_vg : IdVgSweep
    id_vd : IdVdSweep, optional
    title : str
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    if id_vd is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax2 = None

    # --- Transfer curve (log-scale) ---
    for i, vds_val in enumerate(id_vg.vds_values):
        ax1.semilogy(id_vg.vgs, np.maximum(id_vg.ids[i], 1e-12),
                     label=f"Vds = {vds_val:.2f} V")
    ax1.set_xlabel("Vgs (V)")
    ax1.set_ylabel("Id (A)")
    ax1.set_title("Transfer: Id vs Vgs")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(1e-12, None)

    # --- Output curve ---
    if ax2 is not None and id_vd is not None:
        for i, vgs_val in enumerate(id_vd.vgs_values):
            ax2.plot(id_vd.vds, id_vd.ids[i] * 1e3,
                     label=f"Vgs = {vgs_val:.2f} V")
        ax2.set_xlabel("Vds (V)")
        ax2.set_ylabel("Id (mA)")
        ax2.set_title("Output: Id vs Vds")
        ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gm vs Vgs
# ---------------------------------------------------------------------------

def plot_gm(id_vg, vds_idx=-1, figsize=(7, 4)):
    """Plot transconductance Gm = dId/dVgs."""
    fig, ax = plt.subplots(figsize=figsize)
    vds_val = id_vg.vds_values[vds_idx]
    gm = np.gradient(id_vg.ids[vds_idx], id_vg.vgs)

    ax.plot(id_vg.vgs, gm * 1e3, "b-", label=f"Vds = {vds_val:.2f} V")
    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("Gm (mS)")
    ax.set_title("Transconductance: Gm vs Vgs")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Optimization trace
# ---------------------------------------------------------------------------

def plot_optimization_trace(history, title="Optimization Convergence", figsize=(8, 4)):
    """Plot cost vs iteration for optimization monitoring.

    Parameters
    ----------
    history : list of float
        Cost at each iteration.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(history, "b-", alpha=0.7)
    ax.scatter(np.argmin(history), np.min(history),
               c="red", s=60, zorder=5, label=f"Best: {np.min(history):.4e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(
    sensitivity_result,
    title="Sobol' Sensitivity Indices",
    figsize=(9, 5),
):
    """Plot Sobol' first-order and total-effect indices as a grouped bar chart.

    Parameters
    ----------
    sensitivity_result : dict
        Output from SobolAnalyzer.analyze().
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = sensitivity_result["param_names"]
    S1 = sensitivity_result["S1"]
    ST = sensitivity_result["ST"]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, S1, width, label="First-order (S1)", color="steelblue")
    bars2 = ax.bar(x + width / 2, ST, width, label="Total-effect (ST)", color="coral")

    # Mark ST > S1 gap (interaction effects)
    for i in range(len(names)):
        if ST[i] - S1[i] > 0.02:
            ax.annotate("", xy=(x[i] + width / 2, ST[i]),
                        xytext=(x[i] - width / 2, S1[i]),
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sensitivity Index")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, None)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parameter correlation / recovery scatter
# ---------------------------------------------------------------------------

def plot_parameter_correlation(
    true_params, extracted_params, param_names,
    title="Parameter Recovery", figsize=(8, 6),
):
    """Scatter plot of true vs extracted parameters."""
    fig, ax = plt.subplots(figsize=figsize)

    n = len(true_params)
    true_vals = np.array([true_params[nm] for nm in param_names if nm in true_params])
    ext_vals = np.array([extracted_params[nm] for nm in param_names if nm in true_params])
    # Use only matching subset
    names_used = [nm for nm in param_names if nm in true_params]
    n_used = len(names_used)

    # Normalize to [0, 1] for visual comparison
    true_norm = (true_vals - true_vals.min()) / (true_vals.max() - true_vals.min() + 1e-12)
    ext_norm = (ext_vals - true_vals.min()) / (true_vals.max() - true_vals.min() + 1e-12)

    colors = plt.cm.tab20(np.linspace(0, 1, n_used))
    for i in range(n_used):
        ax.scatter(true_norm[i], ext_norm[i], c=[colors[i]], s=60,
                   label=names_used[i], edgecolors="black", linewidth=0.5)

    # Perfect recovery line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

    ax.set_xlabel("True (normalized)")
    ax.set_ylabel("Extracted (normalized)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Extraction comparison (before/after)
# ---------------------------------------------------------------------------

def plot_extraction_comparison(
    target_idvg,
    extracted_idvg,
    vds_idx=-1,
    title="Extraction Fit Quality",
    figsize=(10, 5),
):
    """Overlay target and extracted Id-Vg curves for visual fit assessment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    vds_val = target_idvg.vds_values[vds_idx]
    ids_tgt = target_idvg.ids[vds_idx]
    ids_ext = extracted_idvg.ids[vds_idx]

    # Log-scale overlay
    ax1.semilogy(target_idvg.vgs, np.maximum(ids_tgt, 1e-12),
                 "b-", linewidth=2, label="Target")
    ax1.semilogy(extracted_idvg.vgs, np.maximum(ids_ext, 1e-12),
                 "r--", linewidth=2, label="Extracted")
    ax1.set_xlabel("Vgs (V)")
    ax1.set_ylabel("Id (A)")
    ax1.set_title(f"Id-Vg (Vds={vds_val:.2f}V) — Log Scale")
    ax1.legend()

    # Residual
    residual = ids_tgt - ids_ext
    ax2.plot(target_idvg.vgs, residual * 1e6, "k-", alpha=0.7)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Vgs (V)")
    ax2.set_ylabel("Residual (µA)")
    ax2.set_title("Residual: Target - Extracted")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
