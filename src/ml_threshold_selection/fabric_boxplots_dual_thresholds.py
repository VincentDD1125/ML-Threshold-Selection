from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def compute_fabric_params(eigenvals: Sequence[float]) -> Tuple[float, float]:
    """
    Compute T and P' (Jelínek 1981) from eigenvalues [λ1, λ2, λ3].
    Returns:
        (T, P_prime)
    """
    vals = np.asarray(eigenvals, dtype=float)
    if vals.size != 3 or np.any(vals <= 0):
        return np.nan, np.nan

    l1, l2, l3 = vals
    ln1, ln2, ln3 = np.log(l1), np.log(l2), np.log(l3)

    # T (Jelínek 1981)
    if abs(ln1 - ln3) < 1e-12:
        T = 0.0
    else:
        T = (ln2 - ln3 - ln1 + ln2) / (ln2 - ln3 + ln1 - ln2)

    # P' (Jelínek 1981)
    lm = (l1 + l2 + l3) / 3.0
    ln_m = np.log(lm)
    P_prime = float(np.exp(np.sqrt(2.0 * ((ln1 - ln_m) ** 2 + (ln2 - ln_m) ** 2 + (ln3 - ln_m) ** 2))))
    return T, P_prime


def plot_param_boxplot_by_volume_thresholds(
    bootstrap_samples: Mapping[float, Sequence[float]],
    param: str = "T",  # "T" or "P'"
    inflection_threshold: Optional[float] = None,
    zero_artifact_threshold: Optional[float] = None,
    particle_counts: Optional[Mapping[float, int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 9),
    dpi: int = 300,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw T or P' parameter boxplots across volume thresholds on a logarithmic x-axis,
    matching the project's styling exactly.
    """
    if param not in ("T", "P'"):
        raise ValueError("param must be either 'T' or \"P'\"")

    if not isinstance(bootstrap_samples, Mapping) or len(bootstrap_samples) == 0:
        raise ValueError("bootstrap_samples must be a non-empty mapping of {threshold: samples}.")

    # 1) Filter and order thresholds
    thresholds: List[float] = []
    box_data: List[np.ndarray] = []
    counts: List[int] = []

    for vt in sorted(bootstrap_samples.keys()):
        arr = np.asarray(bootstrap_samples[vt], dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        thresholds.append(float(vt))
        arr = arr[~np.isnan(arr)]
        box_data.append(arr)
        if particle_counts and vt in particle_counts:
            counts.append(int(particle_counts[vt]))
        else:
            counts.append(int(arr.size))

    if not thresholds:
        raise ValueError("No valid samples after filtering.")

    # 2) Figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    # 3) Boxplot at categorical positions (linear axis to match requested style)
    positions = np.arange(1, len(thresholds) + 1, dtype=float)
    widths = 0.6  # fixed width for consistent appearance on linear axis
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=widths,
        meanline=True,
        showmeans=True,
        meanprops=dict(color="#2C3E50", linewidth=2.0, linestyle="--"),
        medianprops=dict(color="#2C3E50", linewidth=2.0),
        whiskerprops=dict(color="#2C3E50", linewidth=2.0),
        capprops=dict(color="#2C3E50", linewidth=2.0),
    )

    # 4) Color rules (exactly as the project)
    def _box_colors(v_threshold: float) -> Tuple[str, str]:
        # Special threshold highlight (kept subtle & consistent with project: light green)
        if inflection_threshold is not None and math.isclose(v_threshold, inflection_threshold, abs_tol=1e-12):
            return "#90EE90", "#000000"  # light green, black edge
        if zero_artifact_threshold is not None and math.isclose(v_threshold, zero_artifact_threshold, abs_tol=1e-12):
            return "#90EE90", "#000000"  # light green, black edge
        if inflection_threshold is not None and v_threshold < inflection_threshold:
            return "#D3D3D3", "#2C3E50"  # gray
        if (
            inflection_threshold is not None
            and zero_artifact_threshold is not None
            and inflection_threshold < v_threshold < zero_artifact_threshold
        ):
            return "#FFA500", "#2C3E50"  # orange
        return "#ADD8E6", "#2C3E50"      # light blue

    for i, patch in enumerate(bp["boxes"]):
        v = thresholds[i]
        fill, edge = _box_colors(v)
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        patch.set_alpha(0.85)
        patch.set_linewidth(2.0)

    # 5) n labels above boxes
    for i, (box, n) in enumerate(zip(bp["boxes"], counts)):
        verts = box.get_path().vertices
        y_top = float(np.max(verts[:, 1])) if verts.size else float(np.nanmax(box_data[i]))
        ax.text(
            positions[i],
            y_top + 0.06,  # slightly higher to match screenshot spacing
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0F2A3A",
        )

    # 6) Reference line, labels, limits
    if param == "T":
        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.5, label="Isotropic (T=0)")
        ax.set_ylabel("T Parameter")
    else:
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Isotropic (P'=1)")
        ax.set_ylabel("P' Parameter")

    def _fmt_v(v: float) -> str:
        return f"(≥{v:.6g}mm³)"

    ax.set_xticks(positions)
    # To prevent huge bbox due to long strings, cap label length and reduce rotation
    labels = [_fmt_v(v) for v in thresholds]
    labels = [lbl if len(lbl) <= 18 else lbl[:17] + '…' for lbl in labels]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontstyle="italic")

    # Y-limits like screenshot: clamp for T and keep symmetric-ish range
    all_vals = np.concatenate(box_data) if len(box_data) > 0 else np.array([0.0])
    if param == "T" and all_vals.size:
        y_min = max(-1.05, float(np.nanmin(all_vals) - 0.1))
        y_max = min(1.05, float(np.nanmax(all_vals) + 0.1))
        ax.set_ylim(y_min, y_max)

    # Styling: border
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("#1F3545")

    # Title (optional)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend (lines only, as in project’s look)
    legend_handles = []
    if inflection_threshold is not None:
        legend_handles.append(plt.Line2D([0], [0], color="#FFD700", lw=6, label="Inflection Point Threshold"))
    if zero_artifact_threshold is not None:
        legend_handles.append(plt.Line2D([0], [0], color="#FF6B6B", lw=6, label="Zero Artifact Threshold"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    # Clean grid (none) — consistent with screenshot
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        # Avoid extremely large canvas caused by tight bbox with many rotated ticklabels
        fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()

    return fig, ax


