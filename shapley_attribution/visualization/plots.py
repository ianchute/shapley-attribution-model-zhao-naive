"""
Visualization utilities for attribution models.

All functions accept an optional ``ax`` parameter for embedding in
larger figure layouts.  When ``ax=None`` a new figure is created and
``plt.show()`` is called automatically.

Functions
---------
plot_attribution        Bar chart of one fitted model's channel scores.
compare_models          Grouped bar chart comparing 2+ models (± ground truth).
plot_performance        Three-panel metric comparison (NMAE / rank_corr / top-k).
plot_journey            Touchpoint-sequence diagram for a single journey.
plot_journeys_heatmap   Heatmap of the per-journey attribution matrix.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ── colour palette ──────────────────────────────────────────────────────────
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]
_GROUND_TRUTH_COLOR = "#2ca02c"
_HIGHLIGHT = "#e05c00"


def _channel_labels(channels):
    """Return display-friendly channel labels."""
    return [f"Ch {c}" if isinstance(c, (int, np.integer)) else str(c)
            for c in channels]


def _auto_show(ax, standalone):
    if standalone:
        plt.tight_layout()
        plt.show()


# ── plot_attribution ─────────────────────────────────────────────────────────

def plot_attribution(
    model,
    ax=None,
    title=None,
    top_k=None,
    ground_truth=None,
    color="#4C72B0",
):
    """Horizontal bar chart of a fitted model's channel attribution scores.

    Parameters
    ----------
    model : fitted BaseAttributionModel
        Any model with ``get_attribution_array()`` and ``channels_``.
    ax : matplotlib.axes.Axes or None
        Target axes; a new figure is created when None.
    title : str or None
        Plot title.  Defaults to the model class name.
    top_k : int or None
        Show only the top-k channels (by attribution score).
    ground_truth : ndarray of shape (n_channels,) or None
        If provided, overlays ground-truth markers (◆) on the bars.
    color : str
        Bar colour.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, max(3, len(model.channels_) * 0.45)))

    scores = model.get_attribution_array()
    channels = model.channels_
    labels = _channel_labels(channels)

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        order = order[:top_k]
    scores_sorted = scores[order]
    labels_sorted = [labels[i] for i in order]

    bars = ax.barh(labels_sorted[::-1], scores_sorted[::-1], color=color,
                   edgecolor="white", linewidth=0.6, height=0.65)

    # Value annotations
    for bar, val in zip(bars, scores_sorted[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#333")

    # Ground truth overlay
    if ground_truth is not None:
        gt = np.asarray(ground_truth)
        if gt.sum() > 0:
            gt = gt / gt.sum()
        for idx_sorted, orig_idx in enumerate(order[::-1]):
            ax.plot(gt[orig_idx], idx_sorted, marker="D", color=_GROUND_TRUTH_COLOR,
                    markersize=7, zorder=5, label="Ground truth" if idx_sorted == 0 else "")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel("Attribution score (normalized)", fontsize=9)
    ax.set_title(title or type(model).__name__, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, scores_sorted.max() * 1.18)

    _auto_show(ax, standalone)
    return ax


# ── compare_models ───────────────────────────────────────────────────────────

def compare_models(
    models,
    ground_truth=None,
    ax=None,
    title="Attribution Comparison",
    channel_names=None,
):
    """Grouped bar chart comparing attribution scores across multiple models.

    Parameters
    ----------
    models : dict[str, fitted BaseAttributionModel] or dict[str, ndarray]
        Mapping from model name to either a fitted model or a pre-computed
        attribution array of shape (n_channels,).
    ground_truth : ndarray of shape (n_channels,) or None
        If provided, shown as a separate bar group.
    ax : matplotlib.axes.Axes or None
    title : str
    channel_names : list or None
        Override channel axis labels.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    standalone = ax is None

    # Resolve model arrays
    model_arrays = {}
    ref_channels = None
    for name, obj in models.items():
        if hasattr(obj, "get_attribution_array"):
            arr = obj.get_attribution_array()
            if ref_channels is None:
                ref_channels = obj.channels_
        else:
            arr = np.asarray(obj, dtype=float)
        if arr.sum() > 0:
            arr = arr / arr.sum()
        model_arrays[name] = arr

    n_channels = len(next(iter(model_arrays.values())))
    if channel_names is None:
        channel_names = (_channel_labels(ref_channels) if ref_channels is not None
                         else [f"Ch {i}" for i in range(n_channels)])

    all_names = list(model_arrays.keys())
    if ground_truth is not None:
        gt = np.asarray(ground_truth, dtype=float)
        gt = gt / gt.sum() if gt.sum() > 0 else gt
        all_names = ["Ground Truth"] + all_names
        model_arrays = {"Ground Truth": gt, **model_arrays}

    n_models = len(all_names)
    colors = ([_GROUND_TRUTH_COLOR] if ground_truth is not None else []) + \
             _PALETTE[:len(models)]

    x = np.arange(n_channels)
    width = 0.8 / n_models

    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, n_channels * 1.1), 4.5))

    for i, (name, color) in enumerate(zip(all_names, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, model_arrays[name], width * 0.9,
                      label=name, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=30 if n_channels > 8 else 0,
                       ha="right", fontsize=9)
    ax.set_ylabel("Attribution score (normalized)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(arr.max() for arr in model_arrays.values()) * 1.2)

    _auto_show(ax, standalone)
    return ax


# ── plot_performance ──────────────────────────────────────────────────────────

def plot_performance(
    results,
    ax=None,
    title="Model Performance",
):
    """Three-panel bar chart: NMAE (lower=better), rank corr, top-k overlap.

    Parameters
    ----------
    results : dict[str, dict]
        Output of ``attribution_summary()``.  Keys are model names; values are
        dicts with keys ``"nmae"``, ``"rank_correlation"``, ``"top_3_overlap"``.
    ax : list of 3 matplotlib.axes.Axes or None
        If None, a new figure with 3 side-by-side subplots is created.
    title : str

    Returns
    -------
    axes : list of matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        axes = ax

    names = list(results.keys())
    colors = _PALETTE[:len(names)]

    metrics = [
        ("nmae", "NMAE", "lower is better", True),
        ("rank_correlation", "Rank Correlation (Spearman ρ)", "higher is better", False),
        ("top_3_overlap", "Top-3 Overlap", "higher is better", False),
    ]

    for ax_i, (key, label, note, lower_better) in zip(axes, metrics):
        vals = [float(results[n].get(key, 0)) for n in names]

        # Highlight best bar
        best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
        bar_colors = [_HIGHLIGHT if i == best_idx else c
                      for i, c in enumerate(colors)]

        bars = ax_i.bar(names, vals, color=bar_colors,
                        edgecolor="white", linewidth=0.6)

        for bar, val in zip(bars, vals):
            ax_i.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax_i.set_title(label, fontsize=10, fontweight="bold")
        ax_i.set_ylabel(note, fontsize=8, color="#666")
        ax_i.spines[["top", "right"]].set_visible(False)
        ax_i.tick_params(axis="x", rotation=30)

        ylim_top = max(vals) * 1.22 if max(vals) > 0 else 0.1
        ax_i.set_ylim(0, ylim_top)

    _auto_show(axes, standalone)
    return axes


# ── plot_journey ──────────────────────────────────────────────────────────────

def plot_journey(
    journey,
    model=None,
    converted=True,
    ax=None,
    title=None,
    channel_labels=None,
):
    """Sequence diagram for a single customer journey.

    Renders touchpoints as labelled boxes connected by arrows.
    When a fitted model is provided, box colour encodes attribution weight.

    Parameters
    ----------
    journey : list
        Ordered list of channel identifiers.
    model : fitted BaseAttributionModel or None
        If provided, touchpoint boxes are coloured by channel attribution.
    converted : bool
        Whether the journey ended in a conversion (affects end-node style).
    ax : matplotlib.axes.Axes or None
    title : str or None
    channel_labels : dict or None
        Mapping from channel id to display name.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    standalone = ax is None
    n = len(journey)
    fig_width = max(6, n * 1.5)

    if standalone:
        fig, ax = plt.subplots(figsize=(fig_width, 2.6))

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    # Attribution weights for colouring
    if model is not None and hasattr(model, "attribution_"):
        scores = model.get_attribution()
        max_score = max(scores.values()) if scores else 1.0
        cmap = cm.get_cmap("Blues")
        norm = Normalize(vmin=0, vmax=max_score)
    else:
        scores = {}
        cmap = None
        norm = None

    for i, ch in enumerate(journey):
        x = i + 0.5

        # Box colour
        if cmap is not None and ch in scores:
            fc = cmap(norm(scores[ch]))
        else:
            fc = "#d0d8e8"

        label = (channel_labels.get(ch, str(ch)) if channel_labels
                 else (f"Ch {ch}" if isinstance(ch, (int, np.integer)) else str(ch)))

        box = mpatches.FancyBboxPatch(
            (x - 0.42, 0.25), 0.84, 0.5,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor="#555", linewidth=1.2, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#222", zorder=4)

        if scores and ch in scores:
            ax.text(x, 0.15, f"{scores[ch]:.3f}", ha="center", va="top",
                    fontsize=7.5, color="#555")

        # Arrow to next touchpoint
        if i < n - 1:
            ax.annotate(
                "", xy=(x + 0.58, 0.5), xytext=(x + 0.42, 0.5),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.4),
            )

    # Conversion outcome node
    outcome_x = n + 0.5
    if converted:
        outcome_color = "#2ca02c"
        outcome_label = "✓ Convert"
    else:
        outcome_color = "#d62728"
        outcome_label = "✗ No conv."

    ax.annotate(
        "", xy=(outcome_x - 0.1, 0.5), xytext=(n + 0.08, 0.5),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.4),
    )
    circle = plt.Circle((outcome_x, 0.5), 0.28, color=outcome_color,
                         zorder=3, alpha=0.9)
    ax.add_patch(circle)
    ax.text(outcome_x, 0.5, outcome_label, ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold", zorder=4)

    if cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal",
                     fraction=0.04, pad=0.02, label="Attribution score")

    ax.set_title(title or f"Journey ({n} touchpoints)", fontsize=10,
                 fontweight="bold", pad=8)

    _auto_show(ax, standalone)
    return ax


# ── plot_journeys_heatmap ─────────────────────────────────────────────────────

def plot_journeys_heatmap(
    model,
    journeys,
    conversions=None,
    ax=None,
    title="Per-Journey Attribution Heatmap",
    max_journeys=60,
    sort_by_channel=None,
):
    """Heatmap of the per-journey attribution matrix from ``model.transform()``.

    Parameters
    ----------
    model : fitted BaseAttributionModel
    journeys : list of list
        Journey data to transform.
    conversions : array-like of shape (n_journeys,) or None
        If provided, converted journeys are marked with a side-bar.
    ax : matplotlib.axes.Axes or None
    title : str
    max_journeys : int
        Cap on number of rows shown (for readability).
    sort_by_channel : int or None
        If provided, rows are sorted by that channel's score descending.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    standalone = ax is None

    matrix = model.transform(journeys)

    # Optionally keep only converting journeys for clarity
    if conversions is not None:
        conv = np.asarray(conversions)
        converting_idx = np.where(conv == 1)[0]
        matrix = matrix[converting_idx]
        if sort_by_channel is not None:
            order = np.argsort(matrix[:, sort_by_channel])[::-1]
            matrix = matrix[order]
        matrix = matrix[:max_journeys]
    else:
        matrix = matrix[:max_journeys]

    n_journeys_shown, n_channels = matrix.shape
    labels = _channel_labels(model.channels_)

    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, n_channels * 0.9),
                                        max(4, n_journeys_shown * 0.18)))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="Attribution score", fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_channels))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Journey index" + (" (converting only)" if conversions is not None else ""),
                  fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    if n_journeys_shown <= 30:
        ax.set_yticks(range(n_journeys_shown))
        ax.set_yticklabels(range(n_journeys_shown), fontsize=7)

    _auto_show(ax, standalone)
    return ax
