"""The four figures a property-prediction README needs.

Matplotlib only, no seaborn, no styling dependencies. Every function takes an
optional ``ax`` so panels can be composed into one figure.

The set is deliberately small. A README with these four, each answering a
question a hiring manager or reviewer would actually ask, beats twenty plots
that answer none:

* :func:`parity_plot` -- is it accurate, and where does it fail?
* :func:`calibration_plot` -- can I trust the error bars?
* :func:`selective_prediction_plot` -- how good is it if I only act on the
  confident predictions?
* :func:`split_comparison_plot` -- how much of the performance was the split?
"""

from __future__ import annotations

import numpy as np

from .metrics import calibration_curve, regression_metrics, selective_prediction_curve

__all__ = [
    "parity_plot",
    "calibration_plot",
    "selective_prediction_plot",
    "split_comparison_plot",
    "summary_figure",
]


def _get_ax(ax=None, figsize=(5, 5)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def parity_plot(
    y_true,
    y_pred,
    sigma=None,
    ax=None,
    title: str = "Parity",
    units: str = "",
    annotate: bool = True,
    color_by=None,
    color_label: str = "",
):
    """Predicted vs observed, with an optional error bar or colour dimension.

    Colouring by chemical family turns a generic scatter into an error analysis:
    a cluster of one colour sitting off the diagonal is a finding you can write
    a paragraph about, and that paragraph is what distinguishes a chemist doing
    ML from an ML engineer guessing at chemistry.
    """
    ax = _get_ax(ax)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo or 1.0)
    lims = (lo - pad, hi + pad)

    if sigma is not None:
        ax.errorbar(
            y_true, y_pred, yerr=np.asarray(sigma, dtype=float),
            fmt="none", ecolor="gray", alpha=0.4, linewidth=0.8, zorder=1,
        )

    if color_by is not None:
        cats = list(dict.fromkeys(color_by))
        cmap = __import__("matplotlib.pyplot", fromlist=["pyplot"]).get_cmap("tab20")
        for i, cat in enumerate(cats):
            m = np.asarray([c == cat for c in color_by])
            ax.scatter(
                y_true[m], y_pred[m], s=34, alpha=0.85,
                color=cmap(i % 20), edgecolor="white", linewidth=0.5,
                label=str(cat), zorder=2,
            )
        if len(cats) <= 12:
            ax.legend(fontsize=7, title=color_label or None, loc="upper left",
                      framealpha=0.9)
    else:
        ax.scatter(y_true, y_pred, s=34, alpha=0.8, color="#2b6cb0",
                   edgecolor="white", linewidth=0.5, zorder=2)

    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    suffix = f" ({units})" if units else ""
    ax.set_xlabel(f"Observed{suffix}")
    ax.set_ylabel(f"Predicted{suffix}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    if annotate:
        m = regression_metrics(y_true, y_pred)
        ax.text(
            0.97, 0.03,
            f"RMSE {m['rmse']:.1f}{(' ' + units) if units else ''}\n"
            f"MAE {m['mae']:.1f}\nR² {m['r2']:.2f}\nρ {m['spearman']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                      edgecolor="0.8"),
        )
    return ax


def calibration_plot(y_true, y_pred, sigma, ax=None, label: str = "model",
                     title: str = "Calibration"):
    """Observed vs expected coverage. On the diagonal is calibrated.

    Points below the diagonal mean the intervals are too narrow -- the model is
    overconfident. This is the default state of an unrecalibrated ensemble, so
    plotting before and after recalibration on the same axes is a good look.
    """
    ax = _get_ax(ax, figsize=(4.5, 4.5))
    expected, observed = calibration_curve(y_true, y_pred, sigma)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="ideal")
    ax.plot(expected, observed, "o-", markersize=4, linewidth=1.6, label=label)
    ax.fill_between(expected, expected, observed, alpha=0.15)
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def selective_prediction_plot(y_true, y_pred, sigma, ax=None, metric: str = "rmse",
                              units: str = "", title: str = "Selective prediction"):
    """Error against coverage after discarding the least confident predictions.

    The most directly actionable figure in the set: it answers "if I only run
    the 30% of candidates the model is most sure about, what error should I
    expect?" A flat curve means the uncertainty is not usable for triage, no
    matter how well calibrated it is on average.
    """
    ax = _get_ax(ax, figsize=(5, 4))
    curve = selective_prediction_curve(y_true, y_pred, sigma)
    ax.plot(curve["coverage"] * 100, curve[metric], "o-", markersize=4,
            linewidth=1.8, color="#2b6cb0")
    full = curve[metric][-1]
    ax.axhline(full, color="gray", linestyle="--", linewidth=1,
               label=f"full coverage ({full:.1f})")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel(f"{metric.upper()}{(' (' + units + ')') if units else ''}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def split_comparison_plot(results: dict[str, dict[str, float]], ax=None,
                          metric: str = "rmse", units: str = "",
                          title: str = "Performance by split"):
    """Grouped bars of one metric across splits and models.

    ``results`` maps ``"split_name"`` to ``{"model_name": value}``. The visual
    point is the height difference between random and structured splits: that
    gap is what a random-split-only paper is hiding.
    """
    import matplotlib.pyplot as plt

    ax = _get_ax(ax, figsize=(7, 4))
    splits = list(results)
    models = list(dict.fromkeys(m for s in splits for m in results[s]))
    x = np.arange(len(splits))
    width = 0.8 / max(len(models), 1)
    cmap = plt.get_cmap("tab10")

    for i, model in enumerate(models):
        vals = [results[s].get(model, np.nan) for s in splits]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=model,
               color=cmap(i % 10), edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=15, ha="right")
    ax.set_ylabel(f"{metric.upper()}{(' (' + units + ')') if units else ''}")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return ax


def summary_figure(y_true, y_pred, sigma, color_by=None, units: str = "",
                   suptitle: str = ""):
    """Three-panel figure: parity, calibration, selective prediction.

    This is the image to put at the top of a README, before any prose.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    parity_plot(y_true, y_pred, sigma=sigma, ax=axes[0], units=units,
                color_by=color_by, title="Predicted vs observed")
    calibration_plot(y_true, y_pred, sigma, ax=axes[1])
    selective_prediction_plot(y_true, y_pred, sigma, ax=axes[2], units=units)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig
