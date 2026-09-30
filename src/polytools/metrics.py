"""Regression metrics and uncertainty diagnostics.

Two ideas here are worth more than the rest of the library combined.

**Calibration.** An ensemble will happily report a standard deviation for every
prediction. That number is meaningless until you check that 80% intervals
actually contain the truth 80% of the time. :func:`calibration_curve` and
:func:`miscalibration_area` do that check, and :class:`SigmaRecalibrator` fixes
the usual failure -- deep ensembles are systematically overconfident, and a
single scalar fitted on validation data typically removes most of the error.

**Selective prediction.** The question a lab actually asks is not "how accurate
is the model" but "if I only trust its most confident third, how accurate is
that third?" :func:`selective_prediction_curve` answers exactly that, and its
plot is the most persuasive single figure you can put in a model README,
because it maps directly onto how the model would be used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "regression_metrics",
    "calibration_curve",
    "miscalibration_area",
    "gaussian_nll",
    "error_uncertainty_correlation",
    "selective_prediction_curve",
    "SigmaRecalibrator",
    "uncertainty_metrics",
    "format_metrics",
]


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "spearman") -> float:
    """Correlation that returns NaN instead of warning when an input is constant.

    A constant input is the normal case for baseline models -- the mean
    predictor emits the same value everywhere -- so this is an expected result,
    not an anomaly worth printing a warning about on every run.
    """
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn = stats.spearmanr if kind == "spearman" else stats.pearsonr
            result = fn(a, b)
    value = float(result.statistic)
    return value if np.isfinite(value) else float("nan")


def _as_arrays(*arrays) -> tuple[np.ndarray, ...]:
    out = [np.asarray(a, dtype=float).ravel() for a in arrays]
    n = len(out[0])
    for a in out:
        if len(a) != n:
            raise ValueError("all inputs must have the same length")
    if n == 0:
        raise ValueError("empty input")
    return tuple(out)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, R^2, correlations, and the error at the 90th percentile.

    ``rmse`` and ``mae`` are in the units of the target. ``r2`` is the
    coefficient of determination against the *test* mean, so it can go negative
    on an extrapolation split -- that is informative, not a bug. ``spearman`` is
    included because for screening applications the ranking matters more than
    the absolute error.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    metrics = {
        "n": float(len(y_true)),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "medae": float(np.median(np.abs(err))),
        "p90_abs_error": float(np.percentile(np.abs(err), 90)),
        "max_abs_error": float(np.abs(err).max()),
        "bias": float(err.mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    metrics["pearson"] = _safe_corr(y_true, y_pred, "pearson")
    metrics["spearman"] = _safe_corr(y_true, y_pred, "spearman")
    return metrics


def gaussian_nll(y_true, y_pred, sigma) -> float:
    """Mean negative log-likelihood under a Gaussian predictive distribution.

    The single number that penalizes both inaccuracy and miscalibration. Lower
    is better. A model that reports huge uncertainty everywhere is punished, and
    so is one that reports tiny uncertainty and is wrong.
    """
    y_true, y_pred, sigma = _as_arrays(y_true, y_pred, sigma)
    sigma = np.clip(sigma, 1e-9, None)
    return float(
        (0.5 * np.log(2 * np.pi * sigma ** 2) + (y_true - y_pred) ** 2 / (2 * sigma ** 2)).mean()
    )


def calibration_curve(
    y_true, y_pred, sigma, n_levels: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Expected vs observed coverage of centred Gaussian intervals.

    Returns ``(expected, observed)`` arrays. Plot one against the other; a
    perfectly calibrated model sits on the diagonal. Below the diagonal means
    overconfident (intervals too narrow), the near-universal failure mode of
    deep ensembles.
    """
    y_true, y_pred, sigma = _as_arrays(y_true, y_pred, sigma)
    sigma = np.clip(sigma, 1e-9, None)
    expected = np.linspace(1.0 / n_levels, 1.0 - 1.0 / n_levels, n_levels)
    z = np.abs(y_true - y_pred) / sigma
    observed = np.array(
        [float((z <= stats.norm.ppf(0.5 + p / 2)).mean()) for p in expected]
    )
    return expected, observed


def miscalibration_area(y_true, y_pred, sigma, n_levels: int = 20) -> float:
    """Area between the calibration curve and the diagonal. 0 is perfect.

    Roughly bounded above by 0.5. Report it alongside RMSE; a model with worse
    RMSE and much better calibration is often the more useful one in the lab.
    """
    expected, observed = calibration_curve(y_true, y_pred, sigma, n_levels)
    # np.trapezoid landed in NumPy 2.0 and np.trapz was REMOVED there, so both
    # lookups must be lazy for the declared numpy>=1.24 floor to actually hold.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201
    return float(trapezoid(np.abs(observed - expected), expected))


def error_uncertainty_correlation(y_true, y_pred, sigma) -> float:
    """Spearman correlation between predicted sigma and realized absolute error.

    Calibration and ranking are different properties. A model can be perfectly
    calibrated on average while its uncertainty carries no information about
    *which* predictions are wrong. That case is useless for triage, and only
    this number reveals it. Anything above ~0.3 is doing real work.
    """
    y_true, y_pred, sigma = _as_arrays(y_true, y_pred, sigma)
    abs_err = np.abs(y_true - y_pred)
    return _safe_corr(sigma, abs_err, "spearman")


def selective_prediction_curve(
    y_true, y_pred, sigma, n_points: int = 20
) -> dict[str, np.ndarray]:
    """Error as a function of coverage after discarding the least confident predictions.

    Returns arrays ``coverage``, ``rmse``, ``mae``. The curve should fall
    monotonically as coverage decreases; if it is flat, the uncertainty estimate
    is not usable for triage regardless of how well calibrated it is.
    """
    y_true, y_pred, sigma = _as_arrays(y_true, y_pred, sigma)
    order = np.argsort(sigma)  # most confident first
    n = len(y_true)
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)

    rmses, maes, covs = [], [], []
    for c in coverages:
        k = max(1, int(round(c * n)))
        sel = order[:k]
        err = y_pred[sel] - y_true[sel]
        covs.append(k / n)
        rmses.append(float(np.sqrt((err ** 2).mean())))
        maes.append(float(np.abs(err).mean()))
    return {
        "coverage": np.asarray(covs),
        "rmse": np.asarray(rmses),
        "mae": np.asarray(maes),
    }


@dataclass
class SigmaRecalibrator:
    """Scale predicted sigmas by a single scalar fitted on held-out data.

    Deep ensembles and dropout estimates are almost always overconfident: the
    spread of an ensemble measures disagreement between models, which is only
    one component of predictive error and omits irreducible noise entirely. A
    one-parameter fix recovers most of the calibration at effectively zero cost.

    Fit on **validation** data, apply to test. Fitting on test and reporting the
    result is a leak, and a reviewer will spot it.
    """

    scale: float = 1.0

    def fit(self, y_true, y_pred, sigma) -> SigmaRecalibrator:
        """Fit the scalar by maximum likelihood (closed form for a Gaussian)."""
        y_true, y_pred, sigma = _as_arrays(y_true, y_pred, sigma)
        sigma = np.clip(sigma, 1e-9, None)
        z2 = ((y_true - y_pred) / sigma) ** 2
        self.scale = float(np.sqrt(max(z2.mean(), 1e-12)))
        return self

    def transform(self, sigma) -> np.ndarray:
        return np.asarray(sigma, dtype=float) * self.scale

    def fit_transform(self, y_true, y_pred, sigma) -> np.ndarray:
        return self.fit(y_true, y_pred, sigma).transform(sigma)


def uncertainty_metrics(y_true, y_pred, sigma) -> dict[str, float]:
    """All uncertainty diagnostics in one dict, for logging into a results table."""
    return {
        "nll": gaussian_nll(y_true, y_pred, sigma),
        "miscalibration_area": miscalibration_area(y_true, y_pred, sigma),
        "error_sigma_spearman": error_uncertainty_correlation(y_true, y_pred, sigma),
        "mean_sigma": float(np.mean(sigma)),
    }


def format_metrics(metrics: dict[str, float], precision: int = 3) -> str:
    """Render a metrics dict as a single readable line for logs."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.{precision}f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)
