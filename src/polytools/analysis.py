"""Error-analysis helpers: out-of-fold residuals and where they concentrate.

A single test-set RMSE tells you *how much* a model is wrong. Error analysis asks
*where* and *why* — which chemistries it fails on, and whether the failures share a
mechanism. That paragraph, written by someone who knows the chemistry, is worth
more than the model, and it is the part of a portfolio nobody can fake.

The methodological point here is **out-of-fold prediction**. A single train/test
split gives you a residual for only the test fraction — too few points to say
anything about *which families* fail. K-fold out-of-fold prediction gives every
polymer a residual, computed while that polymer was held out, so the analysis
covers the whole dataset without ever letting a model score its own training point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["oof_predictions", "group_bias_table", "worst_predictions"]


def oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
    n_splits: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions: each row predicted by a model that never saw it.

    ``model_factory`` returns a fresh, unfitted model implementing
    ``fit``/``predict`` (any :class:`~polytools.models.BaseRegressor`). Returns an
    array aligned with ``y``, suitable for computing a residual per polymer.

    Uses a plain shuffled K-fold. For an out-of-distribution flavour, pass a model
    and analyse the residuals against a grouped split instead — but for "which
    chemistries does the model get wrong on average", shuffled OOF is the right,
    unbiased view because every point gets a held-out prediction.
    """
    from sklearn.model_selection import KFold

    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model = model_factory().fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])
    return oof


def group_bias_table(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence,
) -> list[dict]:
    """Mean *signed* residual per group, sorted most-under-predicted first.

    Signed residual (``pred - true``) is the point: a negative group mean means the
    model systematically under-predicts that chemistry. The spread from the most
    negative to the most positive group is the model's shrinkage — under-predicting
    the high extremes, over-predicting the low ones — which is usually the headline
    of a Tg error analysis.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups, dtype=object)
    resid = y_pred - y_true

    rows = []
    for g in dict.fromkeys(groups):
        mask = groups == g
        rows.append(
            {
                "group": g,
                "n": int(mask.sum()),
                "mean_signed_residual": float(resid[mask].mean()),
                "mean_abs_residual": float(np.abs(resid[mask]).mean()),
                "mean_true": float(y_true[mask].mean()),
            }
        )
    rows.sort(key=lambda r: r["mean_signed_residual"])
    return rows


def worst_predictions(
    names: Sequence,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n: int = 8,
) -> list[dict]:
    """The ``n`` polymers with the largest absolute residual, worst first."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    order = np.argsort(-np.abs(resid))[:n]
    return [
        {
            "name": names[i],
            "true": float(y_true[i]),
            "pred": float(y_pred[i]),
            "residual": float(resid[i]),
        }
        for i in order
    ]
