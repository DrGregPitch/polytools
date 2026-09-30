"""The baseline ladder, with ensemble uncertainty.

Every model here exposes ``fit(X, y)`` and ``predict(X, return_std=False)``.
The uniform interface is the point: swapping a graph neural network into the
benchmark later should require implementing two methods, not rewriting the
evaluation harness.

Climb the ladder in order and report every rung:

1. :class:`MeanRegressor` -- predicts the training mean. If a model cannot beat
   this on your hardest split, that is the headline result and you should say so.
2. :class:`RidgeDescriptorRegressor` -- a linear model on ~20 interpretable
   descriptors. Frequently within a few degrees of far fancier models on small
   polymer datasets.
3. :class:`GBMRegressor` -- gradient boosting on fingerprints and descriptors.
   The baseline a deep model actually has to beat.

Only rung 3 is a serious competitor, and that is exactly why rungs 1 and 2 must
appear in the table. A GNN beating a mean predictor is not a result. A GNN
beating a tuned GBM on a cluster split is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "BaseRegressor",
    "MeanRegressor",
    "RidgeDescriptorRegressor",
    "GBMRegressor",
    "EnsembleRegressor",
    "lightgbm_available",
    "HAS_LIGHTGBM",
]

import importlib.util
import sys


def lightgbm_available() -> bool:
    """Whether LightGBM can be imported, without actually importing it.

    LightGBM is imported *lazily* (only when a LightGBM-backed model is built)
    rather than at module load. The reason is a hard macOS incompatibility:
    LightGBM and PyTorch each bundle their own OpenMP runtime, and loading both
    into one process segfaults the interpreter during parallel work. Importing
    LightGBM eagerly here would load its OpenMP the moment anyone did
    ``import polytools``, poisoning every later torch call. See
    :meth:`GBMRegressor._make` for how the backend is chosen to avoid the clash.
    """
    return importlib.util.find_spec("lightgbm") is not None


#: Back-compat flag. Reflects availability, not that LightGBM has been imported.
HAS_LIGHTGBM = lightgbm_available()


class BaseRegressor:
    """Interface every model in the benchmark implements."""

    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseRegressor:
        raise NotImplementedError

    def predict(self, X: np.ndarray, return_std: bool = False):
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class MeanRegressor(BaseRegressor):
    """Predicts the training mean, with the training standard deviation as sigma.

    The floor. Its RMSE is the standard deviation of the test set, so any model
    with a lower RMSE is doing *something*. On an extrapolation split this
    baseline is surprisingly hard to beat, which is itself the finding.
    """

    name = "mean"

    def __init__(self) -> None:
        self._mean = 0.0
        self._std = 1.0

    def fit(self, X, y) -> MeanRegressor:
        y = np.asarray(y, dtype=float)
        self._mean = float(y.mean())
        self._std = float(y.std()) or 1.0
        return self

    def predict(self, X, return_std: bool = False):
        n = len(X)
        mu = np.full(n, self._mean)
        if return_std:
            return mu, np.full(n, self._std)
        return mu


class RidgeDescriptorRegressor(BaseRegressor):
    """Standardized ridge regression with the penalty chosen by internal CV.

    The reported sigma is a single **constant**: the training residual standard
    deviation, identical for every prediction. That makes the intervals roughly
    the right average size, but carries zero ranking information -- do not feed
    it to selective prediction or error-uncertainty correlation and expect
    triage (the correlation is undefined for a constant). Use the ensemble or
    quantile models when per-point uncertainty matters.
    """

    name = "ridge"

    def __init__(self, alphas: Sequence[float] | None = None) -> None:
        self.alphas = list(alphas) if alphas is not None else list(np.logspace(-3, 4, 30))
        self._pipe: Pipeline | None = None
        self._resid_std = 1.0

    def fit(self, X, y) -> RidgeDescriptorRegressor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_splits = int(min(5, max(2, len(y) // 3)))
        self._pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas, cv=n_splits)),
            ]
        )
        self._pipe.fit(X, y)
        resid = y - self._pipe.predict(X)
        dof = max(len(y) - 1, 1)
        self._resid_std = float(np.sqrt((resid ** 2).sum() / dof)) or 1.0
        return self

    def predict(self, X, return_std: bool = False):
        if self._pipe is None:
            raise RuntimeError("call fit() first")
        mu = self._pipe.predict(np.asarray(X, dtype=float))
        if return_std:
            return mu, np.full(len(mu), self._resid_std)
        return mu


class GBMRegressor(BaseRegressor):
    """Gradient boosting. Uses LightGBM when safe, else sklearn's HistGBM.

    Uncertainty via quantile regression: separate models for the 16th and 84th
    percentiles give a sigma estimate that, unlike an ensemble spread, captures
    heteroscedastic noise. Set ``quantile_uncertainty=False`` to skip the two
    extra fits when you only need point predictions.

    Backend selection (``backend="auto"``): LightGBM if it is installed **and**
    PyTorch is not already loaded in this process, otherwise scikit-learn's
    ``HistGradientBoostingRegressor``. The torch check exists because LightGBM's
    and PyTorch's bundled OpenMP runtimes crash when both are active in one
    process on macOS; sklearn's booster has no such clash, so it is the safe
    choice whenever the GNN is in play. Force a backend with ``backend="lightgbm"``
    or ``backend="sklearn"`` if you know your environment is clean.
    """

    name = "gbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        quantile_uncertainty: bool = True,
        random_state: int = 0,
        backend: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
        )
        self.quantile_uncertainty = quantile_uncertainty
        self.backend = backend
        self.verbose = verbose
        self._model = None
        self._lo = None
        self._hi = None
        self._resid_std = 1.0

    def _use_lightgbm(self) -> bool:
        if self.backend == "lightgbm":
            return True
        if self.backend == "sklearn":
            return False
        if self.backend != "auto":
            raise ValueError(f"unknown backend {self.backend!r}")
        # auto: LightGBM only if available and torch is not loaded in this process
        return lightgbm_available() and "torch" not in sys.modules

    def _make(self, objective: str | None = None, alpha: float | None = None):
        if self._use_lightgbm():
            import lightgbm as lgb  # lazy: keeps its OpenMP out of torch-only runs

            kwargs = dict(self.params, verbose=-1)
            if objective == "quantile":
                kwargs.update(objective="quantile", alpha=alpha)
            return lgb.LGBMRegressor(**kwargs)
        from sklearn.ensemble import HistGradientBoostingRegressor

        kwargs = dict(
            max_iter=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            max_leaf_nodes=self.params["num_leaves"],
            min_samples_leaf=self.params["min_child_samples"],
            random_state=self.params["random_state"],
        )
        if objective == "quantile":
            kwargs.update(loss="quantile", quantile=alpha)
        return HistGradientBoostingRegressor(**kwargs)

    def fit(self, X, y) -> GBMRegressor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._model = self._make().fit(X, y)
        resid = y - self._model.predict(X)
        self._resid_std = float(np.std(resid)) or 1.0

        if self.quantile_uncertainty and len(y) >= 20:
            self._lo = self._make("quantile", 0.16).fit(X, y)
            self._hi = self._make("quantile", 0.84).fit(X, y)
        return self

    def predict(self, X, return_std: bool = False):
        if self._model is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=float)
        mu = self._model.predict(X)
        if not return_std:
            return mu
        if self._lo is not None and self._hi is not None:
            sigma = (self._hi.predict(X) - self._lo.predict(X)) / 2.0
            sigma = np.clip(sigma, 1e-6, None)
        else:
            sigma = np.full(len(mu), self._resid_std)
        return mu, sigma


class EnsembleRegressor(BaseRegressor):
    """Deep-ensemble-style wrapper: refit a model under several seeds.

    Sigma is the spread of member predictions, optionally combined in quadrature
    with each member's own sigma. Bootstrap resampling (``bootstrap=True``)
    increases member diversity and usually improves the uncertainty estimate on
    small datasets, at the cost of slightly worse point predictions.

    The resulting sigma is nearly always **overconfident**, because ensemble
    disagreement captures only epistemic uncertainty. Pass the validation set
    through :class:`polytools.metrics.SigmaRecalibrator` before reporting
    calibration on test.
    """

    name = "ensemble"

    def __init__(
        self,
        factory,
        n_members: int = 5,
        bootstrap: bool = True,
        include_member_sigma: bool = True,
        random_state: int = 0,
    ) -> None:
        self.factory = factory
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.include_member_sigma = include_member_sigma
        self.random_state = random_state
        self.members: list[BaseRegressor] = []
        base_name = getattr(factory(0), "name", "model")
        self.name = f"ensemble[{base_name}]x{n_members}"

    def fit(self, X, y) -> EnsembleRegressor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.members = []
        for i in range(self.n_members):
            if self.bootstrap:
                idx = rng.integers(0, len(y), len(y))
                Xi, yi = X[idx], y[idx]
            else:
                Xi, yi = X, y
            self.members.append(self.factory(i).fit(Xi, yi))
        return self

    def predict(self, X, return_std: bool = False):
        if not self.members:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=float)

        mus, sigs = [], []
        for m in self.members:
            if self.include_member_sigma:
                mu, sd = m.predict(X, return_std=True)
                sigs.append(sd)
            else:
                mu = m.predict(X)
            mus.append(mu)

        mus = np.vstack(mus)
        mean = mus.mean(axis=0)
        if not return_std:
            return mean

        epistemic = mus.var(axis=0)
        if self.include_member_sigma and sigs:
            aleatoric = np.vstack([s ** 2 for s in sigs]).mean(axis=0)
            sigma = np.sqrt(epistemic + aleatoric)
        else:
            sigma = np.sqrt(epistemic)
        return mean, np.clip(sigma, 1e-6, None)
