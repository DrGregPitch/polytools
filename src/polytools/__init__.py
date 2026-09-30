"""polytools -- honest machine learning infrastructure for polymer property prediction.

Shared foundation for a polymer-ML portfolio: polymer-aware featurization,
structured splitters that do not flatter the model, calibrated uncertainty, and
the four figures a results README needs.

Quick start
-----------
>>> from polytools import load_toy_tg, featurize, cluster_split, GBMRegressor
>>> from polytools import regression_metrics, split_difficulty
>>> ds = load_toy_tg()
>>> X, names = featurize(ds.psmiles, "ecfp+desc")
>>> split = cluster_split(ds.psmiles)
>>> model = GBMRegressor().fit(X[split.train], ds.y[split.train])
>>> metrics = regression_metrics(ds.y[split.test], model.predict(X[split.test]))

The bundled dataset is a 60-polymer smoke test, not a benchmark. See
``polytools.data.describe_real_datasets()`` for what to use for real results.
"""

from .analysis import (
    group_bias_table,
    oof_predictions,
    worst_predictions,
)
from .chem import (
    PolymerParseError,
    backbone_length,
    canonical_psmiles,
    cap_mol,
    cyclize_mol,
    cyclize_unit_count,
    is_valid_psmiles,
    mol_from_psmiles,
    oligomerize,
    prepare_many,
    prepare_mol,
    prepare_mol_counted,
    silence_rdkit,
)
from .data import (
    PolymerDataset,
    describe_real_datasets,
    load_csv,
    load_toy_tg,
)
from .featurize import (
    ConcatFeaturizer,
    CopolymerFeaturizer,
    DescriptorFeaturizer,
    Featurizer,
    MorganFeaturizer,
    featurize,
)
from .graphs import (
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    PolymerGraph,
    atom_features,
    batch_graphs,
    bond_features,
    psmiles_to_graph,
)
from .metrics import (
    SigmaRecalibrator,
    calibration_curve,
    error_uncertainty_correlation,
    format_metrics,
    gaussian_nll,
    miscalibration_area,
    regression_metrics,
    selective_prediction_curve,
    uncertainty_metrics,
)
from .models import (
    BaseRegressor,
    EnsembleRegressor,
    GBMRegressor,
    MeanRegressor,
    RidgeDescriptorRegressor,
)
from .splitters import (
    SPLITTERS,
    Split,
    cluster_split,
    extrapolation_split,
    group_split,
    random_split,
    scaffold_split,
    split_difficulty,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # analysis
    "group_bias_table", "oof_predictions", "worst_predictions",
    # chem
    "PolymerParseError", "backbone_length", "canonical_psmiles", "cap_mol",
    "cyclize_mol", "cyclize_unit_count", "is_valid_psmiles", "mol_from_psmiles",
    "oligomerize", "prepare_many", "prepare_mol", "prepare_mol_counted",
    "silence_rdkit",
    # data
    "PolymerDataset", "describe_real_datasets", "load_csv", "load_toy_tg",
    # featurize
    "ConcatFeaturizer", "CopolymerFeaturizer", "DescriptorFeaturizer",
    "Featurizer", "MorganFeaturizer", "featurize",
    # graphs
    "ATOM_FEATURE_NAMES", "BOND_FEATURE_NAMES", "PolymerGraph", "atom_features",
    "batch_graphs", "bond_features", "psmiles_to_graph",
    # metrics
    "SigmaRecalibrator", "calibration_curve", "error_uncertainty_correlation",
    "format_metrics", "gaussian_nll", "miscalibration_area",
    "regression_metrics", "selective_prediction_curve", "uncertainty_metrics",
    # models
    "BaseRegressor", "EnsembleRegressor", "GBMRegressor", "MeanRegressor",
    "RidgeDescriptorRegressor",
    # splitters
    "SPLITTERS", "Split", "cluster_split", "extrapolation_split", "group_split",
    "random_split", "scaffold_split", "split_difficulty",
]
