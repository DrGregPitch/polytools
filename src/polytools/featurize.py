"""Featurizers for polymer repeat units and copolymer compositions.

Every featurizer takes a list of pSMILES strings and returns a dense ``float64``
array of shape ``(n_samples, n_features)``, plus a ``feature_names`` list of the
same width so that model explanations stay interpretable.

The polymer-specific concern is *extensivity*. Descriptors like molecular weight
and heavy-atom count scale with however many repeat units the preparation mode
happened to build, so a 3-mer and a 12-membered macrocycle of the same polymer
get wildly different values for reasons that have nothing to do with the
material. Anything extensive is therefore divided by the number of repeat units
by default (``normalize_extensive=True``), which turns "molecular weight of the
oligomer I happened to construct" into "molecular weight per repeat unit" -- a
real, reportable property of the polymer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from .chem import Mode, prepare_mol_counted

__all__ = [
    "Featurizer",
    "MorganFeaturizer",
    "DescriptorFeaturizer",
    "ConcatFeaturizer",
    "CopolymerFeaturizer",
    "featurize",
]


# Descriptors chosen for polymer property work rather than drug discovery.
# The bool marks whether the descriptor is extensive (scales with molecule size)
# and therefore needs normalizing by repeat-unit count.
_DESCRIPTORS: list[tuple[str, callable, bool]] = [
    ("MolWt", Descriptors.MolWt, True),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount, True),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds, True),
    ("NumHDonors", Descriptors.NumHDonors, True),
    ("NumHAcceptors", Descriptors.NumHAcceptors, True),
    ("TPSA", Descriptors.TPSA, True),
    ("LabuteASA", Descriptors.LabuteASA, True),
    ("MolLogP", Descriptors.MolLogP, True),
    ("MolMR", Descriptors.MolMR, True),
    ("RingCount", Descriptors.RingCount, True),
    ("NumAromaticRings", Descriptors.NumAromaticRings, True),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings, True),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms, True),
    ("BertzCT", Descriptors.BertzCT, True),
    ("Chi0v", Descriptors.Chi0v, True),
    ("Chi1v", Descriptors.Chi1v, True),
    ("Kappa1", Descriptors.Kappa1, True),
    ("Kappa2", Descriptors.Kappa2, True),
    # Intensive: ratios and averages, safe to leave alone.
    ("FractionCSP3", Descriptors.FractionCSP3, False),
    ("HallKierAlpha", Descriptors.HallKierAlpha, False),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge, False),
    ("MinPartialCharge", Descriptors.MinPartialCharge, False),
]


class Featurizer:
    """Base class. Subclasses implement :meth:`_transform_mols`."""

    #: Preparation mode applied to each pSMILES before featurizing.
    def __init__(
        self,
        mode: Mode = "oligomer",
        n_repeat: int = 3,
        cap: str = "H",
        min_ring_size: int = 10,
    ) -> None:
        self.mode = mode
        self.n_repeat = n_repeat
        self.cap = cap
        self.min_ring_size = min_ring_size
        self.feature_names: list[str] = []

    # -- internals -------------------------------------------------------
    def _prepare_counted(self, psmiles: str) -> tuple[Chem.Mol, int]:
        """Prepared molecule plus the repeat-unit count that was ACTUALLY built.

        The count must come from the preparation itself: with fallback enabled,
        ``prepare_mol`` can succeed via a different mode than requested (e.g. a
        fused-ring polyimide that cannot be oligomerised gets cyclised instead),
        and re-deriving the count from the requested mode would divide extensive
        descriptors by the wrong number for exactly those units.
        """
        return prepare_mol_counted(
            psmiles,
            mode=self.mode,
            n_repeat=self.n_repeat,
            cap=self.cap,
            min_ring_size=self.min_ring_size,
        )

    def _transform_mols(self, mols: Sequence[Chem.Mol], units: Sequence[int]):
        raise NotImplementedError

    # -- public API ------------------------------------------------------
    def transform(self, psmiles_list: Sequence[str]) -> np.ndarray:
        mols, units = [], []
        for ps in psmiles_list:
            mol, n_units = self._prepare_counted(ps)
            mols.append(mol)
            units.append(n_units)
        return self._transform_mols(mols, units)

    def fit(self, psmiles_list: Sequence[str], y=None) -> Featurizer:
        """Present for scikit-learn compatibility; these featurizers are stateless."""
        return self

    def fit_transform(self, psmiles_list: Sequence[str], y=None) -> np.ndarray:
        return self.fit(psmiles_list, y).transform(psmiles_list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(mode={self.mode!r}, n_repeat={self.n_repeat})"


class MorganFeaturizer(Featurizer):
    """ECFP-style circular fingerprints (Morgan) over the prepared molecule.

    Counts (``use_counts=True``) usually beat bits for regression on small
    polymer datasets, because a repeat unit with three ester groups really is
    different from one with a single ester -- information a binary vector
    throws away.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_counts: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self.feature_names = [f"ecfp{2 * radius}_{i}" for i in range(n_bits)]

    def _transform_mols(self, mols, units) -> np.ndarray:
        out = np.zeros((len(mols), self.n_bits), dtype=np.float64)
        for i, mol in enumerate(mols):
            if self.use_counts:
                fp = self._gen.GetCountFingerprintAsNumPy(mol)
            else:
                fp = self._gen.GetFingerprintAsNumPy(mol)
            out[i] = fp
        return out


class DescriptorFeaturizer(Featurizer):
    """Curated RDKit descriptors with extensive quantities normalized per repeat unit."""

    def __init__(self, normalize_extensive: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.normalize_extensive = normalize_extensive
        suffix = "_per_unit" if normalize_extensive else ""
        self.feature_names = [
            f"{name}{suffix if extensive else ''}"
            for name, _, extensive in _DESCRIPTORS
        ]

    def _transform_mols(self, mols, units) -> np.ndarray:
        out = np.zeros((len(mols), len(_DESCRIPTORS)), dtype=np.float64)
        for i, (mol, n_units) in enumerate(zip(mols, units)):
            for j, (_, fn, extensive) in enumerate(_DESCRIPTORS):
                try:
                    value = float(fn(mol))
                except Exception:
                    value = np.nan
                if not np.isfinite(value):
                    value = np.nan
                if extensive and self.normalize_extensive and n_units > 0:
                    value = value / n_units
                out[i, j] = value
        # A few descriptors (partial charges) legitimately fail on some elements;
        # impute those NaNs with 0.0. The constant is deliberate: a batch-median
        # impute would make a molecule's features depend on which other molecules
        # it happened to be featurized with -- breaking the stateless contract,
        # letting test rows influence train-row values in whole-dataset
        # featurization, and shifting single-molecule predictions (e.g. the demo)
        # relative to training. Deterministic beats subtle.
        out[~np.isfinite(out)] = 0.0
        return out


class ConcatFeaturizer(Featurizer):
    """Horizontally stack several featurizers.

    Each sub-featurizer keeps its own preparation mode, which is deliberate:
    descriptors are often best on a cyclized unit while fingerprints are best on
    an oligomer that can see across the repeat boundary.
    """

    def __init__(self, featurizers: Sequence[Featurizer]) -> None:
        if not featurizers:
            raise ValueError("need at least one featurizer")
        self.featurizers = list(featurizers)
        self.feature_names = [
            n for f in self.featurizers for n in f.feature_names
        ]

    def transform(self, psmiles_list: Sequence[str]) -> np.ndarray:
        return np.hstack([f.transform(psmiles_list) for f in self.featurizers])

    def fit(self, psmiles_list: Sequence[str], y=None) -> ConcatFeaturizer:
        for f in self.featurizers:
            f.fit(psmiles_list, y)
        return self

    def fit_transform(self, psmiles_list: Sequence[str], y=None) -> np.ndarray:
        return self.fit(psmiles_list, y).transform(psmiles_list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ConcatFeaturizer({self.featurizers!r})"


class CopolymerFeaturizer:
    """Encode a copolymer as a composition-weighted combination of its comonomers.

    Input is a list of ``[(psmiles, mole_fraction), ...]`` records. This is the
    naive baseline that most published copolymer work uses, and it is included
    here precisely so that later work has something honest to beat: a linear
    mixing rule cannot express sequence effects, so by construction it predicts
    the same value for a block and a random copolymer of identical composition.
    That is a real, testable failure mode -- Project 2 in the roadmap is about
    measuring how badly it fails and on which properties.

    ``include_moments=True`` appends the composition entropy and the maximum
    mole fraction, which at least lets a model distinguish a homopolymer from a
    50/50 blend even when their weighted features coincide.
    """

    def __init__(
        self,
        base: Featurizer | None = None,
        include_moments: bool = True,
        normalize: bool = True,
    ) -> None:
        self.base = base if base is not None else DescriptorFeaturizer()
        self.include_moments = include_moments
        self.normalize = normalize
        self.feature_names = list(self.base.feature_names)
        if include_moments:
            self.feature_names += ["comp_entropy", "comp_max_fraction", "n_comonomers"]

    def transform(
        self, compositions: Sequence[Sequence[tuple[str, float]]]
    ) -> np.ndarray:
        rows = []
        for comp in compositions:
            if not comp:
                raise ValueError("empty copolymer composition")
            smis = [c[0] for c in comp]
            fracs = np.asarray([float(c[1]) for c in comp], dtype=np.float64)
            if np.any(fracs < 0):
                raise ValueError("mole fractions must be non-negative")
            total = fracs.sum()
            if total <= 0:
                raise ValueError("mole fractions sum to zero")
            if self.normalize:
                fracs = fracs / total
            elif not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"mole fractions sum to {total:.4f}, not 1.0 "
                    "(pass normalize=True to rescale)"
                )

            feats = self.base.transform(smis)
            weighted = fracs @ feats

            if self.include_moments:
                nonzero = fracs[fracs > 0]
                entropy = float(-(nonzero * np.log(nonzero)).sum())
                weighted = np.concatenate(
                    [weighted, [entropy, float(fracs.max()), float(len(fracs))]]
                )
            rows.append(weighted)
        return np.vstack(rows)

    def fit(self, compositions, y=None) -> CopolymerFeaturizer:
        return self

    def fit_transform(self, compositions, y=None) -> np.ndarray:
        return self.transform(compositions)


def featurize(
    psmiles_list: Sequence[str],
    kind: str = "ecfp+desc",
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """Convenience wrapper returning ``(X, feature_names)``.

    ``kind`` is one of ``"ecfp"``, ``"desc"``, or ``"ecfp+desc"``.
    """
    if kind == "ecfp":
        f: Featurizer = MorganFeaturizer(**kwargs)
    elif kind == "desc":
        f = DescriptorFeaturizer(**kwargs)
    elif kind == "ecfp+desc":
        f = ConcatFeaturizer(
            [MorganFeaturizer(**kwargs), DescriptorFeaturizer(mode="cyclize")]
        )
    else:
        raise ValueError(f"unknown featurizer kind {kind!r}")
    return f.transform(psmiles_list), f.feature_names
