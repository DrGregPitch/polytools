"""Dataset loading, validation and deduplication.

The bundled dataset is a 60-polymer smoke-test set, not a benchmark. It exists
so the test suite and the demo run offline in seconds. **Do not report results
on it.** See :func:`describe_real_datasets` for what to use instead.

The important function here is :func:`load_csv`, which enforces the hygiene
steps that decide whether a study is trustworthy: parse-check every structure,
canonicalize, deduplicate on canonical form, and aggregate duplicate
measurements rather than silently keeping whichever row came last. Duplicate
repeat units with conflicting property values are extremely common in polymer
data -- the same polymer appears in several source compilations with Tg values
that differ by tens of degrees -- and how you resolve them sets a floor on the
error any model can achieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import canonical_psmiles, is_valid_psmiles

__all__ = ["PolymerDataset", "load_csv", "load_toy_tg", "describe_real_datasets"]

_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class PolymerDataset:
    """A validated polymer dataset.

    Attributes
    ----------
    psmiles, y
        Aligned arrays of repeat-unit strings and target values.
    groups
        Optional grouping labels (chemical family, source, measurement year)
        used by :func:`polytools.splitters.group_split`.
    report
        Provenance record: how many rows were dropped and why. Print it in your
        README so reviewers can see the data was cleaned rather than assumed
        clean.
    """

    psmiles: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None
    names: np.ndarray | None = None
    target_name: str = "y"
    units: str = ""
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PolymerDataset(n={len(self)}, target={self.target_name!r} "
            f"[{self.units}], range=({self.y.min():.1f}, {self.y.max():.1f}))"
        )

    def to_frame(self) -> pd.DataFrame:
        data = {"psmiles": self.psmiles, self.target_name: self.y}
        if self.names is not None:
            data["name"] = self.names
        if self.groups is not None:
            data["group"] = self.groups
        return pd.DataFrame(data)


def load_csv(
    path: str | Path,
    psmiles_col: str = "psmiles",
    target_col: str = "y",
    group_col: str | None = None,
    name_col: str | None = None,
    dedup: str = "median",
    max_duplicate_spread: float | None = None,
    units: str = "",
) -> PolymerDataset:
    """Load and validate a polymer CSV.

    Parameters
    ----------
    dedup
        How to resolve repeat units that appear more than once after
        canonicalization: ``"median"`` (robust, recommended), ``"mean"``,
        ``"first"``, or ``"drop"`` to discard every conflicting entry.
    max_duplicate_spread
        If set, duplicate groups whose values span more than this are dropped
        entirely rather than averaged. Use it when a large disagreement between
        sources means at least one is wrong and averaging would invent a value
        that no one measured.
    """
    df = pd.read_csv(path)
    n_raw = len(df)

    for col in (psmiles_col, target_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in {list(df.columns)}")

    df = df.dropna(subset=[psmiles_col, target_col])
    n_after_na = len(df)

    valid_mask = df[psmiles_col].astype(str).map(is_valid_psmiles)
    dropped_invalid = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    df["_canonical"] = df[psmiles_col].astype(str).map(canonical_psmiles)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    n_before_dedup = len(df)
    spread = df.groupby("_canonical")[target_col].agg(lambda s: s.max() - s.min())
    n_duplicated = int((df.groupby("_canonical").size() > 1).sum())

    dropped_spread = 0
    if max_duplicate_spread is not None:
        bad = set(spread[spread > max_duplicate_spread].index)
        dropped_spread = int(df["_canonical"].isin(bad).sum())
        df = df[~df["_canonical"].isin(bad)]

    if dedup == "drop":
        counts = df.groupby("_canonical").size()
        df = df[df["_canonical"].map(counts) == 1]
    elif dedup in ("median", "mean", "first"):
        agg = {target_col: dedup if dedup != "first" else "first"}
        for col in (group_col, name_col, psmiles_col):
            if col:
                agg[col] = "first"
        df = df.groupby("_canonical", as_index=False).agg(agg)
    else:
        raise ValueError(f"unknown dedup strategy {dedup!r}")

    report = {
        "rows_in_file": n_raw,
        "dropped_missing": n_raw - n_after_na,
        "dropped_unparseable": dropped_invalid,
        "duplicate_groups": n_duplicated,
        "dropped_high_spread": dropped_spread,
        "rows_before_dedup": n_before_dedup,
        "rows_final": len(df),
        "dedup_strategy": dedup,
    }
    if n_duplicated:
        dup_spreads = spread[spread > 0]
        if len(dup_spreads):
            report["median_duplicate_spread"] = float(dup_spreads.median())
            report["max_duplicate_spread_seen"] = float(dup_spreads.max())

    return PolymerDataset(
        psmiles=df[psmiles_col].to_numpy(dtype=object),
        y=df[target_col].to_numpy(dtype=float),
        groups=df[group_col].to_numpy(dtype=object) if group_col else None,
        names=df[name_col].to_numpy(dtype=object) if name_col else None,
        target_name=target_col,
        units=units,
        report=report,
    )


def load_toy_tg() -> PolymerDataset:
    """Load the bundled 60-polymer glass-transition set.

    Approximate literature Tg values for common homopolymers, spanning -125 to
    228 C across eighteen chemical families. Assembled to exercise the code
    paths -- fluoropolymers, siloxanes, aromatic backbones, 1,1-disubstituted
    units -- not to benchmark anything. Values are rounded consensus figures and
    several (PTFE especially) are genuinely disputed in the literature.
    """
    return load_csv(
        _DATA_DIR / "homopolymer_tg_toy.csv",
        psmiles_col="psmiles",
        target_col="tg_c",
        group_col="family",
        name_col="name",
        units="degC",
    )


def describe_real_datasets() -> str:
    """Pointers to datasets worth actually reporting results on.

    Deliberately a docstring rather than a downloader: licences and hosting for
    these move, and a hard-coded URL that silently 404s is worse than none.
    Check the current terms yourself before publishing anything derived
    from them.
    """
    return """
Real polymer datasets to graduate to. Licences verified 2026 -- but hosting and
terms move, so re-check before you publish. `scripts/fetch_data.py` downloads the
permissively-licensed ones on demand.

PERMISSIVELY LICENSED (safe for a public portfolio, with attribution):

  RadonPy PI1070      1,077 homopolymers, MD-computed density, thermal
   [BSD-3-Clause]     conductivity, refractive index, heat capacity and more, with
                      repeat-unit pSMILES. `fetch_data.py radonpy`. Physically
                      grounded, computed not experimental -- label it as such.

  BCDB                5,387 diblock copolymers mined from the literature: block
   [MIT]              chemistries (BigSMILES), volume-fraction composition, and
                      observed phase morphology. The best open *copolymer*
                      structure-property set. github.com/olsenlabmit/BCDB.

  OPoly26             CC-BY-4.0 on HuggingFace. QM (DFT) properties on ~2,444
   [CC-BY-4.0]        monomers -- cluster/monomer level, not bulk polymer.

GATED OR RESTRICTIVELY LICENSED (results may be reportable, data is not
redistributable -- do not commit it):

  PolyInfo (NIMS)     The large curated experimental compilation (~500k points,
                      ~7k with Tg). Registration required; redistribution
                      restricted. The gold standard if you have access.

  polyVERSE / Khazana Ramprasad group. Many real experimental properties (gas
   [custom GTRC]      permeability, dielectric, melt viscosity, ...) but under a
                      restrictive Georgia Tech licence -- not open.

  polyOne             ~100M hypothetical polymers, 29 ML-*predicted* properties.
   [non-commercial]   Non-commercial licence, ~29 GB. It is model output; training
                      on it teaches you that model, not chemistry.

  PI1M                ~1M generated repeat units, unlabelled. For pretraining or
                      screening, not supervised training.

Practical advice: a few thousand clean labels with an honest split beats a million
computed ones with a random split. The bottleneck is label quality, not quantity.
""".strip()
