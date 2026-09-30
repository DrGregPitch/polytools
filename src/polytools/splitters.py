"""Dataset splitters, and a diagnostic for how hard a split actually is.

A random split on a polymer dataset is close to meaningless. Polymer datasets
are dense with near-duplicates -- the methacrylate series alone differs only in
side-chain length -- so a random test set is almost always within one methylene
of something in training. Models score well and then collapse on the first
genuinely new chemistry a lab sends them.

Every splitter here returns integer index arrays into the original dataset, so
they compose with any featurizer or model. The recommended reporting standard
for a portfolio project is to publish **random and at least one structured
split side by side**: the gap between them is the single most informative number
in the whole study, and quoting only the random number is the most common way
otherwise-good chemistry ML gets dismissed by reviewers.

:func:`split_difficulty` exists so the claim "this is an out-of-distribution
split" can be checked rather than asserted. It reports the distribution of
nearest-neighbour Tanimoto similarity from each test polymer to the training
set. A random split on a typical polymer dataset lands around 0.6-0.9; a good
structured split pushes the median well below that.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from .chem import Mode, prepare_mol

__all__ = [
    "Split",
    "random_split",
    "group_split",
    "scaffold_split",
    "cluster_split",
    "extrapolation_split",
    "split_difficulty",
    "SPLITTERS",
]


class Split(dict):
    """A train/val/test split: a dict of name -> index array, with a summary."""

    @property
    def train(self) -> np.ndarray:
        return self["train"]

    @property
    def val(self) -> np.ndarray:
        return self["val"]

    @property
    def test(self) -> np.ndarray:
        return self["test"]

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Split({self.sizes()})"


def _finalize(
    order_or_groups, n: int, frac_train: float, frac_val: float
) -> Split:
    """Turn an ordered index list into a Split honouring the requested fractions."""
    idx = np.asarray(order_or_groups, dtype=int)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    return Split(
        train=idx[:n_train],
        val=idx[n_train : n_train + n_val],
        test=idx[n_train + n_val :],
    )


def _check_fracs(frac_train: float, frac_val: float, frac_test: float) -> None:
    total = frac_train + frac_val + frac_test
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"fractions must sum to 1.0, got {total}")
    if min(frac_train, frac_val, frac_test) < 0:
        raise ValueError("fractions must be non-negative")


def random_split(
    n: int,
    frac_train: float = 0.7,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
    seed: int = 0,
) -> Split:
    """Uniformly random split. Include it as the optimistic reference point."""
    _check_fracs(frac_train, frac_val, frac_test)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return _finalize(order, n, frac_train, frac_val)


def group_split(
    groups: Sequence,
    frac_train: float = 0.7,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
    seed: int = 0,
) -> Split:
    """Split so that no group (e.g. chemical family) spans two partitions.

    This is the most interpretable structured split for polymers: hold out every
    fluoropolymer, or every polyester, and see whether the model generalizes to
    a chemistry it has never seen. Groups are assigned whole, largest first, to
    whichever partition is furthest below its quota.
    """
    _check_fracs(frac_train, frac_val, frac_test)
    n = len(groups)
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[g].append(i)

    rng = np.random.default_rng(seed)
    keys = list(buckets)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(buckets[k]), reverse=True)

    quotas = {"train": frac_train * n, "val": frac_val * n, "test": frac_test * n}
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for key in keys:
        deficits = {k: quotas[k] - len(assigned[k]) for k in assigned}
        target = max(deficits, key=deficits.get)
        assigned[target].extend(buckets[key])

    return Split(**{k: np.asarray(sorted(v), dtype=int) for k, v in assigned.items()})


def _scaffold(psmiles: str, mode: Mode) -> str:
    """Bemis-Murcko scaffold of the prepared molecule, as a SMILES string.

    Computing this on the raw pSMILES would be wrong: the dummy atoms are not
    valid ring or linker atoms, and RDKit's behaviour on them is not something
    to build an evaluation protocol on. Preparing first gives a real molecule.
    Acyclic repeat units (polyethylene, PDMS) have an empty Murcko scaffold;
    they are collected under a single ``"<acyclic>"`` key so they stay together
    rather than each forming a singleton group.
    """
    try:
        mol = prepare_mol(psmiles, mode=mode)
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaf)
        return smi if smi else "<acyclic>"
    except Exception:
        return "<invalid>"


def scaffold_split(
    psmiles_list: Sequence[str],
    frac_train: float = 0.7,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
    mode: Mode = "cap",
    seed: int = 0,
) -> Split:
    """Bemis-Murcko scaffold split, the standard structured split in molecular ML.

    Weaker than :func:`cluster_split` for polymers, because so many important
    repeat units are acyclic and collapse into one bucket. Report it for
    comparability with the wider literature, not as your hardest split.
    """
    scaffolds = [_scaffold(ps, mode) for ps in psmiles_list]
    return group_split(scaffolds, frac_train, frac_val, frac_test, seed=seed)


def cluster_split(
    psmiles_list: Sequence[str],
    frac_train: float = 0.7,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
    cutoff: float = 0.6,
    radius: int = 2,
    n_bits: int = 2048,
    mode: Mode = "oligomer",
    seed: int = 0,
) -> Split:
    """Butina-cluster the polymers by fingerprint similarity, then split by cluster.

    ``cutoff`` is a Tanimoto *distance* threshold: two polymers join the same
    cluster when their distance is below it. Lower cutoff means tighter, more
    numerous clusters and therefore an easier split. 0.6 is a reasonable
    starting point for polymers, which are more self-similar than drug-like
    molecules; tune it and report what you used.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprint(prepare_mol(ps, mode=mode)) for ps in psmiles_list]
    n = len(fps)

    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return group_split(labels, frac_train, frac_val, frac_test, seed=seed)


def extrapolation_split(
    y: Sequence[float],
    frac_train: float = 0.7,
    frac_val: float = 0.1,
    frac_test: float = 0.2,
    direction: str = "high",
) -> Split:
    """Train on one end of the property range, test on the other.

    This is the split that matters commercially and the one most papers avoid.
    Nobody needs a model to find another polymer with a Tg near the middle of
    the training distribution; they need one that finds a Tg *higher than
    anything known*. Expect the numbers to be bad. Report them anyway, and
    report the sign of the bias -- models trained on the low end almost
    universally under-predict the high end, and quantifying that shrinkage is
    genuinely useful to an experimentalist.

    ``direction="high"`` puts the highest values in test; ``"low"`` reverses it.
    """
    _check_fracs(frac_train, frac_val, frac_test)
    y = np.asarray(y, dtype=float)
    n = len(y)
    order = np.argsort(y)
    if direction == "high":
        pass
    elif direction == "low":
        order = order[::-1]
    else:
        raise ValueError("direction must be 'high' or 'low'")
    return _finalize(order, n, frac_train, frac_val)


def split_difficulty(
    psmiles_list: Sequence[str],
    split: Split,
    radius: int = 2,
    n_bits: int = 2048,
    mode: Mode = "oligomer",
) -> dict[str, float]:
    """Quantify how far the test set really is from the training set.

    Returns summary statistics of the nearest-neighbour Tanimoto similarity from
    each test polymer to its closest training polymer, plus the fraction of the
    test set that has a very close training analogue (>0.9), which is the number
    that exposes a leaky split.

    Put this table in your README. It converts "we used a scaffold split" from
    an unverifiable claim into a measurement.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprint(prepare_mol(ps, mode=mode)) for ps in psmiles_list]

    train_fps = [fps[i] for i in split["train"]]
    test_idx = split["test"]
    if not len(train_fps) or not len(test_idx):
        return {"n_test": 0.0}

    nn = []
    for i in test_idx:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        nn.append(max(sims) if sims else 0.0)
    nn = np.asarray(nn)

    return {
        "n_test": float(len(nn)),
        "nn_similarity_mean": float(nn.mean()),
        "nn_similarity_median": float(np.median(nn)),
        "nn_similarity_p90": float(np.percentile(nn, 90)),
        "frac_near_duplicate_gt_0.9": float((nn > 0.9).mean()),
        "frac_novel_lt_0.4": float((nn < 0.4).mean()),
    }


#: Registry so scripts and configs can select a splitter by name.
SPLITTERS = {
    "random": random_split,
    "group": group_split,
    "scaffold": scaffold_split,
    "cluster": cluster_split,
    "extrapolation": extrapolation_split,
}
