"""Polymer-aware RDKit helpers.

A polymer repeat unit is written as a *pSMILES* string: an ordinary SMILES with
exactly two dummy atoms (``[*]`` or ``*``) marking the two backbone connection
points. Polystyrene is ``[*]CC([*])c1ccccc1``.

Nothing in mainstream cheminformatics handles this correctly by default:

* Descriptors computed on the raw pSMILES include the dummy atoms, so MolWt,
  TPSA and every count-based descriptor are wrong.
* Bemis-Murcko scaffolds silently drop or mangle dummies.
* Fingerprints hash the dummy atom environment, which encodes *where the
  chemist chose to cut the repeat unit* rather than any property of the polymer.

This module fixes that by converting a repeat unit into a well-defined finite
molecule before any downstream cheminformatics happens. Three conventions are
supported, selected by ``mode``:

``cap``
    Replace each dummy with a capping group (default hydrogen). Fast, but the
    end groups perturb descriptors -- worst for short repeat units, where the
    caps are a large fraction of the molecule.

``cyclize``
    Bond the head anchor directly to the tail anchor, giving a macrocycle.
    Removes end-group artefacts entirely and is the closest single-unit
    approximation to an infinite chain. Preferred default for descriptors.
    Fails when head and tail anchors are the same atom.

``oligomer``
    Build an explicit n-mer and cap it. Most faithful for sequence-sensitive
    descriptors, and the only mode that lets a fingerprint see across the
    repeat-unit boundary. Slower, and grows the molecule n-fold.

References for the cyclization convention: it is the standard trick in the
polymer-informatics literature for removing chain-end bias from repeat-unit
descriptors. Verify against your own data before trusting it for a new property.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem  # noqa: F401  (registers some functionality)

Mode = Literal["cap", "cyclize", "oligomer", "raw"]

__all__ = [
    "PolymerParseError",
    "mol_from_psmiles",
    "connection_points",
    "cap_mol",
    "cyclize_mol",
    "cyclize_unit_count",
    "oligomerize",
    "backbone_length",
    "prepare_mol",
    "prepare_many",
    "canonical_psmiles",
    "is_valid_psmiles",
    "silence_rdkit",
]


class PolymerParseError(ValueError):
    """Raised when a pSMILES string is not a usable polymer repeat unit."""


def silence_rdkit() -> None:
    """Suppress RDKit's C++ logging. Call once at import time in scripts."""
    RDLogger.DisableLog("rdApp.*")


def mol_from_psmiles(psmiles: str, sanitize: bool = True) -> Chem.Mol:
    """Parse a pSMILES string into an RDKit Mol, keeping the dummy atoms.

    Raises
    ------
    PolymerParseError
        If the string does not parse, or does not carry exactly two dummy atoms.
    """
    if not isinstance(psmiles, str) or not psmiles.strip():
        raise PolymerParseError("empty pSMILES")

    mol = Chem.MolFromSmiles(psmiles, sanitize=sanitize)
    if mol is None:
        raise PolymerParseError(f"RDKit could not parse: {psmiles!r}")

    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise PolymerParseError(
            f"expected exactly 2 connection points, found {len(dummies)}: {psmiles!r}"
        )

    for idx in dummies:
        if mol.GetAtomWithIdx(idx).GetDegree() != 1:
            raise PolymerParseError(
                f"connection point {idx} is not terminal (degree != 1): {psmiles!r}"
            )
    return mol


def is_valid_psmiles(psmiles: str) -> bool:
    """Non-raising validity check. Useful for filtering a dataframe column."""
    try:
        mol_from_psmiles(psmiles)
        return True
    except PolymerParseError:
        return False


def connection_points(mol: Chem.Mol) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``((dummy_idx, anchor_idx), (dummy_idx, anchor_idx))``.

    The first pair is the *head*, the second the *tail*, in the order the dummy
    atoms appear in the molecule. The anchor is the heavy atom each dummy is
    bonded to.
    """
    pairs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                raise PolymerParseError("connection point is not terminal")
            pairs.append((atom.GetIdx(), neighbors[0].GetIdx()))
    if len(pairs) != 2:
        raise PolymerParseError(f"expected 2 connection points, found {len(pairs)}")
    return pairs[0], pairs[1]


def cap_mol(mol: Chem.Mol, cap: str = "H") -> Chem.Mol:
    """Replace both dummy atoms with a capping group.

    Parameters
    ----------
    cap
        ``"H"`` for hydrogen (implicit), or an element symbol such as ``"C"``
        to cap with a methyl group.
    """
    rw = Chem.RWMol(mol)
    dummies = sorted(
        (a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), reverse=True
    )
    if cap == "H":
        for idx in dummies:
            rw.RemoveAtom(idx)
    else:
        try:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(cap)
        except Exception as exc:  # pragma: no cover - defensive
            raise PolymerParseError(f"unknown capping element {cap!r}") from exc
        for idx in dummies:
            rw.GetAtomWithIdx(idx).SetAtomicNum(atomic_num)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as exc:
        raise PolymerParseError(f"capping produced an invalid molecule: {exc}") from exc
    return out


def backbone_length(mol: Chem.Mol) -> int:
    """Number of atoms on the shortest path from the head anchor to the tail anchor.

    For a vinyl polymer this is 2 (the two backbone carbons); for PET it is 10.
    Used to size the ring in :func:`cyclize_mol`.
    """
    (_, a_head), (_, a_tail) = connection_points(mol)
    if a_head == a_tail:
        return 1
    path = Chem.GetShortestPath(mol, a_head, a_tail)
    return len(path) if path else 1


def _assemble(mol: Chem.Mol, n: int) -> tuple[Chem.RWMol, int, int, int, int]:
    """Join ``n`` copies head-to-tail, leaving the two terminal dummies in place.

    Returns ``(rwmol, head_dummy, head_anchor, tail_dummy, tail_anchor)`` with
    indices valid in the returned molecule. Internal dummies are consumed.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    (d_head, a_head), (d_tail, a_tail) = connection_points(mol)
    size = mol.GetNumAtoms()

    combined = mol
    for _ in range(n - 1):
        combined = Chem.CombineMols(combined, mol)
    rw = Chem.RWMol(combined)

    consumed: list[int] = []
    for i in range(n - 1):
        rw.AddBond(a_tail + i * size, a_head + (i + 1) * size, Chem.BondType.SINGLE)
        consumed.append(d_tail + i * size)
        consumed.append(d_head + (i + 1) * size)

    term_head_d, term_head_a = d_head, a_head
    term_tail_d, term_tail_a = d_tail + (n - 1) * size, a_tail + (n - 1) * size

    # Removing atoms shifts indices, so track the survivors explicitly.
    survivors = [term_head_d, term_head_a, term_tail_d, term_tail_a]
    for idx in sorted(consumed, reverse=True):
        rw.RemoveAtom(idx)
        survivors = [s - 1 if s > idx else s for s in survivors]

    return rw, survivors[0], survivors[1], survivors[2], survivors[3]


def oligomerize(mol: Chem.Mol, n: int = 3, cap: str = "H") -> Chem.Mol:
    """Build a linear n-mer by joining ``n`` copies head-to-tail, then cap it.

    The tail anchor of copy *i* is bonded to the head anchor of copy *i+1*.
    Unlike a single capped repeat unit, an n-mer lets fingerprints see atom
    environments that span the repeat-unit boundary.
    """
    rw, d_head, _, d_tail, _ = _assemble(mol, n)

    if cap == "H":
        for idx in sorted([d_head, d_tail], reverse=True):
            rw.RemoveAtom(idx)
    else:
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(cap)
        for idx in (d_head, d_tail):
            rw.GetAtomWithIdx(idx).SetAtomicNum(atomic_num)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as exc:
        raise PolymerParseError(f"oligomerization failed: {exc}") from exc
    return out


def cyclize_unit_count(
    mol: Chem.Mol, min_ring_size: int = 10, max_repeat: int = 12
) -> int:
    """How many repeat units :func:`cyclize_mol` will assemble for this unit.

    Kept as the single source of truth so extensivity normalization elsewhere can
    never drift out of sync with the ring actually built.
    """
    per_unit = max(backbone_length(mol) - 1, 1)
    n = max(2, -(-min_ring_size // per_unit))  # ceiling division
    return min(n, max_repeat)


def cyclize_mol(
    mol: Chem.Mol, min_ring_size: int = 10, max_repeat: int = 12
) -> Chem.Mol:
    """Build a macrocycle from enough repeat units to reach ``min_ring_size``.

    Naively bonding the head anchor to the tail anchor of a *single* repeat unit
    does not work for the most common case in polymer science: in any vinyl
    polymer the two anchors are already bonded to each other, so the operation
    is either a no-op or creates a duplicate bond. Even when it succeeds it can
    produce an absurdly strained three- or four-membered ring whose fingerprint
    says more about the strain than about the polymer.

    So this function first assembles as many repeat units as needed for the
    backbone ring to reach ``min_ring_size`` atoms, then closes it. The result
    has no end groups at all, which is the point: descriptors reflect the repeat
    chemistry rather than the chemist's arbitrary choice of where to cut.

    A macrocycle is still not an infinite chain. Ring strain and
    transannular contacts are artefacts. Treat this as one convention among
    several and check it against ``mode="oligomer"`` on your own data.
    """
    n = cyclize_unit_count(mol, min_ring_size=min_ring_size, max_repeat=max_repeat)

    rw, d_head, a_head, d_tail, a_tail = _assemble(mol, n)

    if rw.GetBondBetweenAtoms(a_head, a_tail) is not None:
        raise PolymerParseError(
            "terminal anchors are already bonded; cannot cyclize this unit"
        )
    rw.AddBond(a_head, a_tail, Chem.BondType.SINGLE)
    for idx in sorted([d_head, d_tail], reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as exc:
        raise PolymerParseError(f"cyclization produced an invalid molecule: {exc}") from exc
    return out


def prepare_mol_counted(
    psmiles: str,
    mode: Mode = "oligomer",
    n_repeat: int = 3,
    cap: str = "H",
    min_ring_size: int = 10,
    fallback: bool = True,
) -> tuple[Chem.Mol, int]:
    """Like :func:`prepare_mol`, but also return how many repeat units were built.

    The count reflects the mode that **actually succeeded** -- with
    ``fallback=True`` that may differ from the requested mode, and dividing an
    extensive descriptor by the *requested* mode's count would silently be wrong
    for exactly the awkward units (fused-ring polyimides) that trigger fallback.
    Extensivity normalization must use this count, never re-derive it.
    """
    mol = mol_from_psmiles(psmiles)

    if mode == "raw":
        return mol, 1

    builders = {
        "cap": lambda: (cap_mol(mol, cap=cap), 1),
        "oligomer": lambda: (oligomerize(mol, n=n_repeat, cap=cap), n_repeat),
        "cyclize": lambda: (
            cyclize_mol(mol, min_ring_size=min_ring_size),
            cyclize_unit_count(mol, min_ring_size=min_ring_size),
        ),
    }
    if mode not in builders:
        raise ValueError(f"unknown mode {mode!r}")

    order = [mode]
    if fallback:
        order += [m for m in ("cyclize", "oligomer", "cap") if m != mode]

    last_exc: Exception | None = None
    for m in order:
        try:
            return builders[m]()
        except PolymerParseError as exc:
            last_exc = exc
    raise last_exc  # every mode failed; surface the last error


def prepare_mol(
    psmiles: str,
    mode: Mode = "oligomer",
    n_repeat: int = 3,
    cap: str = "H",
    min_ring_size: int = 10,
    fallback: bool = True,
) -> Chem.Mol:
    """Convert a pSMILES into a finite molecule ready for cheminformatics.

    This is the single entry point the rest of the library uses. Every
    featurizer and splitter routes through it, so the repeat-unit convention is
    defined in exactly one place.

    Parameters
    ----------
    mode
        ``"oligomer"`` (default), ``"cap"``, ``"cyclize"``, or ``"raw"`` to keep
        the dummy atoms untouched. ``cyclize`` is the preferred choice for
        descriptor work (no end-group artefacts); ``oligomer`` for fingerprints.
    fallback
        If True and the requested mode fails on a structurally awkward unit, try
        the other modes in the order ``cyclize -> oligomer -> cap`` and return the
        first that succeeds. Set False when you want every failure to be loud.

        This matters on real data: some perfectly valid repeat units -- fused-ring
        polyimides especially -- fail to kekulize when capped or oligomerised but
        cyclise cleanly, and vice versa. The fallback keeps them in the dataset
        instead of silently dropping a whole chemical class. ``cyclize`` leads the
        fallback order because it is empirically the most robust across awkward
        aromatics.
    """
    return prepare_mol_counted(
        psmiles, mode=mode, n_repeat=n_repeat, cap=cap,
        min_ring_size=min_ring_size, fallback=fallback,
    )[0]


def canonical_psmiles(psmiles: str) -> str:
    """Canonicalize a pSMILES, normalizing dummy-atom notation to ``[*]``.

    Two spellings of the same repeat unit -- ``*CC*`` and ``[*]CC[*]`` -- map to
    the same string, which is what you need before deduplicating a dataset.
    Note this does **not** resolve the deeper ambiguity that the same polymer
    can be written with different repeat-unit choices.
    """
    mol = mol_from_psmiles(psmiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_many(
    psmiles_list: Sequence[str],
    mode: Mode = "oligomer",
    n_repeat: int = 3,
    cap: str = "H",
    skip_invalid: bool = False,
) -> tuple[list[Chem.Mol], list[int]]:
    """Vectorized :func:`prepare_mol`.

    Returns ``(mols, kept_indices)``. When ``skip_invalid`` is True, unparseable
    entries are dropped and their positions omitted from ``kept_indices``,
    letting the caller realign the target array.
    """
    mols: list[Chem.Mol] = []
    kept: list[int] = []
    for i, ps in enumerate(psmiles_list):
        try:
            mols.append(prepare_mol(ps, mode=mode, n_repeat=n_repeat, cap=cap))
            kept.append(i)
        except (PolymerParseError, ValueError):
            if skip_invalid:
                continue
            raise
    return mols, kept
