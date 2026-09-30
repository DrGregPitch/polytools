"""Graph representation of polymer repeat units, with periodic backbone edges.

Pure NumPy and RDKit -- no deep learning framework required. The arrays here
feed directly into PyTorch Geometric, DGL, JAX or a hand-rolled message-passing
layer, which is why this module has no torch dependency: the polymer-specific
reasoning belongs in the representation, not in the model code.

The central design decision
--------------------------
The obvious way to graph a repeat unit is to include the two dummy atoms as
nodes. This is wrong, and wrong in a way that quietly degrades every model built
on it. The dummy atoms are not chemical entities; they are notation marking
where the chemist chose to cut an infinite chain. Feeding them to a GNN teaches
it about that arbitrary choice, and two different-but-equivalent ways of writing
the same polymer produce different graphs.

The alternative used here:

1. **Drop the dummy atoms as nodes.** They carry no chemistry.
2. **Mark their anchors with a feature flag.** The model still learns which
   atoms sit on the backbone connection points, which is real information.
3. **Add a periodic edge** joining the tail anchor back to the head anchor.

Step 3 is what makes it a polymer graph rather than a molecule graph. That edge
encodes the fact that the chain continues -- the head anchor's true chemical
environment includes the tail anchor of the neighbouring unit, and they are the
same atom types by construction. Without it, a GNN sees the repeat unit's ends
as chain termini and learns end-group chemistry that does not exist in the
material. With it, message passing wraps around, and after k layers every atom
has aggregated information from k bonds away *along the real chain*.

The periodic edge is flagged in ``edge_is_periodic`` so a model can embed it
with its own learned edge type, or ablate it. That ablation -- periodic edge on
versus off -- is a clean, publishable experiment and a good first result for the
representation project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .chem import PolymerParseError, connection_points, mol_from_psmiles

__all__ = [
    "PolymerGraph",
    "ATOM_FEATURE_NAMES",
    "BOND_FEATURE_NAMES",
    "atom_features",
    "bond_features",
    "psmiles_to_graph",
    "batch_graphs",
]

# Elements covering essentially all synthetic polymers. Anything else lands in
# the "other" slot rather than crashing, but if you see that slot firing often
# on your dataset, extend this list rather than ignoring it.
_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P", "B"]

_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

ATOM_FEATURE_NAMES: list[str] = (
    [f"element_{e}" for e in _ELEMENTS]
    + ["element_other"]
    + [f"degree_{d}" for d in range(5)]
    + [f"num_h_{h}" for h in range(5)]
    + [f"hybrid_{h.name}" for h in _HYBRIDIZATIONS]
    + ["hybrid_other"]
    + [
        "formal_charge",
        "is_aromatic",
        "is_in_ring",
        "is_backbone_anchor",  # polymer-specific
        "mass_scaled",
    ]
)

BOND_FEATURE_NAMES: list[str] = (
    [f"bond_{b.name}" for b in _BOND_TYPES]
    + ["bond_other", "is_conjugated", "is_in_ring", "is_periodic"]
)


def _one_hot(value, choices) -> list[float]:
    """One-hot with a trailing 'other' slot, so unseen values never crash."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except (ValueError, IndexError):
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom, is_anchor: bool = False) -> np.ndarray:
    """Feature vector for one atom.

    ``is_anchor`` marks an atom bonded to a connection point. It is the only
    polymer-specific feature and it matters: backbone attachment atoms behave
    differently from pendant ones, and without the flag the model cannot tell
    which part of the unit continues the chain.
    """
    feats: list[float] = []
    feats += _one_hot(atom.GetSymbol(), _ELEMENTS)
    feats += _one_hot(min(atom.GetDegree(), 4), list(range(4)))
    feats += _one_hot(min(atom.GetTotalNumHs(), 4), list(range(4)))
    feats += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
    feats += [
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(is_anchor),
        atom.GetMass() / 100.0,  # scaled so it does not dominate unnormalized input
    ]
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond | None, is_periodic: bool = False) -> np.ndarray:
    """Feature vector for one bond.

    ``bond`` may be None for a synthetic periodic edge, which has no underlying
    RDKit bond. Periodic edges are typed as single bonds and flagged, letting a
    model learn a distinct embedding for them.
    """
    if bond is None:
        feats = _one_hot(Chem.BondType.SINGLE, _BOND_TYPES) + [0.0, 0.0]
    else:
        feats = _one_hot(bond.GetBondType(), _BOND_TYPES) + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
    feats.append(float(is_periodic))
    return np.asarray(feats, dtype=np.float32)


@dataclass
class PolymerGraph:
    """A repeat unit as a graph.

    Attributes
    ----------
    node_features
        ``(n_nodes, n_atom_features)``.
    edge_index
        ``(2, n_edges)`` in COO format, with both directions present.
    edge_features
        ``(n_edges, n_bond_features)``.
    edge_is_periodic
        Boolean mask over edges, True for the wrap-around backbone edges. Use it
        to ablate periodicity or to give those edges a separate embedding.
    anchor_indices
        Node indices of the two backbone connection atoms.
    psmiles
        The source string, kept for traceability when debugging a bad prediction.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_is_periodic: np.ndarray
    anchor_indices: tuple[int, int]
    psmiles: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PolymerGraph(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"periodic={int(self.edge_is_periodic.sum())}, psmiles={self.psmiles!r})"
        )


def psmiles_to_graph(psmiles: str, periodic: bool = True) -> PolymerGraph:
    """Convert a repeat unit into a graph with optional periodic backbone edges.

    Parameters
    ----------
    periodic
        When True (default), add edges wrapping the tail anchor back to the head
        anchor so message passing continues along the chain. Set False to build
        the ablation baseline -- the same graph treated as a finite molecule.

    Notes
    -----
    If both connection points attach to the *same* atom (a 1,1-disubstituted
    unit written unusually), the periodic edge would be a self-loop and is
    skipped, with the anchor flag still set. The graph stays valid.
    """
    mol = mol_from_psmiles(psmiles)
    (d_head, a_head), (d_tail, a_tail) = connection_points(mol)
    dummies = {d_head, d_tail}

    # Map original atom indices to graph node indices, skipping dummy atoms.
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in dummies]
    remap = {old: new for new, old in enumerate(keep)}
    if not keep:
        raise PolymerParseError(f"repeat unit has no heavy atoms: {psmiles!r}")

    anchors = {a_head, a_tail}
    nodes = np.vstack(
        [
            atom_features(mol.GetAtomWithIdx(idx), is_anchor=idx in anchors)
            for idx in keep
        ]
    )

    src: list[int] = []
    dst: list[int] = []
    efeat: list[np.ndarray] = []
    eper: list[bool] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in dummies or j in dummies:
            continue  # bonds to connection points are notation, not chemistry
        bi, bj = remap[i], remap[j]
        f = bond_features(bond)
        src += [bi, bj]
        dst += [bj, bi]
        efeat += [f, f]
        eper += [False, False]

    if periodic and a_head != a_tail:
        f = bond_features(None, is_periodic=True)
        hi, ti = remap[a_head], remap[a_tail]
        src += [ti, hi]
        dst += [hi, ti]
        efeat += [f, f]
        eper += [True, True]

    n_bond_feats = len(BOND_FEATURE_NAMES)
    edge_index = (
        np.asarray([src, dst], dtype=np.int64)
        if src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_features = (
        np.vstack(efeat) if efeat else np.zeros((0, n_bond_feats), dtype=np.float32)
    )

    return PolymerGraph(
        node_features=nodes,
        edge_index=edge_index,
        edge_features=edge_features.astype(np.float32),
        edge_is_periodic=np.asarray(eper, dtype=bool),
        anchor_indices=(remap[a_head], remap[a_tail]),
        psmiles=psmiles,
    )


def batch_graphs(graphs: list[PolymerGraph]) -> dict[str, np.ndarray]:
    """Collate graphs into one block-diagonal batch.

    Returns a dict with ``node_features``, ``edge_index``, ``edge_features``,
    ``edge_is_periodic`` and ``batch`` -- the last mapping each node to its
    source graph, which is what a scatter-based readout needs to pool per
    molecule. This is the standard collation any GNN framework expects, so
    wrapping it in ``torch.from_numpy`` is the whole integration step.
    """
    if not graphs:
        raise ValueError("no graphs to batch")

    node_blocks, edge_blocks, efeat_blocks, per_blocks, batch_ids = [], [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        node_blocks.append(g.node_features)
        edge_blocks.append(g.edge_index + offset)
        efeat_blocks.append(g.edge_features)
        per_blocks.append(g.edge_is_periodic)
        batch_ids.append(np.full(g.n_nodes, gi, dtype=np.int64))
        offset += g.n_nodes

    return {
        "node_features": np.vstack(node_blocks).astype(np.float32),
        "edge_index": np.hstack(edge_blocks).astype(np.int64),
        "edge_features": np.vstack(efeat_blocks).astype(np.float32),
        "edge_is_periodic": np.concatenate(per_blocks),
        "batch": np.concatenate(batch_ids),
        "n_graphs": np.int64(len(graphs)),
    }
