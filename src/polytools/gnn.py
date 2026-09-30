"""A directed message-passing neural network (D-MPNN) for polymer repeat units.

This is the neural rung of the model ladder. It consumes the polymer graphs from
:mod:`polytools.graphs` -- dummy atoms dropped, backbone anchors flagged, periodic
edge wrapping the chain -- and learns a property directly from structure instead of
from a hand-designed fingerprint.

The architecture is the *directed* MPNN of Yang et al. 2019 (the model inside
chemprop): messages live on directed bonds rather than atoms, and a bond's message
excludes the reverse bond, which suppresses the noisy "tottering" that plain
atom-based message passing suffers. It is a published, respected design -- the
right thing to reproduce for a portfolio rather than invent.

Two deliberate choices make this fit the rest of the library:

* **Same interface as every other model.** ``fit(X, y)`` and
  ``predict(X, return_std=False)``, where ``X`` is an object array of
  :class:`~polytools.graphs.PolymerGraph`. So every split, metric, calibration
  step and figure in the benchmark keeps working unchanged -- swapping the GNN in
  is two methods, not a rewrite.
* **Torch is optional.** It is imported lazily and guarded by ``HAS_TORCH``, the
  same pattern :mod:`polytools.models` uses for LightGBM, so the core library
  installs and the base test suite runs without a deep-learning framework.

Uncertainty comes from **MC-dropout**: keep dropout active at inference, sample a
handful of forward passes, and take the mean and standard deviation. It is the
cheapest honest uncertainty for a single neural net, and it plugs straight into the
same :class:`~polytools.metrics.SigmaRecalibrator` as the ensembles.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .graphs import (
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    PolymerGraph,
    batch_graphs,
    psmiles_to_graph,
)
from .models import BaseRegressor

__all__ = [
    "HAS_TORCH",
    "graphs_from_psmiles",
    "GNNRegressor",
]

# macOS ships one libomp with LightGBM and another with PyTorch; loading both in
# one process aborts with the notorious "multiple OpenMP runtimes" crash. This is
# the documented workaround, and it must be set *before* torch is imported. It is
# a setdefault, so a user who has already made a deliberate choice wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn

    HAS_TORCH = True
    # Some MPS kernels are not implemented; fall back to CPU for those ops rather
    # than crashing mid-train. Harmless on CPU-only machines.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # LightGBM and PyTorch each bundle their own OpenMP runtime. With both loaded
    # in one process (any run that touches GBMRegressor *and* the GNN), their
    # thread pools collide and segfault the interpreter mid-training. Pinning
    # torch to a single CPU thread removes the contention entirely. The graphs
    # here are tiny repeat units, so this costs no meaningful wall-clock. A user
    # who has already raised the thread count keeps their choice.
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def graphs_from_psmiles(
    psmiles_list: Sequence[str], periodic: bool = True
) -> np.ndarray:
    """Build an object array of :class:`PolymerGraph`, one per repeat unit.

    Returned as a NumPy object array so it indexes exactly like a feature matrix:
    ``graphs[split.train]`` gives the training graphs, which is what lets the GNN
    drop into the existing benchmark harness untouched.

    ``periodic=False`` builds the ablation representation -- the same graphs with
    the chain-wrapping backbone edge removed. Training the model on ``periodic=True``
    vs ``False`` and comparing is the clean first experiment of Project 1.
    """
    graphs = [psmiles_to_graph(ps, periodic=periodic) for ps in psmiles_list]
    arr = np.empty(len(graphs), dtype=object)
    arr[:] = graphs
    return arr


def _reverse_edge_index(edge_index: np.ndarray) -> np.ndarray:
    """For each directed edge, the index of its reverse (v->u for u->v).

    The D-MPNN message for a bond must exclude the bond pointing back the way it
    came, so it needs to find that reverse bond.

    We rely on a construction invariant rather than matching endpoints:
    ``psmiles_to_graph`` always appends the two directions of a bond consecutively
    (edge ``2k`` and ``2k+1`` are reverses), and ``batch_graphs`` concatenates
    per-graph edge blocks in order, so that pairing survives batching. Reverse is
    therefore just ``e XOR 1``.

    Endpoint-matching would be wrong here: a vinyl repeat unit like polyethylene
    ``[*]CC[*]`` has its two backbone carbons directly bonded *and* joined by the
    periodic edge, so the real bond and the periodic edge are parallel edges with
    identical endpoints. Only the construction pairing distinguishes them.
    """
    e = edge_index.shape[1]
    if e % 2 != 0:  # pragma: no cover - our builder never emits an odd edge count
        raise ValueError(f"expected an even number of directed edges, got {e}")
    rev = np.arange(e, dtype=np.int64)
    rev[0::2] += 1
    rev[1::2] -= 1
    # cheap invariant check: paired edges really are reverses of each other
    src, dst = edge_index[0], edge_index[1]
    if not (np.array_equal(src[rev], dst) and np.array_equal(dst[rev], src)):
        raise ValueError(
            "edge ordering broke the reverse-pair invariant; "
            "graphs must come from psmiles_to_graph/batch_graphs"
        )
    return rev


if HAS_TORCH:

    class _DMPNN(nn.Module):
        """Directed message-passing network with a small feed-forward read-out head."""

        def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden: int = 128,
            depth: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.depth = depth
            # bond message initialisation from [source atom features, bond features]
            self.W_i = nn.Linear(node_dim + edge_dim, hidden, bias=False)
            self.W_h = nn.Linear(hidden, hidden, bias=False)
            # atom read-out from [atom features, aggregated incoming bond messages]
            self.W_o = nn.Linear(node_dim + hidden, hidden)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ReLU()
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(
            self,
            x: torch.Tensor,        # (N, node_dim)
            edge_index: torch.Tensor,  # (2, E) long
            edge_attr: torch.Tensor,   # (E, edge_dim)
            rev: torch.Tensor,         # (E,) long
            batch: torch.Tensor,       # (N,) long
            n_graphs: int,
        ) -> torch.Tensor:
            src, dst = edge_index[0], edge_index[1]
            n_nodes = x.shape[0]

            # h0_e = ReLU(W_i [x_src(e), e])
            h0 = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=1)))
            h = h0

            for _ in range(self.depth - 1):
                # sum of messages arriving at each node, over all incoming edges
                node_in = x.new_zeros((n_nodes, h.shape[1]))
                node_in.index_add_(0, dst, h)
                # message for edge e = (incoming to its source) minus its reverse edge
                m = node_in[src] - h[rev]
                h = self.act(h0 + self.dropout(self.W_h(m)))

            # atom representation: gather final bond messages incoming to each atom
            node_msg = x.new_zeros((n_nodes, h.shape[1]))
            node_msg.index_add_(0, dst, h)
            atom_h = self.dropout(self.act(self.W_o(torch.cat([x, node_msg], dim=1))))

            # mean-pool atoms per graph
            graph_sum = x.new_zeros((n_graphs, atom_h.shape[1]))
            graph_sum.index_add_(0, batch, atom_h)
            counts = x.new_zeros((n_graphs, 1))
            counts.index_add_(0, batch, torch.ones((n_nodes, 1), device=x.device))
            graph_h = graph_sum / counts.clamp(min=1.0)

            return self.ffn(graph_h).squeeze(-1)


class GNNRegressor(BaseRegressor):
    """D-MPNN regressor with MC-dropout uncertainty, matching the ``BaseRegressor`` API.

    Parameters
    ----------
    hidden, depth, dropout
        Network width, number of message-passing steps, and dropout rate.
    epochs, lr, batch_size, weight_decay
        Standard training knobs. Defaults are tuned for small polymer datasets
        (tens to low hundreds of points), where a big network would simply
        memorise.
    mc_samples
        Number of stochastic forward passes used to estimate sigma when
        ``predict(..., return_std=True)`` is called.
    device
        ``"auto"`` picks Apple ``mps`` if present else ``cpu``. Pass ``"cpu"``
        explicitly for fully deterministic runs (tests do this).

    Notes
    -----
    Targets are standardised internally and predictions mapped back, because a
    network trains far better on a zero-mean, unit-variance target than on raw
    degrees Celsius. The scaling is fit on the training ``y`` only.
    """

    name = "dmpnn"

    def __init__(
        self,
        hidden: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        mc_samples: int = 30,
        device: str = "auto",
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        if not HAS_TORCH:  # pragma: no cover - environment dependent
            raise ImportError(
                "GNNRegressor requires PyTorch. Install it with: "
                'pip install -e ".[gnn]"  (or: pip install torch)'
            )
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.mc_samples = mc_samples
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self._model: _DMPNN | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    # -- device -----------------------------------------------------------
    def _resolve_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # -- batching ---------------------------------------------------------
    def _to_tensors(self, graphs: Sequence[PolymerGraph], device):
        batch = batch_graphs(list(graphs))
        rev = _reverse_edge_index(batch["edge_index"])
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        return {
            "x": t(batch["node_features"], torch.float32),
            "edge_index": t(batch["edge_index"], torch.long),
            "edge_attr": t(batch["edge_features"], torch.float32),
            "rev": t(rev, torch.long),
            "batch": t(batch["batch"], torch.long),
            "n_graphs": int(batch["n_graphs"]),
        }

    # -- API --------------------------------------------------------------
    def fit(self, X, y) -> GNNRegressor:
        graphs = list(np.asarray(X, dtype=object))
        y = np.asarray(y, dtype=np.float64)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_std = (y - self._y_mean) / self._y_std

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = self._resolve_device()

        node_dim = len(ATOM_FEATURE_NAMES)
        edge_dim = len(BOND_FEATURE_NAMES)
        self._model = _DMPNN(
            node_dim, edge_dim, self.hidden, self.depth, self.dropout
        ).to(device)
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.SmoothL1Loss()  # robust to the occasional bad label

        n = len(graphs)
        self._model.train()
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                tensors = self._to_tensors([graphs[i] for i in idx], device)
                target = torch.as_tensor(
                    y_std[idx], dtype=torch.float32, device=device
                )
                opt.zero_grad()
                pred = self._model(**tensors)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)
            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:3d}  loss {epoch_loss / n:.4f}")
        return self

    def _forward_all(self, graphs, device) -> np.ndarray:
        tensors = self._to_tensors(graphs, device)
        with torch.no_grad():
            out = self._model(**tensors)
        return out.detach().to("cpu").numpy()

    def predict(self, X, return_std: bool = False):
        if self._model is None:
            raise RuntimeError("call fit() first")
        graphs = list(np.asarray(X, dtype=object))
        device = self._resolve_device()

        if not return_std:
            self._model.eval()
            mu = self._forward_all(graphs, device)
            return mu * self._y_std + self._y_mean

        # MC-dropout: keep dropout active, sample several passes
        self._model.train()
        torch.manual_seed(self.random_state)
        samples = np.stack(
            [self._forward_all(graphs, device) for _ in range(self.mc_samples)]
        )
        mu = samples.mean(axis=0) * self._y_std + self._y_mean
        sigma = samples.std(axis=0) * self._y_std
        sigma = np.clip(sigma, 1e-6, None)
        return mu, sigma
