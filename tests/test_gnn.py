"""Tests for the D-MPNN rung.

Kept separate from ``test_polytools.py`` so the core suite never imports torch:
the base library must install and test without a deep-learning framework. Every
test here skips cleanly when torch is absent (e.g. base CI), and runs on CPU with
a fixed seed for determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch is missing

from polytools import (  # noqa: E402
    MeanRegressor,
    load_toy_tg,
    random_split,
    regression_metrics,
)
from polytools.gnn import (  # noqa: E402
    GNNRegressor,
    _reverse_edge_index,
    graphs_from_psmiles,
)
from polytools.graphs import batch_graphs, psmiles_to_graph  # noqa: E402

PS = ["[*]CC[*]", "[*]CC([*])c1ccccc1", "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"]


@pytest.fixture(scope="module")
def toy():
    return load_toy_tg()


def test_graphs_from_psmiles_is_indexable_like_a_feature_matrix():
    G = graphs_from_psmiles(PS)
    assert G.dtype == object and G.shape == (3,)
    idx = np.array([0, 2])
    assert [g.psmiles for g in G[idx]] == [PS[0], PS[2]]  # fancy-indexes like X


def test_reverse_edge_index_is_an_involution():
    """rev(rev(e)) == e for every edge, and rev(e) really is the reverse bond."""
    batch = batch_graphs([psmiles_to_graph(p) for p in PS])
    ei = batch["edge_index"]
    rev = _reverse_edge_index(ei)
    assert np.array_equal(rev[rev], np.arange(len(rev)))
    for e in range(ei.shape[1]):
        assert ei[0, e] == ei[1, rev[e]] and ei[1, e] == ei[0, rev[e]]


def test_gnn_fits_and_beats_mean_baseline(toy):
    """The neural rung must at least clear the floor on a random split."""
    G = graphs_from_psmiles(toy.psmiles)
    sp = random_split(len(toy), seed=0)
    gnn = GNNRegressor(epochs=200, hidden=64, device="cpu", random_state=0)
    gnn.fit(G[sp.train], toy.y[sp.train])
    mu = gnn.predict(G[sp.test])
    base = MeanRegressor().fit(None, toy.y[sp.train]).predict(np.zeros(len(sp.test)))
    r_gnn = regression_metrics(toy.y[sp.test], mu)["rmse"]
    r_base = regression_metrics(toy.y[sp.test], base)["rmse"]
    assert r_gnn < r_base


def test_gnn_mc_dropout_uncertainty_is_positive_and_varies(toy):
    G = graphs_from_psmiles(toy.psmiles)
    sp = random_split(len(toy), seed=0)
    gnn = GNNRegressor(epochs=120, hidden=48, mc_samples=20, device="cpu",
                       random_state=0).fit(G[sp.train], toy.y[sp.train])
    mu, sd = gnn.predict(G[sp.test], return_std=True)
    assert mu.shape == sd.shape == (len(sp.test),)
    assert np.isfinite(mu).all() and (sd > 0).all()
    assert np.std(sd) > 0, "MC-dropout sigma is constant -- dropout is not active"


def test_periodic_ablation_changes_predictions(toy):
    """Turning the chain-wrapping edge on vs off must actually change the model.

    This is the experiment Project 1 reports; if the two representations produced
    identical predictions, the ablation would be meaningless. Uses a subset for
    speed -- the point is that the periodic edge reaches the model, not accuracy.
    """
    sp = random_split(len(toy), seed=0)
    G_on = graphs_from_psmiles(toy.psmiles, periodic=True)
    G_off = graphs_from_psmiles(toy.psmiles, periodic=False)

    on = GNNRegressor(epochs=80, hidden=48, device="cpu", random_state=0)
    on.fit(G_on[sp.train], toy.y[sp.train])
    off = GNNRegressor(epochs=80, hidden=48, device="cpu", random_state=0)
    off.fit(G_off[sp.train], toy.y[sp.train])

    mu_on = on.predict(G_on[sp.test])
    mu_off = off.predict(G_off[sp.test])
    assert not np.allclose(mu_on, mu_off)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GNNRegressor(device="cpu").predict(graphs_from_psmiles(PS))
