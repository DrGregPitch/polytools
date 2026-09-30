"""Test suite for polytools.

Tests are organized by the invariant they protect, not by module. Several of
these were written in response to real bugs found during development -- notably
``test_cyclize_actually_cyclizes_vinyl_polymers``, which catches the case where
head-to-tail closure of a single vinyl repeat unit silently degenerates into
plain capping because the two anchor atoms are already bonded. That bug is
invisible without an explicit ring-count assertion, and it would have quietly
invalidated every descriptor computed in ``cyclize`` mode.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from polytools import (
    CopolymerFeaturizer,
    DescriptorFeaturizer,
    EnsembleRegressor,
    GBMRegressor,
    MeanRegressor,
    MorganFeaturizer,
    PolymerParseError,
    RidgeDescriptorRegressor,
    SigmaRecalibrator,
    backbone_length,
    canonical_psmiles,
    cluster_split,
    extrapolation_split,
    featurize,
    group_split,
    is_valid_psmiles,
    load_toy_tg,
    miscalibration_area,
    mol_from_psmiles,
    prepare_mol,
    random_split,
    regression_metrics,
    scaffold_split,
    silence_rdkit,
    split_difficulty,
)

silence_rdkit()

PS_STYRENE = "[*]CC([*])c1ccccc1"
PS_ETHYLENE = "[*]CC[*]"
PS_PET = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
PS_PIB = "[*]CC(C)(C)[*]"
PS_PDMS = "[*][Si](C)(C)O[*]"

ALL_PS = [PS_STYRENE, PS_ETHYLENE, PS_PET, PS_PIB, PS_PDMS]


# --------------------------------------------------------------------------
# chem
# --------------------------------------------------------------------------

def test_parses_valid_psmiles():
    mol = mol_from_psmiles(PS_STYRENE)
    assert mol.GetNumAtoms() == 10  # 8 heavy + 2 dummies


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not_a_smiles",
        "CCO",             # no connection points
        "[*]CC",           # only one
        "[*]CC([*])[*]",   # three
    ],
)
def test_rejects_bad_psmiles(bad):
    with pytest.raises(PolymerParseError):
        mol_from_psmiles(bad)
    assert not is_valid_psmiles(bad)


def test_canonicalization_unifies_dummy_notation():
    assert canonical_psmiles("*CC*") == canonical_psmiles("[*]CC[*]")


def test_canonicalization_distinguishes_different_polymers():
    assert canonical_psmiles(PS_STYRENE) != canonical_psmiles(PS_ETHYLENE)


def test_backbone_length():
    assert backbone_length(mol_from_psmiles(PS_STYRENE)) == 2
    assert backbone_length(mol_from_psmiles(PS_PET)) == 10


def test_cap_removes_all_dummy_atoms():
    for ps in ALL_PS:
        mol = prepare_mol(ps, mode="cap")
        assert all(a.GetAtomicNum() != 0 for a in mol.GetAtoms())


def test_oligomer_scales_heavy_atom_count():
    """An n-mer must contain n times the repeat unit's heavy atoms."""
    for ps in ALL_PS:
        unit_heavy = sum(
            1 for a in mol_from_psmiles(ps).GetAtoms() if a.GetAtomicNum() != 0
        )
        for n in (1, 2, 5):
            mol = prepare_mol(ps, mode="oligomer", n_repeat=n)
            assert mol.GetNumAtoms() == n * unit_heavy, f"{ps} n={n}"


def test_cyclize_actually_cyclizes_vinyl_polymers():
    """Regression test for a real bug.

    In any vinyl polymer the head and tail anchors are already bonded to each
    other, so naive head-to-tail closure of a single repeat unit either fails or
    no-ops, and the code silently fell back to capping. The result looked
    plausible -- a valid molecule with sensible descriptors -- while being the
    wrong molecule entirely. Asserting an actual ring exists is the only way to
    catch it.
    """
    for ps in (PS_STYRENE, PS_ETHYLENE, PS_PIB, PS_PDMS):
        mol = prepare_mol(ps, mode="cyclize")
        assert mol.GetRingInfo().NumRings() >= 1, f"{ps} produced no ring"
        assert all(a.GetAtomicNum() != 0 for a in mol.GetAtoms())
        largest = max(len(r) for r in mol.GetRingInfo().AtomRings())
        assert largest >= 8, f"{ps} ring of {largest} atoms is unphysically strained"


def test_cyclize_gives_known_chemistry_for_pdms():
    """Cyclized PDMS should be a cyclosiloxane: alternating Si and O in the ring."""
    mol = prepare_mol(PS_PDMS, mode="cyclize")
    ring = max(mol.GetRingInfo().AtomRings(), key=len)
    symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in ring)
    assert set(symbols) == {"Si", "O"}
    assert symbols.count("Si") == symbols.count("O")


def test_preparation_modes_give_different_molecules():
    smis = {
        mode: Chem.MolToSmiles(prepare_mol(PS_STYRENE, mode=mode))
        for mode in ("cap", "oligomer", "cyclize")
    }
    assert len(set(smis.values())) == 3, smis


def test_fallback_rescues_units_that_fail_the_requested_mode():
    """Real-data hardening: a fused-ring polyimide fails to kekulize when
    oligomerised/capped but cyclises cleanly. With fallback it must still prepare;
    without fallback the failure is loud."""
    polyimide = "*n1c(=O)c2cc3c(cc2c1=O)c(=O)n(c3=O)CCCCCCCCC*"
    # cyclize succeeds, so oligomer-with-fallback should return a valid molecule
    mol = prepare_mol(polyimide, mode="oligomer", fallback=True)
    assert mol is not None and all(a.GetAtomicNum() != 0 for a in mol.GetAtoms())
    # without fallback, oligomer mode raises for this unit
    with pytest.raises(PolymerParseError):
        prepare_mol(polyimide, mode="oligomer", fallback=False)


def test_extensivity_count_reflects_the_mode_that_actually_ran():
    """Regression: when fallback reroutes a unit to a different mode, the
    repeat-unit count must follow, or per-unit descriptors are silently scaled
    by the wrong factor for exactly that chemical family.

    The polyimide below falls back from oligomer to cyclize; dividing its
    macrocycle descriptors by the oligomer's n_repeat=3 gave MolWt_per_unit
    ~3.3x too large. Pin mode-independence instead: per-unit MolWt must agree
    between the fallback path and an explicit cyclize within a few percent.
    """
    from rdkit.Chem import Descriptors as D

    from polytools import prepare_mol_counted

    polyimide = "*n1c(=O)c2cc3c(cc2c1=O)c(=O)n(c3=O)CCCCCCCCC*"
    mol_fb, n_fb = prepare_mol_counted(polyimide, mode="oligomer", fallback=True)
    mol_cy, n_cy = prepare_mol_counted(polyimide, mode="cyclize", fallback=False)
    assert n_fb == n_cy > 1  # the count followed the fallback to cyclize
    per_unit_fb = D.MolWt(mol_fb) / n_fb
    per_unit_cy = D.MolWt(mol_cy) / n_cy
    assert per_unit_fb == pytest.approx(per_unit_cy, rel=0.05)

    f = DescriptorFeaturizer(mode="oligomer", normalize_extensive=True)
    X = f.transform([polyimide])
    mw = X[0, f.feature_names.index("MolWt_per_unit")]
    assert mw == pytest.approx(per_unit_cy, rel=0.05)


# --------------------------------------------------------------------------
# featurize
# --------------------------------------------------------------------------

def test_descriptor_normalization_recovers_repeat_unit_mass():
    """MolWt per repeat unit must be near the true monomer mass, mode-independent.

    Styrene repeat unit is 104.15 g/mol. Without normalization a 3-mer reports
    ~314 and a macrocycle ~1040, neither of which is a property of the polymer.
    """
    for mode in ("cap", "oligomer", "cyclize"):
        f = DescriptorFeaturizer(mode=mode, normalize_extensive=True)
        X = f.transform([PS_STYRENE])
        mw = X[0, f.feature_names.index("MolWt_per_unit")]
        assert 102 < mw < 108, f"mode={mode} gave MolWt/unit={mw}"


def test_descriptors_are_finite():
    X = DescriptorFeaturizer().transform(ALL_PS)
    assert np.isfinite(X).all()


def test_morgan_shape_and_sensitivity():
    f = MorganFeaturizer(n_bits=512)
    X = f.transform(ALL_PS)
    assert X.shape == (len(ALL_PS), 512)
    assert (X > 0).any(axis=1).all()
    # different polymers must not collide into identical fingerprints
    assert len(np.unique(X, axis=0)) == len(ALL_PS)


def test_featurize_kinds_have_matching_names():
    for kind in ("ecfp", "desc", "ecfp+desc"):
        X, names = featurize(ALL_PS, kind, **({"n_bits": 256} if "ecfp" in kind else {}))
        assert X.shape[1] == len(names)


def test_copolymer_weighting_is_linear():
    cf = CopolymerFeaturizer()
    idx = cf.feature_names.index("MolWt_per_unit")
    pure_a = cf.transform([[(PS_STYRENE, 1.0)]])[0, idx]
    pure_b = cf.transform([[(PS_ETHYLENE, 1.0)]])[0, idx]
    mixed = cf.transform([[(PS_STYRENE, 0.5), (PS_ETHYLENE, 0.5)]])[0, idx]
    assert mixed == pytest.approx(0.5 * pure_a + 0.5 * pure_b, rel=1e-6)


def test_copolymer_normalizes_fractions():
    cf = CopolymerFeaturizer(normalize=True)
    a = cf.transform([[(PS_STYRENE, 1.0), (PS_ETHYLENE, 1.0)]])
    b = cf.transform([[(PS_STYRENE, 0.5), (PS_ETHYLENE, 0.5)]])
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_copolymer_rejects_bad_composition():
    cf = CopolymerFeaturizer()
    with pytest.raises(ValueError):
        cf.transform([[]])
    with pytest.raises(ValueError):
        cf.transform([[(PS_STYRENE, -0.5), (PS_ETHYLENE, 1.5)]])


def test_copolymer_cannot_distinguish_sequence():
    """Documents the known limitation the mixing rule has by construction.

    A block and a random copolymer of identical composition are indistinguishable
    to a linear mixing rule. This test pins that behaviour so it stays a
    *documented* limitation rather than becoming a silent surprise -- and so any
    future sequence-aware featurizer has an explicit contract to break.
    """
    cf = CopolymerFeaturizer()
    x1 = cf.transform([[(PS_STYRENE, 0.5), (PS_ETHYLENE, 0.5)]])
    x2 = cf.transform([[(PS_ETHYLENE, 0.5), (PS_STYRENE, 0.5)]])
    np.testing.assert_allclose(x1, x2, rtol=1e-9)


# --------------------------------------------------------------------------
# splitters
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy():
    return load_toy_tg()


def test_toy_dataset_loads(toy):
    assert len(toy) >= 55
    assert toy.groups is not None
    assert toy.y.min() < -100 and toy.y.max() > 200


def test_splits_are_disjoint_and_complete(toy):
    n = len(toy)
    splits = {
        "random": random_split(n),
        "scaffold": scaffold_split(toy.psmiles),
        "cluster": cluster_split(toy.psmiles),
        "family": group_split(toy.groups),
        "extrap": extrapolation_split(toy.y),
    }
    for name, sp in splits.items():
        allidx = np.concatenate([sp.train, sp.val, sp.test])
        assert len(allidx) == n, name
        assert len(np.unique(allidx)) == n, f"{name} has overlapping indices"


def test_group_split_never_leaks_a_group(toy):
    sp = group_split(toy.groups)
    fam = np.asarray(toy.groups)
    train_f, test_f = set(fam[sp.train]), set(fam[sp.test])
    assert not (train_f & test_f)


def test_extrapolation_split_holds_out_the_top(toy):
    sp = extrapolation_split(toy.y, direction="high")
    assert toy.y[sp.test].min() >= toy.y[sp.train].max()


def test_structured_splits_are_harder_than_random(toy):
    """The whole point of the library: structured splits must be measurably harder."""
    rnd = split_difficulty(toy.psmiles, random_split(len(toy), seed=0))
    grp = split_difficulty(toy.psmiles, group_split(toy.groups, seed=0))
    assert grp["nn_similarity_median"] < rnd["nn_similarity_median"]
    assert grp["frac_near_duplicate_gt_0.9"] <= rnd["frac_near_duplicate_gt_0.9"]


def test_split_fractions_are_respected(toy):
    sp = random_split(len(toy), frac_train=0.6, frac_val=0.2, frac_test=0.2)
    assert abs(len(sp.train) / len(toy) - 0.6) < 0.05


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        random_split(10, frac_train=0.5, frac_val=0.3, frac_test=0.3)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_perfect_prediction_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_metrics_handle_constant_predictions_without_warning():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, np.full(5, 3.0))
    assert np.isnan(m["spearman"])
    assert m["r2"] == pytest.approx(0.0, abs=1e-9)


def test_recalibrator_recovers_known_scale():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(0, 10, n)
    pred = y + rng.normal(0, 3.0, n)
    stated = np.full(n, 1.0)  # 3x overconfident
    recal = SigmaRecalibrator().fit(y, pred, stated)
    assert recal.scale == pytest.approx(3.0, rel=0.1)
    before = miscalibration_area(y, pred, stated)
    after = miscalibration_area(y, pred, recal.transform(stated))
    assert after < before / 3


def test_miscalibration_area_zero_for_calibrated_model():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.normal(0, 5, n)
    pred = y + rng.normal(0, 2.0, n)
    assert miscalibration_area(y, pred, np.full(n, 2.0)) < 0.03


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def test_models_fit_predict_and_report_uncertainty(toy):
    X, _ = featurize(toy.psmiles, "desc")
    sp = random_split(len(toy), seed=0)
    for model in (MeanRegressor(), RidgeDescriptorRegressor(),
                  GBMRegressor(n_estimators=50)):
        model.fit(X[sp.train], toy.y[sp.train])
        mu, sd = model.predict(X[sp.test], return_std=True)
        assert mu.shape == sd.shape == (len(sp.test),)
        assert np.isfinite(mu).all() and (sd > 0).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RidgeDescriptorRegressor().predict(np.zeros((2, 3)))


def test_mean_regressor_rmse_equals_test_std(toy):
    """The floor is well defined: predicting the train mean gives RMSE ~ test std."""
    X, _ = featurize(toy.psmiles, "desc")
    sp = random_split(len(toy), seed=0)
    m = MeanRegressor().fit(X[sp.train], toy.y[sp.train])
    rmse = regression_metrics(toy.y[sp.test], m.predict(X[sp.test]))["rmse"]
    assert rmse == pytest.approx(np.std(toy.y[sp.test]), rel=0.35)


def test_ensemble_uncertainty_is_positive_and_varies(toy):
    X, _ = featurize(toy.psmiles, "desc")
    sp = random_split(len(toy), seed=0)
    ens = EnsembleRegressor(
        factory=lambda i: GBMRegressor(n_estimators=40, random_state=i,
                                       quantile_uncertainty=False),
        n_members=4,
    ).fit(X[sp.train], toy.y[sp.train])
    _, sd = ens.predict(X[sp.test], return_std=True)
    assert (sd > 0).all()
    assert np.std(sd) > 0, "ensemble sigma is constant -- members are identical"


def test_a_real_model_beats_the_mean_baseline_on_random_split(toy):
    """Sanity check that the pipeline learns anything at all."""
    X, _ = featurize(toy.psmiles, "ecfp+desc", n_bits=512)
    sp = random_split(len(toy), seed=0)
    base = MeanRegressor().fit(X[sp.train], toy.y[sp.train])
    gbm = GBMRegressor(n_estimators=300).fit(X[sp.train], toy.y[sp.train])
    r_base = regression_metrics(toy.y[sp.test], base.predict(X[sp.test]))["rmse"]
    r_gbm = regression_metrics(toy.y[sp.test], gbm.predict(X[sp.test]))["rmse"]
    assert r_gbm < r_base


# --------------------------------------------------------------------------
# analysis (error analysis helpers)
# --------------------------------------------------------------------------

from polytools import (  # noqa: E402
    group_bias_table,
    oof_predictions,
    worst_predictions,
)


def test_oof_predictions_cover_every_row_and_are_held_out(toy):
    """Every polymer gets a prediction, and it beats the mean out of fold."""
    X, _ = featurize(toy.psmiles, "desc")
    oof = oof_predictions(X, toy.y, lambda: GBMRegressor(n_estimators=100), n_splits=5)
    assert oof.shape == toy.y.shape
    assert np.isfinite(oof).all()
    # out-of-fold error must still beat predicting the global mean
    rmse_oof = regression_metrics(toy.y, oof)["rmse"]
    assert rmse_oof < np.std(toy.y)


def test_group_bias_table_is_sorted_and_signed():
    y_true = np.array([100.0, 120.0, -50.0, -40.0])
    y_pred = np.array([70.0, 90.0, -10.0, 0.0])   # under high group, over low group
    groups = np.array(["hi", "hi", "lo", "lo"])
    table = group_bias_table(y_true, y_pred, groups)
    assert table[0]["mean_signed_residual"] < table[-1]["mean_signed_residual"]
    hi = next(r for r in table if r["group"] == "hi")
    assert hi["mean_signed_residual"] == pytest.approx(-30.0)  # (70-100 + 90-120)/2


def test_worst_predictions_returns_largest_absolute_errors():
    names = np.array(["a", "b", "c"])
    worst = worst_predictions(names, [0.0, 0.0, 0.0], [1.0, 50.0, -5.0], n=2)
    assert [w["name"] for w in worst] == ["b", "c"]
    assert worst[0]["residual"] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------

from polytools.graphs import (  # noqa: E402
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    batch_graphs,
    psmiles_to_graph,
)


def test_graph_feature_widths_match_declared_names():
    """Silent width drift between the arrays and their names breaks every
    downstream explanation, so pin it."""
    g = psmiles_to_graph(PS_STYRENE)
    assert g.node_features.shape[1] == len(ATOM_FEATURE_NAMES)
    assert g.edge_features.shape[1] == len(BOND_FEATURE_NAMES)


def test_dummy_atoms_are_not_graph_nodes():
    """Connection points are notation, not chemistry, and must never be nodes."""
    for ps in ALL_PS:
        mol = mol_from_psmiles(ps)
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 0)
        assert psmiles_to_graph(ps).n_nodes == heavy


def test_backbone_anchors_are_flagged():
    g = psmiles_to_graph(PS_STYRENE)
    col = ATOM_FEATURE_NAMES.index("is_backbone_anchor")
    flagged = set(np.where(g.node_features[:, col] > 0)[0].tolist())
    assert flagged == set(g.anchor_indices)


def test_periodic_edges_added_and_ablatable():
    on = psmiles_to_graph(PS_STYRENE, periodic=True)
    off = psmiles_to_graph(PS_STYRENE, periodic=False)
    assert on.n_edges == off.n_edges + 2          # both directions
    assert on.edge_is_periodic.sum() == 2
    assert off.edge_is_periodic.sum() == 0


def test_periodic_polyethylene_graph_is_a_cycle():
    """The point of the periodic edge: the chain must wrap.

    With periodicity on, every backbone carbon in polyethylene has graph degree
    2 -- the repeat unit closes into a ring rather than presenting two chain
    ends that do not exist in the material.
    """
    g = psmiles_to_graph(PS_ETHYLENE, periodic=True)
    degrees = np.bincount(g.edge_index[0], minlength=g.n_nodes)
    assert (degrees == 2).all(), degrees

    finite = psmiles_to_graph(PS_ETHYLENE, periodic=False)
    fin_deg = np.bincount(finite.edge_index[0], minlength=finite.n_nodes)
    assert (fin_deg == 1).all(), "non-periodic graph should have two loose ends"


def test_edges_are_bidirectional():
    g = psmiles_to_graph(PS_PET)
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert all((j, i) in pairs for i, j in pairs)


def test_batching_offsets_indices_correctly():
    graphs = [psmiles_to_graph(ps) for ps in ALL_PS]
    b = batch_graphs(graphs)
    assert b["node_features"].shape[0] == sum(g.n_nodes for g in graphs)
    assert b["edge_index"].shape[1] == sum(g.n_edges for g in graphs)
    assert b["edge_index"].max() < b["node_features"].shape[0]
    assert np.bincount(b["batch"]).tolist() == [g.n_nodes for g in graphs]


def test_batch_rejects_empty():
    with pytest.raises(ValueError):
        batch_graphs([])
