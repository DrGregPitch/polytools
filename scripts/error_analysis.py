#!/usr/bin/env python3
"""Error analysis: where the model fails, and a chemically literate look at why.

    python scripts/error_analysis.py --outdir results

Trains the best point model (gradient boosting on ECFP+descriptors), gets an
out-of-fold residual for *every* polymer via K-fold, and writes three things:

* ``error_analysis.png`` -- parity coloured by family, and mean signed residual
  per family (the shrinkage plot).
* ``error_analysis_structures.png`` -- the repeat-unit structures of the
  worst-predicted polymers, labelled true -> predicted.
* ``error_analysis.csv`` -- per-polymer residuals, and a family bias table.

The figures are the evidence; the *paragraph you write from them* is the
deliverable. Run this on real data (``--data``) and the mechanism you describe --
which chemistries fail and why -- is the part of the portfolio that shows your PhD.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from polytools import (
    GBMRegressor,
    featurize,
    group_bias_table,
    load_csv,
    load_toy_tg,
    oof_predictions,
    prepare_mol,
    regression_metrics,
    silence_rdkit,
    worst_predictions,
)


def load_dataset(args):
    if args.data is None:
        return load_toy_tg(), True
    ds = load_csv(
        args.data, psmiles_col=args.psmiles_col, target_col=args.target_col,
        group_col=args.group_col, name_col=args.name_col, units=args.units,
    )
    return ds, False


def structure_grid(psmiles, legends, path):
    """Render worst-predicted repeat units as a labelled structure grid."""
    from rdkit.Chem import Draw
    mols = [prepare_mol(ps, mode="cap") for ps in psmiles]
    img = Draw.MolsToGridImage(
        mols, legends=legends, molsPerRow=4,
        subImgSize=(260, 200), returnPNG=False,
    )
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--folds", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data", type=Path, default=None,
                        help="CSV of real data instead of the toy set")
    parser.add_argument("--psmiles-col", default="psmiles")
    parser.add_argument("--target-col", default="tg_c")
    parser.add_argument("--group-col", default="family")
    parser.add_argument("--name-col", default="name")
    parser.add_argument("--units", default="degC")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    silence_rdkit()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds, is_toy = load_dataset(args)
    print(f"Dataset: {ds}")
    X, _ = featurize(ds.psmiles, "ecfp+desc", n_bits=1024)

    # A residual for every polymer, each predicted while held out.
    oof = oof_predictions(
        X, ds.y,
        # backend="auto" resolves to LightGBM when it is installed and torch is
        # not loaded (both true here), and falls back to sklearn on a core
        # install -- forcing "lightgbm" would crash environments without the
        # optional [gbm] extra and bypass the OpenMP-clash guard in models.py.
        lambda: GBMRegressor(n_estimators=400, quantile_uncertainty=False),
        n_splits=args.folds, seed=args.seed,
    )
    resid = oof - ds.y
    m = regression_metrics(ds.y, oof)
    print(f"\nOut-of-fold RMSE {m['rmse']:.1f}  MAE {m['mae']:.1f}  "
          f"bias {m['bias']:+.1f}  ({args.folds}-fold)")

    per_poly = pd.DataFrame({
        "name": ds.names, "family": ds.groups, "y_true": ds.y,
        "y_pred_oof": oof, "residual": resid,
    })
    per_poly.to_csv(args.outdir / "error_analysis.csv", index=False)

    if ds.groups is not None:
        bias = group_bias_table(ds.y, oof, ds.groups)
        bt = pd.DataFrame(bias)
        bt.to_csv(args.outdir / "error_analysis_family_bias.csv", index=False)
        print("\nMean signed residual by family (negative = model under-predicts):")
        for r in bias:
            print(f"  {str(r['group']):16s} {r['mean_signed_residual']:+7.1f}  "
                  f"(n={r['n']}, mean Tg {r['mean_true']:+.0f})")

    worst = worst_predictions(ds.names, ds.y, oof, n=8)
    print("\nWorst-predicted polymers (largest |residual|):")
    for w in worst:
        print(f"  {str(w['name']):40s} {w['true']:+6.0f} -> {w['pred']:+6.0f}  "
              f"({w['residual']:+.0f})")

    if args.no_figures:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from polytools.plots import parity_plot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    parity_plot(ds.y, oof, ax=axes[0], units=ds.units,
                color_by=ds.groups if ds.groups is not None else None,
                color_label="family", title="Out-of-fold parity, by family")

    if ds.groups is not None:
        bt_sorted = bt.sort_values("mean_signed_residual")
        colors = ["#c0392b" if v < 0 else "#2b6cb0"
                  for v in bt_sorted["mean_signed_residual"]]
        axes[1].barh(bt_sorted["group"].astype(str),
                     bt_sorted["mean_signed_residual"], color=colors,
                     edgecolor="white")
        axes[1].axvline(0, color="k", linewidth=0.8)
        axes[1].set_xlabel(f"mean signed residual ({ds.units})")
        axes[1].set_title("Systematic bias by family\n(red = under-predicted)")
        axes[1].grid(axis="x", alpha=0.25)
    suptitle = "Error analysis" + (" (toy set -- method demonstration, not a benchmark)"
                                   if is_toy else "")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # worst-predicted structures, labelled
    idx = [int(np.where(ds.names == w["name"])[0][0]) for w in worst]
    legends = [f"{w['name']}: {w['true']:.0f}->{w['pred']:.0f}C" for w in worst]
    try:
        structure_grid(ds.psmiles[idx], legends,
                       args.outdir / "error_analysis_structures.png")
        print(f"\nFigures + structures written to {args.outdir}/")
    except Exception as exc:  # pragma: no cover - drawing is best-effort
        print(f"\nFigures written to {args.outdir}/ (structure grid skipped: {exc})")

    if is_toy:
        print("\nNOTE: the toy set has approximate/disputed labels (PTFE especially); "
              "some\napparent 'errors' are label noise. This demonstrates the method.")


if __name__ == "__main__":
    main()
