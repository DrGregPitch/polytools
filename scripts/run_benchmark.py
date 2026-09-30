#!/usr/bin/env python3
"""Run the full model ladder across every split and write results + figures.

    python scripts/run_benchmark.py --outdir results

This is the template for Project 1 in the roadmap. The structure -- every model
on every split, uncertainty recalibrated on validation, figures written to disk,
results table dumped as CSV -- is what makes a result reproducible. Swap in a
real dataset and a graph neural network and the harness does not change.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from polytools import (
    EnsembleRegressor,
    GBMRegressor,
    MeanRegressor,
    RidgeDescriptorRegressor,
    SigmaRecalibrator,
    cluster_split,
    extrapolation_split,
    featurize,
    group_split,
    load_csv,
    load_toy_tg,
    random_split,
    regression_metrics,
    scaffold_split,
    silence_rdkit,
    split_difficulty,
    uncertainty_metrics,
)


def load_dataset(args):
    """Toy set by default, or a real CSV via --data (run through the hygiene pipeline)."""
    if args.data is None:
        return load_toy_tg(), True
    ds = load_csv(
        args.data,
        psmiles_col=args.psmiles_col,
        target_col=args.target_col,
        group_col=args.group_col,
        name_col=args.name_col,
        units=args.units,
    )
    return ds, False


def build_splits(ds) -> dict:
    """Random first as the optimistic reference, then progressively harder.

    The ``family`` split is only built when the dataset carries group labels;
    real CSVs without a ``--group-col`` simply skip it.
    """
    n = len(ds)
    splits = {
        "random": random_split(n, seed=0),
        "scaffold": scaffold_split(ds.psmiles),
        "cluster": cluster_split(ds.psmiles, cutoff=0.6),
        "extrapolation_high": extrapolation_split(ds.y, direction="high"),
    }
    if ds.groups is not None:
        splits["family"] = group_split(ds.groups, seed=0)
    return splits


def build_models() -> dict:
    return {
        "mean": lambda: MeanRegressor(),
        "ridge_desc": lambda: RidgeDescriptorRegressor(),
        "gbm": lambda: GBMRegressor(n_estimators=400, quantile_uncertainty=True),
        "gbm_ensemble": lambda: EnsembleRegressor(
            factory=lambda i: GBMRegressor(
                n_estimators=400, random_state=i, quantile_uncertainty=False
            ),
            n_members=5,
            bootstrap=True,
            include_member_sigma=False,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--data", type=Path, default=None,
                        help="CSV of real data to use instead of the toy set")
    parser.add_argument("--psmiles-col", default="psmiles")
    parser.add_argument("--target-col", default="tg_c")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--name-col", default=None)
    parser.add_argument("--units", default="degC")
    args = parser.parse_args()

    silence_rdkit()
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds, is_toy = load_dataset(args)
    print(f"Dataset: {ds}")
    print(f"  provenance: {json.dumps(ds.report)}")

    # Feature matrices are computed once and reused: featurization is
    # deterministic and split-independent, so recomputing per split would only
    # waste time. It is NOT a leak -- no target information is involved.
    print("\nFeaturizing...")
    feature_sets = {
        "desc": featurize(ds.psmiles, "desc")[0],
        "ecfp": featurize(ds.psmiles, "ecfp", n_bits=1024)[0],
        "ecfp+desc": featurize(ds.psmiles, "ecfp+desc", n_bits=1024)[0],
    }
    for k, v in feature_sets.items():
        print(f"  {k:10s} {v.shape}")

    model_features = {
        "mean": "desc",
        "ridge_desc": "desc",
        "gbm": "ecfp+desc",
        "gbm_ensemble": "ecfp+desc",
    }

    splits = build_splits(ds)
    models = build_models()

    print("\nSplit difficulty (nearest-neighbour Tanimoto, test -> train):")
    diff_rows = []
    for name, sp in splits.items():
        d = split_difficulty(ds.psmiles, sp)
        d["split"] = name
        d.update({f"n_{k}": len(v) for k, v in sp.items()})
        diff_rows.append(d)
        print(
            f"  {name:20s} median={d['nn_similarity_median']:.3f}  "
            f">0.9={d['frac_near_duplicate_gt_0.9']:.2f}  "
            f"<0.4={d['frac_novel_lt_0.4']:.2f}"
        )
    pd.DataFrame(diff_rows).to_csv(args.outdir / "split_difficulty.csv", index=False)

    rows = []
    predictions = {}
    print("\nBenchmark:")
    for split_name, sp in splits.items():
        tr, va, te = sp.train, sp.val, sp.test
        if len(tr) < 5 or len(te) < 3:
            print(f"  {split_name}: too small, skipping")
            continue

        for model_name, factory in models.items():
            X = feature_sets[model_features[model_name]]
            model = factory().fit(X[tr], ds.y[tr])

            # Recalibrate sigma on validation, never on test.
            mu_va, sd_va = model.predict(X[va], return_std=True)
            recal = SigmaRecalibrator().fit(ds.y[va], mu_va, sd_va) if len(va) >= 4 \
                else SigmaRecalibrator()

            mu_te, sd_te = model.predict(X[te], return_std=True)
            sd_raw = sd_te.copy()
            sd_te = recal.transform(sd_te)

            row = {"split": split_name, "model": model_name}
            row.update(regression_metrics(ds.y[te], mu_te))
            row.update(uncertainty_metrics(ds.y[te], mu_te, sd_te))
            row["sigma_scale"] = recal.scale
            row["miscalibration_area_raw"] = uncertainty_metrics(
                ds.y[te], mu_te, sd_raw
            )["miscalibration_area"]
            rows.append(row)

            predictions[(split_name, model_name)] = (te, mu_te, sd_te)
            print(
                f"  {split_name:20s} {model_name:14s} "
                f"RMSE={row['rmse']:6.1f}  MAE={row['mae']:6.1f}  "
                f"R2={row['r2']:6.2f}  MA={row['miscalibration_area']:.3f} "
                f"(raw {row['miscalibration_area_raw']:.3f})"
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "results.csv", index=False)

    pivot = results.pivot(index="model", columns="split", values="rmse")
    order = [m for m in models if m in pivot.index]
    pivot = pivot.loc[order]
    print("\nRMSE (degC) by model and split:")
    print(pivot.round(1).to_string())
    pivot.to_csv(args.outdir / "rmse_table.csv")

    # Written by hand rather than via DataFrame.to_markdown(), which silently
    # requires `tabulate` -- an undeclared dependency is exactly the kind of
    # thing that makes a repo fail to run on a reviewer's machine.
    with open(args.outdir / "rmse_table.md", "w") as fh:
        cols = list(pivot.columns)
        fh.write("| model | " + " | ".join(cols) + " |\n")
        fh.write("|:---|" + "---:|" * len(cols) + "\n")
        for model_name, row in pivot.round(1).iterrows():
            fh.write(
                f"| {model_name} | "
                + " | ".join("" if pd.isna(v) else f"{v:.1f}" for v in row)
                + " |\n"
            )

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from polytools.plots import split_comparison_plot, summary_figure

        best = "gbm_ensemble" if "gbm_ensemble" in models else "gbm"
        for split_name in ("random", "cluster"):
            key = (split_name, best)
            if key not in predictions:
                continue
            te, mu, sd = predictions[key]
            fig = summary_figure(
                ds.y[te], mu, sd,
                color_by=[ds.groups[i] for i in te] if ds.groups is not None else None,
                units=ds.units,
                suptitle=f"{best} on {split_name} split"
                + (" (toy dataset -- not a benchmark)" if is_toy else ""),
            )
            fig.savefig(args.outdir / f"summary_{split_name}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

        comp = {
            s: dict(zip(results[results.split == s].model,
                        results[results.split == s].rmse))
            for s in results.split.unique()
        }
        ax = split_comparison_plot(comp, metric="rmse", units=ds.units)
        ax.figure.savefig(args.outdir / "split_comparison.png", dpi=150,
                          bbox_inches="tight")
        plt.close(ax.figure)
        print(f"\nFigures written to {args.outdir}/")

    print(f"\nDone in {time.time() - t0:.1f}s. Results in {args.outdir}/")
    if is_toy:
        print(
            "\nREMINDER: the bundled dataset is a 60-polymer smoke test. These "
            "numbers are\nnot a benchmark result and must not be reported as one."
        )


if __name__ == "__main__":
    main()
