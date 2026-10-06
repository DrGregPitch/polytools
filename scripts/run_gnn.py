#!/usr/bin/env python3
"""Train the D-MPNN neural rung, with the periodic-edge ablation.

    python scripts/run_gnn.py --outdir results

Run as its own process, separate from ``run_benchmark.py``, on purpose: LightGBM
and PyTorch bundle incompatible OpenMP runtimes that crash when co-loaded on
macOS, so the GBM ladder (LightGBM) and the neural rung (PyTorch) each get a clean
process. Here torch *is* loaded, so any GBMRegressor would transparently fall back
to the sklearn backend -- but this script sticks to the mean floor and the GNN, so
the two tables stay directly comparable to the LightGBM benchmark on shared rows
(mean) and additive on the rest (the GNN).

The headline experiment: **periodic edge on vs. off, same architecture, same
split.** With it, message passing wraps around the chain; without it, the model
sees two chain ends the material does not have. Reporting whichever way it comes
out -- helps, hurts, or no effect, with a reason -- is the point.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from polytools import (
    MeanRegressor,
    SigmaRecalibrator,
    cluster_split,
    extrapolation_split,
    group_split,
    load_csv,
    load_toy_tg,
    random_split,
    regression_metrics,
    silence_rdkit,
    uncertainty_metrics,
)
from polytools.gnn import HAS_TORCH, GNNRegressor, graphs_from_psmiles


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
    n = len(ds)
    splits = {
        "random": random_split(n, seed=0),
        "cluster": cluster_split(ds.psmiles, cutoff=0.6),
        "extrapolation_high": extrapolation_split(ds.y, direction="high"),
    }
    if ds.groups is not None:
        splits["family"] = group_split(ds.groups, seed=0)
    return splits


def evaluate(model, G, y, sp) -> dict:
    """Fit on train, recalibrate sigma on val, score on test. Returns a row."""
    tr, va, te = sp.train, sp.val, sp.test
    model.fit(G[tr], y[tr])

    mu_va, sd_va = model.predict(G[va], return_std=True)
    recal = SigmaRecalibrator().fit(y[va], mu_va, sd_va) if len(va) >= 4 \
        else SigmaRecalibrator()

    mu_te, sd_raw = model.predict(G[te], return_std=True)
    sd_te = recal.transform(sd_raw)

    row = regression_metrics(y[te], mu_te)
    row.update(uncertainty_metrics(y[te], mu_te, sd_te))
    row["miscalibration_area_raw"] = uncertainty_metrics(
        y[te], mu_te, sd_raw
    )["miscalibration_area"]
    row["sigma_scale"] = recal.scale
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--device", default="cpu",
                        help="cpu, mps, or auto (cpu is deterministic)")
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

    if not HAS_TORCH:
        raise SystemExit(
            'PyTorch is required. Install the neural rung with:  pip install -e ".[gnn]"'
        )

    silence_rdkit()
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds, is_toy = load_dataset(args)
    print(f"Dataset: {ds}")

    # Two representations of the same polymers: with and without the chain-wrapping
    # periodic edge. This is the ablation.
    G_on = graphs_from_psmiles(ds.psmiles, periodic=True)
    G_off = graphs_from_psmiles(ds.psmiles, periodic=False)

    def make_gnn():
        return GNNRegressor(
            hidden=args.hidden, depth=args.depth, epochs=args.epochs,
            device=args.device, random_state=args.seed,
        )

    variants = {
        "mean": (None, "mean baseline (floor)"),
        "dmpnn_periodic": (G_on, "D-MPNN, periodic edge ON"),
        "dmpnn_finite": (G_off, "D-MPNN, periodic edge OFF"),
    }

    splits = build_splits(ds)
    rows = []
    print(f"\nNeural rung (test RMSE, {ds.units}):")
    for split_name, sp in splits.items():
        if len(sp.train) < 5 or len(sp.test) < 3:
            continue
        for key, (G, _desc) in variants.items():
            if key == "mean":
                # graph-free floor, evaluated on the same split
                m = MeanRegressor().fit(None, ds.y[sp.train])
                mu = m.predict(np.zeros(len(sp.test)))
                sd = np.full(len(sp.test), m._std)
                row = regression_metrics(ds.y[sp.test], mu)
                row.update(uncertainty_metrics(ds.y[sp.test], mu, sd))
                row["miscalibration_area_raw"] = row["miscalibration_area"]
                row["sigma_scale"] = 1.0
            else:
                row = evaluate(make_gnn(), G, ds.y, sp)
            row["split"] = split_name
            row["model"] = key
            rows.append(row)
            print(f"  {split_name:20s} {key:16s} RMSE={row['rmse']:6.1f}  "
                  f"MA={row['miscalibration_area']:.3f} (raw {row['miscalibration_area_raw']:.3f})")

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "results_gnn.csv", index=False)

    pivot = results.pivot(index="model", columns="split", values="rmse")
    order = [m for m in variants if m in pivot.index]
    pivot = pivot.loc[order]
    print(f"\nRMSE ({ds.units}) by model and split:")
    print(pivot.round(1).to_string())
    pivot.to_csv(args.outdir / "rmse_gnn.csv")

    with open(args.outdir / "rmse_gnn.md", "w") as fh:
        cols = list(pivot.columns)
        fh.write("| model | " + " | ".join(cols) + " |\n")
        fh.write("|:---|" + "---:|" * len(cols) + "\n")
        for model_name, r in pivot.round(1).iterrows():
            fh.write(f"| {model_name} | "
                     + " | ".join("" if pd.isna(v) else f"{v:.1f}" for v in r)
                     + " |\n")

    # The ablation verdict, stated plainly per split.
    print("\nPeriodic-edge ablation (RMSE, ON minus OFF; negative = periodic helps):")
    for split_name in pivot.columns:
        if "dmpnn_periodic" in pivot.index and "dmpnn_finite" in pivot.index:
            delta = pivot.loc["dmpnn_periodic", split_name] - pivot.loc["dmpnn_finite", split_name]
            verdict = "periodic helps" if delta < -1 else \
                      "periodic hurts" if delta > 1 else "no clear effect"
            print(f"  {split_name:20s} {delta:+6.1f}   {verdict}")

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from polytools.plots import split_comparison_plot

        comp = {
            s: dict(zip(results[results.split == s].model,
                        results[results.split == s].rmse))
            for s in results.split.unique()
        }
        ax = split_comparison_plot(comp, metric="rmse", units=args.units,
                                   title="Neural rung: periodic-edge ablation by split")
        ax.figure.savefig(args.outdir / "gnn_ablation.png", dpi=150,
                          bbox_inches="tight")
        plt.close(ax.figure)
        print(f"\nFigure written to {args.outdir}/gnn_ablation.png")

    print(f"\nDone in {time.time() - t0:.1f}s. Results in {args.outdir}/")
    if is_toy:
        print("\nREMINDER: the bundled dataset is a 60-polymer smoke test. These numbers "
              "are\nnot a benchmark result and must not be reported as one.")


if __name__ == "__main__":
    main()
