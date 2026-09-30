#!/usr/bin/env python3
"""Download an openly-licensed real polymer dataset for use with --data.

    python scripts/fetch_data.py radonpy      # ~1,077 homopolymers, MD properties

No dataset is committed to this repo (a portfolio repo should carry no data blobs);
this fetches one on demand from its permissively-licensed source, into a gitignored
cache. Respect each dataset's license and cite it if you publish results.

Datasets
--------
``radonpy``
    RadonPy PI1070: 1,077 homopolymers with molecular-dynamics-computed properties
    (density, thermal conductivity, refractive index, heat capacity, and more) and
    repeat-unit pSMILES. **BSD-3-Clause.**
    Source: https://github.com/RadonPy/RadonPy   Cite: Hayashi et al., npj Comput.
    Mater. 8, 222 (2022).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SOURCES = {
    "radonpy": {
        "url": (
            "https://raw.githubusercontent.com/RadonPy/RadonPy/"
            "648c9a492808339c9bb7ad2c1137e5a7b07614ca/data/PI1070.csv"
        ),
        "filename": "PI1070.csv",
        "license": "BSD-3-Clause (RadonPy). Cite Hayashi et al., npj Comput. Mater. 8, 222 (2022).",
        "psmiles_col": "smiles",
        "targets": "density, thermal_conductivity, refractive_index, Cp, ...",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SOURCES), help="which dataset to fetch")
    parser.add_argument("--outdir", default="data_cache", type=Path)
    args = parser.parse_args()

    src = SOURCES[args.dataset]
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / src["filename"]

    print(f"Fetching {args.dataset} from:\n  {src['url']}")
    urllib.request.urlretrieve(src["url"], dest)  # noqa: S310 - trusted https source
    size_kb = dest.stat().st_size / 1024
    print(f"Saved {dest} ({size_kb:.0f} KB)")
    print(f"\nLicense: {src['license']}")
    print(f"\nUse it:\n  python scripts/run_benchmark.py --data {dest} "
          f"--psmiles-col {src['psmiles_col']} --target-col density --units g/cm3")


if __name__ == "__main__":
    main()
