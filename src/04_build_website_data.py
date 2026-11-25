"""
Step 6 - Prepare the website (docs/ folder, served by GitHub Pages).

Copies the figures into docs/figures/ and writes docs/data.json, which the
interactive chart on the website reads (observed rates + solved ODE paths +
estimated parameters for every country/product).

Run after 02_ode_model.py and 03_passthrough_regression.py.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"


def main():
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIGURES.glob("*.*"):
        shutil.copy2(f, DOCS / "figures" / f.name)

    paths = pd.read_csv(RESULTS / "ode_fitted_paths.csv")
    ode = pd.read_csv(RESULTS / "ode_parameters.csv")
    betas = pd.read_csv(RESULTS / "phase_betas.csv")
    dl = pd.read_csv(RESULTS / "dl_cumulative_passthrough.csv")
    dl12 = dl[dl.horizon_months == 12].set_index("series")

    def r(x, n=3):
        return None if pd.isna(x) else round(float(x), n)

    series = {}
    for _, p in ode.iterrows():
        s = p.series
        b = betas.set_index("series").loc[s]
        series[s] = {
            "country": p.country, "product": p["product"],
            "observed": [r(v) for v in paths[s]],
            "model": [r(v) for v in paths[s + "_model"]],
            "k": r(p.k), "half_life": r(p.half_life_months, 2),
            "within_month": bool(p.adjusts_within_month),
            "alpha": r(p.alpha), "beta": r(p.beta_long_run), "r2": r(p.R2),
            "beta_hiking": r(b.beta_hiking), "beta_cutting": r(b.beta_cutting),
            "dl12": r(dl12.loc[s, "cumulative_passthrough"]),
        }

    out = {"months": paths["month"].tolist(),
           "policy": [r(v) for v in paths["ECB_DFR"]],
           "series": series}
    (DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {DOCS / 'data.json'} and copied {len(list(FIGURES.glob('*.*')))} figure files.")


if __name__ == "__main__":
    main()
