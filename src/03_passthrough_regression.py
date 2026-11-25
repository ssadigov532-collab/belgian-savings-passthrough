"""
Step 3b - ANALYSIS 2: how much of the ECB's moves reaches customers?

Two complementary measures, both different from the ODE in analysis 1
(which imposes one smooth exponential adjustment path):

(A) Cumulative pass-through ("beta") per phase of the 2022-2025 cycle
        beta = change in bank rate / change in ECB rate
    hiking phase : 2022-06 -> 2024-05  (ECB rate -0.50 % -> 4.00 %)
    cutting phase: 2024-05 -> 2025-12  (ECB rate  4.00 % -> 2.00 %)
    beta = 1 means the full ECB move reached the customer; beta = 0.2 means
    only 20 % did. Comparing up and down shows whether banks are faster
    to pass on moves that favour them.

(B) Distributed-lag regression on the full 2003-2026 history (OLS)
        d r_t = c + sum_{j=0..12} b_j * d P_{t-j} + e_t
    d = month-on-month change. b_j is the effect on this month's bank rate of
    an ECB move j months ago; the cumulative sum  B(h) = b_0 + ... + b_h  is the
    share of an ECB move passed on after h months. No functional form is
    imposed, so B(h) is a model-free check on the ODE's exponential path.
    Standard errors: Newey-West (HAC, 12 lags) because monthly rate changes are
    autocorrelated.

Outputs
-------
  results/phase_betas.csv
  results/dl_cumulative_passthrough.csv
  figures/fig2_country_comparison.pdf  (+ .png)
  figures/fig3_cumulative_passthrough.pdf (+ .png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "rates_monthly.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COUNTRIES = {"BE": "Belgium", "DE": "Germany", "FR": "France", "NL": "Netherlands", "U2": "Euro area"}
PRODUCTS = {"savings": "Savings accounts", "overnight": "Current accounts",
            "term1y": "Term deposits (<=1y)", "mortgage": "Mortgages"}
PHASES = {"hiking": ("2022-06", "2024-05"), "cutting": ("2024-05", "2025-12")}
LAGS = 12


def phase_betas(df):
    rows = []
    for c, cname in COUNTRIES.items():
        for prod, pname in PRODUCTS.items():
            col = f"{c}_{prod}"
            row = {"country": cname, "product": pname, "series": col}
            for ph, (a, b) in PHASES.items():
                d_bank = df.loc[b, col] - df.loc[a, col]
                d_ecb = df.loc[b, "ECB_DFR"] - df.loc[a, "ECB_DFR"]
                row[f"{ph}_change_bank_pp"] = d_bank
                row[f"{ph}_change_ecb_pp"] = d_ecb
                row[f"beta_{ph}"] = d_bank / d_ecb
            rows.append(row)
    return pd.DataFrame(rows)


def distributed_lag(df, col):
    d = df[[col, "ECB_DFR"]].dropna().diff()
    X = pd.concat({f"dP_lag{j}": d["ECB_DFR"].shift(j) for j in range(LAGS + 1)}, axis=1)
    data = pd.concat([d[col].rename("dr"), X], axis=1).dropna()
    model = sm.OLS(data["dr"], sm.add_constant(data.drop(columns="dr")))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    b = res.params.drop("const").to_numpy()
    cov = res.cov_params().drop(index="const", columns="const").to_numpy()
    out = []
    for h in range(LAGS + 1):
        w = np.r_[np.ones(h + 1), np.zeros(LAGS - h)]   # sum of b_0..b_h
        cum = w @ b
        se = np.sqrt(w @ cov @ w)
        out.append({"series": col, "horizon_months": h, "cumulative_passthrough": cum,
                    "se": se, "ci_low": cum - 1.96 * se, "ci_high": cum + 1.96 * se,
                    "n_obs": int(res.nobs), "sample_start": data.index[0], "R2": res.rsquared})
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, index_col="month")

    # (A) phase betas
    betas = phase_betas(df)
    betas.to_csv(RESULTS / "phase_betas.csv", index=False, float_format="%.4f")
    print("Pass-through by phase (share of the ECB move that reached customers):")
    print(betas[["series", "beta_hiking", "beta_cutting"]].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    # (B) distributed-lag regressions
    rows = []
    for c in COUNTRIES:
        for prod in PRODUCTS:
            rows += distributed_lag(df, f"{c}_{prod}")
    dl = pd.DataFrame(rows)
    dl.to_csv(RESULTS / "dl_cumulative_passthrough.csv", index=False, float_format="%.4f")
    print("\nCumulative pass-through after 12 months (distributed-lag OLS, full sample):")
    print(dl[dl.horizon_months == LAGS][["series", "cumulative_passthrough", "ci_low", "ci_high", "n_obs"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    ode = pd.read_csv(RESULTS / "ode_parameters.csv")
    figure_country_comparison(betas, ode)
    figure_cumulative(dl)


def figure_country_comparison(betas, ode):
    """Figure 2: savings-account pass-through by country (hiking vs cutting) + ODE half-life."""
    order = ["BE", "DE", "FR", "NL", "U2"]
    names = [COUNTRIES[c] for c in order]
    sav = betas.set_index("series").loc[[f"{c}_savings" for c in order]]
    mort = betas.set_index("series").loc[[f"{c}_mortgage" for c in order]]
    o = ode.set_index("series").loc[[f"{c}_savings" for c in order]]
    hl = o["half_life_months"]
    hl_labels = ["< 1 m" if fast else f"{v:.1f} m" for v, fast in zip(hl, o["adjusts_within_month"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(order))
    w = 0.27
    bars = [
        (sav["beta_hiking"], "Savings — hiking 2022-24", "#1e8449"),
        (sav["beta_cutting"], "Savings — cutting 2024-25", "#82e0aa"),
        (mort["beta_hiking"], "Mortgages — hiking 2022-24", "#c0392b"),
    ]
    for i, (vals, lab, col) in enumerate(bars):
        bb = ax1.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        ax1.bar_label(bb, fmt="%.2f", fontsize=8, padding=2)
    ax1.set_xticks(x, names)
    ax1.set_ylabel("Pass-through β  (Δ bank rate / Δ ECB rate)")
    ax1.set_title("Share of the ECB's move that reached customers", fontsize=11)
    ax1.axhline(1, color="grey", ls="--", lw=0.8)
    ax1.text(-0.45, 1.01, "full pass-through", ha="left", va="bottom", fontsize=8, color="grey")
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))

    cols = ["#1e8449" if c == "BE" else "#95a5a6" for c in order]
    hb = ax2.barh(names[::-1], hl.to_numpy()[::-1], color=cols[::-1])
    ax2.bar_label(hb, labels=hl_labels[::-1], fontsize=8.5, padding=3)
    ax2.set_xlabel("Half-life of savings rate adjustment (months, ODE)")
    ax2.set_title("How fast savings rates adjust", fontsize=11)
    ax2.set_xlim(0, max(hl) * 1.25)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is ax1 else "x", alpha=0.25)
    fig.text(0.01, 0.01, "Source: ECB Data Portal. France's savings rate is mostly the state-set Livret A rate.",
             fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "fig2_country_comparison.pdf")
    fig.savefig(FIGURES / "fig2_country_comparison.png", dpi=160)
    plt.close(fig)


def figure_cumulative(dl):
    """Figure 3: cumulative pass-through B(h) for Belgian products (with 95 % CI) vs euro-area savings."""
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "BE_term1y": ("Belgium — term deposits (≤1y)", "#2874a6", "-"),
        "BE_mortgage": ("Belgium — mortgages", "#c0392b", "-"),
        "BE_savings": ("Belgium — savings accounts", "#1e8449", "-"),
        "U2_savings": ("Euro area — savings accounts", "#1e8449", "--"),
        "BE_overnight": ("Belgium — current accounts", "#7f8c8d", "-"),
    }
    for s, (lab, col, ls) in styles.items():
        d = dl[dl.series == s]
        ax.plot(d.horizon_months, d.cumulative_passthrough, ls, color=col, lw=2.2, marker="o", ms=3.5, label=lab)
        if ls == "-":
            ax.fill_between(d.horizon_months, d.ci_low, d.ci_high, color=col, alpha=0.12)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Months after an ECB rate change")
    ax.set_ylabel("Cumulative pass-through  B(h) = b₀ + … + b_h")
    ax.set_title("After an ECB move of 1 pp, how much has the bank rate moved?\n"
                 "Distributed-lag regression, monthly data 2003–2026 (shaded: 95 % CI)", fontsize=11)
    ax.set_xticks(range(0, LAGS + 1, 2))
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_cumulative_passthrough.pdf")
    fig.savefig(FIGURES / "fig3_cumulative_passthrough.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
