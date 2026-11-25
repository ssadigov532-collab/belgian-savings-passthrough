"""
Step 3a - ANALYSIS 1: a first-order linear ODE for interest-rate pass-through.

Model
-----
Let P(t) be the ECB policy rate and r(t) a bank's customer rate. We assume the
bank rate moves towards a long-run target  r*(t) = alpha + beta * P(t)  at a
speed proportional to the gap:

        dr/dt = k * ( alpha + beta * P(t) - r(t) )          (1)

  * k     > 0  adjustment speed (per month).  Half-life = ln(2) / k months:
               the time needed to close half of a gap.
  * beta       long-run pass-through: if the ECB raises P by 1 percentage point,
               the bank rate eventually rises by beta points.
  * alpha      long-run spread (bank rate when P = 0).

(1) is a first-order, linear, non-homogeneous ODE (Chapter 3). Written as
r' + k r = k (alpha + beta P(t)), the integrating factor is e^{kt}.  If P is
constant (= P_n) during month n, the exact solution over that month is

        r(n) = r*_n + ( r(n-1) - r*_n ) * e^{-k},   r*_n = alpha + beta * P_n   (2)

so the bank rate closes a fraction (1 - e^{-k}) of the gap every month.

Estimation
----------
Starting from the first observed rate, we solve the ODE forward month by month
with (2) and choose (k, alpha, beta) to minimise the sum of squared differences
between the solved path and the observed rates (non-linear least squares,
scipy.optimize.least_squares). Standard errors come from the Jacobian.
As a check, we also integrate (1) numerically with scipy.integrate.solve_ivp
and confirm that it gives the same path as the exact formula (2).

Sample: January 2021 - latest month (the full 2022-23 hiking cycle plus the
2024-25 cutting cycle).

Outputs
-------
  results/ode_parameters.csv        k, half-life, alpha, beta, R^2 per series
  results/ode_fitted_paths.csv      observed vs model path
  figures/fig1_belgium_ode_fit.pdf  (+ .png for the website)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "rates_monthly.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

START, END = "2021-01", None
K_MAX = 5.0  # upper bound for k: half-life ln2/5 = 0.14 months, i.e. "immediate"
COUNTRIES = {"BE": "Belgium", "DE": "Germany", "FR": "France", "NL": "Netherlands", "U2": "Euro area"}
PRODUCTS = {"savings": "Savings accounts", "overnight": "Current accounts",
            "term1y": "Term deposits (<=1y)", "mortgage": "Mortgages"}


# ---------------------------------------------------------------------------
# ODE solution
# ---------------------------------------------------------------------------
def solve_exact(params, policy, r0):
    """Exact month-by-month solution (2) of the ODE (1)."""
    k, alpha, beta = params
    decay = np.exp(-k)
    r = np.empty(len(policy))
    r[0] = r0
    for n in range(1, len(policy)):
        target = alpha + beta * policy[n]
        r[n] = target + (r[n - 1] - target) * decay
    return r


def solve_numeric(params, policy, r0):
    """Same ODE, integrated numerically (Runge-Kutta) as a cross-check."""
    k, alpha, beta = params

    def rhs(t, r):
        n = min(int(np.ceil(t)), len(policy) - 1)  # P is constant within month n
        return k * (alpha + beta * policy[n] - r)

    t_eval = np.arange(len(policy))
    sol = solve_ivp(rhs, (0, len(policy) - 1), [r0], t_eval=t_eval,
                    max_step=0.05, rtol=1e-8, atol=1e-10)
    return sol.y[0]


def fit(observed, policy):
    """Non-linear least squares for (k, alpha, beta)."""
    r0 = observed[0]

    def residuals(p):
        return solve_exact(p, policy, r0) - observed

    res = least_squares(residuals, x0=[0.2, 0.0, 0.5],
                        bounds=([1e-3, -5, -0.5], [K_MAX, 10, 2]))
    fitted = solve_exact(res.x, policy, r0)
    sse = np.sum((observed - fitted) ** 2)
    sst = np.sum((observed - observed.mean()) ** 2)
    n, p = len(observed), 3
    # standard errors: sigma^2 (J'J)^-1
    sigma2 = sse / (n - p)
    try:
        cov = sigma2 * np.linalg.inv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return res.x, se, fitted, 1 - sse / sst, np.sqrt(sse / n)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, index_col="month").loc[START:END]
    policy = df["ECB_DFR"].to_numpy()

    rows, paths = [], {"month": df.index, "ECB_DFR": policy}
    for c, cname in COUNTRIES.items():
        for prod, pname in PRODUCTS.items():
            col = f"{c}_{prod}"
            obs = df[col].to_numpy()
            (k, a, b), (se_k, se_a, se_b), fitted, r2, rmse = fit(obs, policy)
            rows.append({
                "country": cname, "product": pname, "series": col,
                "k": k, "se_k": se_k,
                "half_life_months": np.log(2) / k,
                "monthly_gap_closed_pct": 100 * (1 - np.exp(-k)),
                "alpha": a, "se_alpha": se_a,
                "beta_long_run": b, "se_beta": se_b,
                "R2": r2, "RMSE_pp": rmse, "n_months": len(obs),
                # k at its upper bound = the rate adjusts within the same month;
                # the half-life is then only known to be < 1 month.
                "adjusts_within_month": bool(k >= K_MAX - 1e-6),
            })
            paths[col] = obs
            paths[col + "_model"] = fitted

    params = pd.DataFrame(rows)
    params.to_csv(RESULTS / "ode_parameters.csv", index=False, float_format="%.4f")
    pd.DataFrame(paths).to_csv(RESULTS / "ode_fitted_paths.csv", index=False, float_format="%.4f")

    # cross-check: exact formula vs numerical integration for Belgian savings
    p_be = params.loc[params.series == "BE_savings", ["k", "alpha", "beta_long_run"]].iloc[0].to_numpy()
    r0 = df["BE_savings"].iloc[0]
    diff = np.max(np.abs(solve_exact(p_be, policy, r0) - solve_numeric(p_be, policy, r0)))
    print(f"Check: max |exact - solve_ivp| for BE savings = {diff:.2e} percentage points")

    show = params[["series", "half_life_months", "adjusts_within_month", "beta_long_run", "se_beta", "R2"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    make_figure(df, paths, params)


def make_figure(df, paths, params):
    """Figure 1: Belgian rates - observed (dots) vs solved ODE (lines)."""
    months = pd.to_datetime(df.index)
    colors = {"mortgage": "#c0392b", "term1y": "#2874a6", "savings": "#1e8449"}
    labels = {"mortgage": "Mortgages", "term1y": "Term deposits (≤1y)", "savings": "Savings accounts"}

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.step(months, df["ECB_DFR"], where="mid", color="black", lw=2, label="ECB deposit facility rate P(t)")
    for prod, col in colors.items():
        s = f"BE_{prod}"
        p = params.set_index("series").loc[s]
        ax.plot(months, paths[s], "o", ms=3.5, color=col, alpha=0.55)
        hl = "< 1 month" if p.adjusts_within_month else f"{p.half_life_months:.1f} months"
        ax.plot(months, paths[s + "_model"], "-", lw=2.2, color=col,
                label=f"{labels[prod]}: half-life {hl}, long-run β = {p.beta_long_run:.2f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Interest rate (% per year)")
    ax.set_title("Belgium: bank rates vs ECB policy rate — dots = data, lines = ODE  dr/dt = k(α + βP − r)",
                 fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "Source: ECB Data Portal (MIR, FM). Household new business, monthly.",
             fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGURES / "fig1_belgium_ode_fit.pdf")
    fig.savefig(FIGURES / "fig1_belgium_ode_fit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
