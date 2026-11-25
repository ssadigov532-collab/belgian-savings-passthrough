# How slowly do Belgian banks pass ECB rate hikes on to savers?

*Project for Advanced Mathematics for Business and Technology (VUB, 2026–27).*

**Website:** _coming soon_

## Question

Between July 2022 and September 2023 the European Central Bank raised its
deposit facility rate from **−0.50 % to 4.00 %**. Mortgage rates in Belgium went up
quickly, but the rate on the Belgian savings account barely moved. This led to a
public debate in 2023, and in September 2023 the Belgian government issued a
one-year state note that raised about €22 billion.

This project asks:

> **How fast, and how far, do Belgian banks pass changes in the ECB policy rate
> on to savers? How does Belgium compare with Germany, France, the Netherlands
> and with Belgian mortgage rates?**

## Data

All data comes from the [ECB Data Portal](https://data.ecb.europa.eu/), which is
public and needs no login. It is all monthly data from January 2003 onwards.

| Series | ECB key | Meaning |
|---|---|---|
| ECB deposit facility rate | `FM.D.U2.EUR.4F.KR.DFR.LEV` | policy rate (daily, averaged per month) |
| Savings accounts | `MIR.M.{country}.B.L23.D.R.A.2250.EUR.N` | deposits redeemable at notice ≤ 3 months, households |
| Overnight deposits | `MIR.M.{country}.B.L21.A.R.A.2250.EUR.N` | current accounts, households |
| Term deposits ≤ 1y | `MIR.M.{country}.B.L22.F.R.A.2250.EUR.N` | new business, households |
| Mortgages | `MIR.M.{country}.B.A2C.A.R.A.2250.EUR.N` | loans for house purchase, new business |

The countries are `BE`, `DE`, `FR`, `NL` and `U2` (the euro area).

To download everything again, run:

```bash
pip install -r requirements.txt
python src/01_download_data.py
```

## Analyses

_In progress._

1. **ODE adjustment model.** The bank rate *r(t)* moves towards a long-run
   target set by the policy rate *P(t)*:
   *dr/dt = k · (α + β·P(t) − r(t))*.
   This is a first-order linear ODE. The speed *k* gives a half-life of
   ln 2 / k months, and *β* is the long-run share of the policy rate that
   reaches the customer.
2. **Deposit-beta regression.** This measures what fraction of the cumulative
   2022–23 hikes reached each product in each country.

## Main findings

_To be added._

## Repository structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # files exactly as downloaded from the ECB
│   └── processed/      # tidy monthly table used by the analyses
└── src/
    └── 01_download_data.py
```
