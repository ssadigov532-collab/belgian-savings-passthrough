"""
Step 1 - Download the data from the ECB Data Portal (free, no login needed).

What we download (all monthly, 2003 onwards):
  * ECB deposit facility rate (DFR): the policy rate the ECB pays banks on
    their overnight deposits. Published as a daily series of levels; we take
    the monthly average.
  * MFI interest rate statistics (MIR) for Belgium, Germany, France, the
    Netherlands and the euro area as a whole:
      - L23.D  deposits redeemable at notice up to 3 months
               (= the classic "savings account"; in Belgium the regulated
                 "gereglementeerde spaarrekening / compte d'epargne reglemente")
      - L21.A  overnight deposits (current accounts)
      - L22.F  term deposits with agreed maturity up to 1 year (new business)
      - A2C.A  loans to households for house purchase (new business)
    All rates are annualised agreed rates (%) on business with households.

Output:
  data/raw/ecb_dfr_daily.csv      raw daily policy-rate file
  data/raw/ecb_mir_monthly.csv    raw bank-rate file (long format)
  data/processed/rates_monthly.csv  tidy wide table, one row per month

Run:  python src/01_download_data.py
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
API = "https://data-api.ecb.europa.eu/service/data/"
START = "2003-01"

COUNTRIES = ["BE", "DE", "FR", "NL", "U2"]  # U2 = euro area
# (bank-balance-sheet item, maturity code) -> short readable name
PRODUCTS = {
    ("L23", "D"): "savings",    # notice up to 3 months
    ("L21", "A"): "overnight",  # current accounts
    ("L22", "F"): "term1y",     # term deposits up to 1 year
    ("A2C", "A"): "mortgage",   # house-purchase loans
}


def fetch(flow: str, key: str) -> pd.DataFrame:
    """Download one ECB SDMX query as a DataFrame."""
    url = API + flow + "/" + key
    r = requests.get(url, params={"format": "csvdata", "startPeriod": START}, timeout=120)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- 1. ECB policy rate (deposit facility rate), daily -> monthly mean ---
    dfr = fetch("FM", "D.U2.EUR.4F.KR.DFR.LEV")
    dfr[["TIME_PERIOD", "OBS_VALUE"]].to_csv(RAW / "ecb_dfr_daily.csv", index=False)
    dfr["month"] = dfr["TIME_PERIOD"].str[:7]
    dfr_m = dfr.groupby("month")["OBS_VALUE"].mean().rename("ECB_DFR")

    # --- 2. Bank interest rates (MIR) ---
    frames = []
    for (item, maturity), name in PRODUCTS.items():
        key = f"M.{'+'.join(COUNTRIES)}.B.{item}.{maturity}.R.A.2250.EUR.N"
        df = fetch("MIR", key)
        df["product"] = name
        df["country"] = df["KEY"].str.split(".").str[2]
        frames.append(df[["TIME_PERIOD", "country", "product", "OBS_VALUE", "KEY", "TITLE"]])
    mir = pd.concat(frames, ignore_index=True)
    mir.to_csv(RAW / "ecb_mir_monthly.csv", index=False)

    # --- 3. Tidy wide table: one column per country_product ---
    mir["series"] = mir["country"] + "_" + mir["product"]
    wide = mir.pivot(index="TIME_PERIOD", columns="series", values="OBS_VALUE")
    wide = wide.join(dfr_m, how="left")
    wide.index.name = "month"
    wide = wide[["ECB_DFR"] + sorted(c for c in wide.columns if c != "ECB_DFR")]
    wide.to_csv(PROCESSED / "rates_monthly.csv")

    print(f"Saved {len(wide)} months ({wide.index.min()} to {wide.index.max()}), "
          f"{wide.shape[1]} series -> {PROCESSED / 'rates_monthly.csv'}")


if __name__ == "__main__":
    main()
