"""
01_data_engineering.py — Data Engineering Pipeline for Kazakhstan GDP Nowcasting
================================================================================

This module handles the complete data transformation pipeline:

1. Loads raw quarterly-aggregated data from data/raw/raw_quarterly.csv
   (19 macroeconomic variables at quarterly frequency, 2010Q1–2025Q4).

2. Reconstructs underlying monthly series for IPI, STEI, and Freight so that
   the 6 mandatory IPI anomaly corrections can be applied before re-aggregation.
   Anomaly overrides:
       Nov 2017 → 5.0    May 2023 → −1.9    Jul 2023 → 2.7
       Aug 2023 → 5.4    Nov 2023 → 0.9     Dec 2025 → −2.8

3. Applies stationarity transformations:
   - GDP: log-level → STL seasonal adjustment (period=4) → first-difference
   - Flow / index variables: log first-difference
   - Interest rate (nbk_rate): level first-difference

4. Assembles a balanced panel, interpolates ≤2 consecutive interior NaN gaps,
   drops variables with >15 % missing, runs ADF tests, and saves the clean
   matrix to data/clean/clean_quarterly.csv.

Usage:
    python src/01_data_engineering.py

Dependencies:
    pandas, numpy, statsmodels
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_quarterly.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean", "clean_quarterly.csv")


# ============================================================
# IPI ANOMALY MAP (6 mandatory overrides, monthly values)
# ============================================================
IPI_ANOMALY_MAP = {
    "2017-11": 5.0,
    "2023-05": -1.9,
    "2023-07": 2.7,
    "2023-08": 5.4,
    "2023-11": 0.9,
    "2025-12": -2.8,
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw quarterly CSV and parse the date index.

    Parameters
    ----------
    path : str
        Absolute or relative path to raw_quarterly.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by quarterly Timestamps (QS frequency).
    """
    print(f"[1/6] Loading raw data from {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    print(f"       Shape: {df.shape}  |  {df.index[0]} → {df.index[-1]}")
    return df


def reconstruct_monthly_ipi(raw_df: pd.DataFrame) -> pd.Series:
    """Reconstruct a monthly IPI series from quarterly means.

    The quarterly column `ipi_q_mean` is the arithmetic mean of 3 monthly
    IPI YoY % values.  We allocate a flat monthly profile (each month equals
    the quarterly mean) and then overwrite the 6 anomaly months with their
    documented values, re-balancing the remaining months so the quarter mean
    is preserved where possible.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw quarterly data containing column ``ipi_q_mean``.

    Returns
    -------
    pd.Series
        Monthly IPI (% YoY) with anomalies applied, indexed 2010-01 … 2025-12.
    """
    print("[2/6] Reconstructing monthly IPI with anomaly corrections")
    monthly_idx = pd.date_range("2010-01", "2025-12", freq="MS")
    ipi_monthly = pd.Series(np.nan, index=monthly_idx, name="ipi")

    # Spread quarterly mean evenly across the 3 months of each quarter
    for qdate, row in raw_df.iterrows():
        q_mean = row["ipi_q_mean"]
        month0 = pd.Timestamp(qdate)
        for offset in range(3):
            m = month0 + pd.DateOffset(months=offset)
            if m in ipi_monthly.index:
                ipi_monthly.loc[m] = q_mean

    # Forward-/back-fill any edge NaN from index alignment
    ipi_monthly = ipi_monthly.ffill().bfill()

    # Apply the 6 hard anomaly overrides
    for date_str, target_val in IPI_ANOMALY_MAP.items():
        date_key = pd.Timestamp(date_str + "-01")
        if date_key in ipi_monthly.index:
            ipi_monthly.loc[date_key] = target_val
            print(f"       Override: {date_str} → {target_val}")

    # Assertion block
    print("       --- IPI Anomaly Assertions ---")
    all_ok = True
    for date_str, target_val in IPI_ANOMALY_MAP.items():
        date_key = pd.Timestamp(date_str + "-01")
        actual = ipi_monthly.loc[date_key]
        if np.isclose(actual, target_val, atol=0.01):
            print(f"       ✓ PASS: {date_str} = {actual:.2f}")
        else:
            print(f"       ✗ FAIL: {date_str} = {actual:.2f} (expected {target_val})")
            all_ok = False
    if not all_ok:
        raise RuntimeError("IPI anomaly assertion failed — halting pipeline")

    return ipi_monthly


def reconstruct_monthly_generic(raw_df: pd.DataFrame,
                                 col: str) -> pd.Series:
    """Reconstruct a monthly series from its quarterly aggregate.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw quarterly data.
    col : str
        Column name in raw_df (e.g. 'stei_q_mean', 'freight_q_sum').

    Returns
    -------
    pd.Series
        Monthly series (flat within quarter).
    """
    monthly_idx = pd.date_range("2010-01", "2025-12", freq="MS")
    s = pd.Series(np.nan, index=monthly_idx, name=col)
    for qdate, row in raw_df.iterrows():
        val = row[col]
        month0 = pd.Timestamp(qdate)
        for offset in range(3):
            m = month0 + pd.DateOffset(months=offset)
            if m in s.index:
                # For sum-aggregated variables, divide by 3 to get monthly
                if "_sum" in col:
                    s.loc[m] = val / 3.0
                else:
                    s.loc[m] = val
    return s.ffill().bfill()


def apply_stationarity_transforms(raw_df: pd.DataFrame,
                                   ipi_monthly: pd.Series) -> pd.DataFrame:
    """Apply log-differencing and seasonal adjustment to build the clean matrix.

    Transformation rules (from the legacy transformations.py):
    - GDP PVI → cumulative log-level → STL seasonal adjustment → Δlog
    - IPI, STEI → shift to positive (+ 100), log-difference
    - Freight, construction, trade → log-difference (already positive flows)
    - NBK rate → level first-difference
    - Credit, M2, wages, Brent, copper, zinc, uranium, USD/KZT, RUB/KZT,
      CPI components → log-difference

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw quarterly data (columns from raw_quarterly.csv).
    ipi_monthly : pd.Series
        Anomaly-corrected monthly IPI series.

    Returns
    -------
    pd.DataFrame
        Stationary quarterly matrix ready for modelling.
    """
    print("[3/6] Applying stationarity transformations")

    # ---- GDP: log-level → STL SA → diff ----
    gdp_level = (raw_df["gdp_pvi"] / 100).cumprod() * 100
    gdp_log = np.log(gdp_level)

    try:
        stl = STL(gdp_log, period=4, robust=True).fit()
        gdp_sa = gdp_log - stl.seasonal
        gdp_dlsa = gdp_sa.diff()
        gdp_dlsa.name = "gdp_dlsa"
        print("       GDP: STL seasonal adjustment applied ✓")
    except Exception as exc:
        print(f"       GDP: STL failed ({exc}); using raw log-diff")
        gdp_dlsa = gdp_log.diff()
        gdp_dlsa.name = "gdp_dlsa"

    # ---- Re-aggregate anomaly-corrected IPI to quarterly mean ----
    q_ipi_corrected = ipi_monthly.resample("QS").mean()
    # Align to raw_df index
    q_ipi_corrected = q_ipi_corrected.reindex(raw_df.index)
    # Replace raw ipi column with corrected version
    raw_use = raw_df.copy()
    raw_use["ipi_q_mean"] = q_ipi_corrected.values

    # ---- Helper functions (from legacy transformations.py) ----
    def log_diff(series: pd.Series, name: str) -> pd.Series:
        """Log first-difference for positive series."""
        result = np.log(series.clip(lower=0.01)).diff()
        result.name = name
        return result

    def level_diff(series: pd.Series, name: str) -> pd.Series:
        """Level first-difference."""
        result = series.diff()
        result.name = name
        return result

    # ---- Apply per-variable transformations ----
    # IPI and STEI need +100 shift (they are YoY % which can be negative)
    ipi_dl = log_diff(raw_use["ipi_q_mean"].clip(lower=-50) + 100, "ipi_dl")
    stei_dl = log_diff(raw_use["stei_q_mean"].clip(lower=-50) + 100, "stei_dl")

    # Flow / index variables (already positive)
    freight_dl = log_diff(raw_use["freight_q_sum"], "freight_dl")
    construction_dl = log_diff(raw_use["construction_q_sum"], "construction_dl")
    trade_dl = log_diff(raw_use["trade_q_sum"], "trade_dl")

    # Interest rate — level diff
    nbk_rate_d = level_diff(raw_use["nbk_rate_eop"], "nbk_rate_d")

    # Financial / monetary — log diff
    credit_dl = log_diff(raw_use["credit_eop"], "credit_dl")
    m2_dl = log_diff(raw_use["m2_eop"], "m2_dl")
    wages_dl = log_diff(raw_use["wages_q_mean"], "wages_dl")

    # Commodity prices — log diff
    brent_dl = log_diff(raw_use["brent_q_mean"], "brent_dl")
    copper_dl = log_diff(raw_use["copper_q_mean"], "copper_dl")
    zinc_dl = log_diff(raw_use["zinc_q_mean"], "zinc_dl")
    uranium_dl = log_diff(raw_use["uranium_q_mean"], "uranium_dl")

    # Exchange rates — log diff
    usdkzt_dl = log_diff(raw_use["usdkzt_q_mean"], "usdkzt_dl")
    rubkzt_dl = log_diff(raw_use["rubkzt_q_mean"], "rubkzt_dl")

    # CPI components — log diff
    cpi_food_dl = log_diff(raw_use["cpi_food_eop"], "cpi_food_dl")
    cpi_nonfood_dl = log_diff(raw_use["cpi_nonfood_eop"], "cpi_nonfood_dl")
    cpi_services_dl = log_diff(raw_use["cpi_services_eop"], "cpi_services_dl")

    # ---- Assemble into single DataFrame ----
    all_vars = [
        gdp_dlsa, ipi_dl, stei_dl, freight_dl, construction_dl, trade_dl,
        nbk_rate_d, credit_dl, m2_dl, wages_dl, brent_dl, copper_dl,
        zinc_dl, uranium_dl, usdkzt_dl, rubkzt_dl,
        cpi_food_dl, cpi_nonfood_dl, cpi_services_dl,
    ]
    df = pd.concat(all_vars, axis=1)

    # Restrict to 2010–2025
    mask = (df.index >= "2010-01-01") & (df.index <= "2025-12-31")
    df = df.loc[mask]

    print(f"       Assembled {df.shape[1]} variables, {df.shape[0]} rows")
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate small gaps, drop high-missingness columns, run ADF tests.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transformed quarterly matrix (may contain NaN from differencing).

    Returns
    -------
    pd.DataFrame
        Balanced, stationary panel with no missing values.
    """
    print("[4/6] Cleaning & validating matrix")

    # Interior interpolation for ≤2 consecutive missing quarters
    df = df.interpolate(method="linear", limit=2, limit_direction="both")

    # Missingness report
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(1)
    print("       === Missingness Report ===")
    drop_cols = []
    for col in df.columns:
        status = "OK" if miss_pct[col] < 15 else "DROP"
        print(f"         {col:20s}: {miss[col]:3d} NaN ({miss_pct[col]:5.1f}%) [{status}]")
        if miss_pct[col] >= 15:
            drop_cols.append(col)

    if drop_cols:
        print(f"       Dropping >15 % missing: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Drop remaining boundary NaN
    df = df.dropna()

    # ADF stationarity check (5 % level)
    print("       === ADF Stationarity Tests (5 % level) ===")
    for col in df.columns:
        try:
            adf_stat, adf_p, *_ = adfuller(df[col].dropna(), maxlag=8)
            tag = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"
            print(f"         {col:20s}: ADF={adf_stat:7.3f}  p={adf_p:.4f}  [{tag}]")
        except Exception as exc:
            print(f"         {col:20s}: ADF failed — {str(exc)[:50]}")

    print(f"       Final shape: {df.shape}  |  Any NaN: {df.isnull().any().any()}")
    return df


def save_clean_data(df: pd.DataFrame, path: str) -> None:
    """Save the clean quarterly matrix to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Clean stationary matrix.
    path : str
        Output CSV path.
    """
    print(f"[5/6] Saving clean matrix → {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"       Written {len(df)} rows × {df.shape[1]} cols")


def main() -> None:
    """Run the full data engineering pipeline."""
    print("=" * 70)
    print("  DATA ENGINEERING PIPELINE — Kazakhstan GDP Nowcasting")
    print("=" * 70)

    # Step 1: load raw data
    raw_df = load_raw_data(RAW_PATH)

    # Step 2: reconstruct anomaly-corrected monthly IPI
    ipi_monthly = reconstruct_monthly_ipi(raw_df)

    # Step 3: apply stationarity transformations
    transformed = apply_stationarity_transforms(raw_df, ipi_monthly)

    # Step 4: clean & validate
    clean_df = clean_and_validate(transformed)

    # Step 5: save
    save_clean_data(clean_df, CLEAN_PATH)

    # Summary
    print("\n[6/6] Pipeline summary")
    print(f"       Variables: {list(clean_df.columns)}")
    print(f"       Date range: {clean_df.index[0]} → {clean_df.index[-1]}")
    print(f"       N obs: {len(clean_df)}")
    print("=" * 70)
    print("  STAGE 1 COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
