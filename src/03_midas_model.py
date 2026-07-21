"""
03_midas_model.py — MIDAS Regression with Exponential Almon Weights
====================================================================

This module implements Mixed-Data Sampling (MIDAS) regression for nowcasting
Kazakhstan's quarterly GDP growth using higher-frequency monthly predictors.

Key specifications (from legacy models.py and fix_part2.py):
    - 3 high-frequency predictors:
        IPI       (m = 6 monthly lags)
        Freight   (m = 9 monthly lags)
        STEI      (m = 6 monthly lags)
    - Exponential Almon weighting: w(k) ∝ exp(θ₁·k + θ₂·k²)
    - Parameter bounds: θ₁ ∈ [-2, 0.5],  θ₂ ∈ [-0.5, 0.1]
    - 49-point grid search (7 × 7 starting values) + Nelder-Mead refinement
    - Degeneracy guard: reject solutions where max(w) > 0.90
    - AR(1) component: y_t = α + ρ·y_{t-1} + Σ βᵢ·w'X_i + ε_t

Functions:
    almon_weights()      — Compute normalised Exponential Almon weights
    build_midas_lag_matrix() — Construct the HF lag matrix for one predictor
    midas_objective()    — Sum of squared residuals for optimisation
    estimate_midas()     — Full estimation with grid search + bounded L-BFGS-B
    rolling_midas()      — Recursive pseudo-OOS evaluation (Q1 2021 – Q4 2025)

Outputs saved to data/results/:
    rolling_forecasts.csv           — adds midas_nowcast, midas_backcast columns
    weights_{ipi,freight,stei}.csv  — estimated Almon weight profiles

Usage:
    python src/03_midas_model.py

Dependencies:
    pandas, numpy, scipy
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_quarterly.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean", "clean_quarterly.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

# ============================================================
# PREDICTOR SPECIFICATION
# ============================================================
# Each predictor: (column name in raw data, aggregation type, optimal m)
PREDICTORS = {
    "ipi":     {"raw_col": "ipi_q_mean",     "agg": "mean", "m": 6},
    "freight": {"raw_col": "freight_q_sum",   "agg": "sum",  "m": 9},
    "stei":    {"raw_col": "stei_q_mean",     "agg": "mean", "m": 6},
}


# ============================================================
# EXPONENTIAL ALMON WEIGHTS
# ============================================================

def almon_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """Compute normalised Exponential Almon lag weights.

    w(k) = exp(θ₁·k + θ₂·k²) / Σ exp(θ₁·j + θ₂·j²),   k = 1, …, K

    Parameters
    ----------
    K : int
        Number of monthly lags.
    theta1 : float
        Linear decay parameter.
    theta2 : float
        Quadratic (curvature) parameter.

    Returns
    -------
    np.ndarray, shape (K,)
        Normalised weights summing to 1.
    """
    k = np.arange(1, K + 1, dtype=float)
    log_w = theta1 * k + theta2 * k ** 2
    # Numerical stabilisation: shift by max before exp
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if w_sum == 0 or np.isnan(w_sum) or np.isinf(w_sum):
        return np.ones(K) / K
    return w / w_sum


# ============================================================
# MIDAS LAG MATRIX
# ============================================================

def build_midas_lag_matrix(monthly_series: pd.Series,
                           quarterly_dates: pd.DatetimeIndex,
                           m: int) -> np.ndarray:
    """Build the MIDAS high-frequency lag matrix.

    For each quarterly date, looks back ``m`` months to collect the
    monthly predictor values.

    Parameters
    ----------
    monthly_series : pd.Series
        Monthly-frequency predictor series.
    quarterly_dates : pd.DatetimeIndex
        Quarterly observation dates (quarter-start convention).
    m : int
        Number of monthly lags to include.

    Returns
    -------
    np.ndarray, shape (n_quarters, m)
        Lag matrix where row t contains the m most recent monthly observations.
    """
    X = np.zeros((len(quarterly_dates), m))
    for t, qdate in enumerate(quarterly_dates):
        for j in range(m):
            target = qdate - pd.DateOffset(months=j)
            if target in monthly_series.index:
                X[t, j] = monthly_series.loc[target]
            else:
                nearest_idx = np.argmin(np.abs(monthly_series.index - target))
                X[t, j] = monthly_series.iloc[nearest_idx]
    return X


# ============================================================
# MIDAS OBJECTIVE
# ============================================================

def midas_objective(params: np.ndarray,
                    y: np.ndarray,
                    X_list: list[np.ndarray],
                    m_list: list[int],
                    y_lag: np.ndarray) -> float:
    """Compute the sum of squared residuals for the MIDAS model.

    Model:  y_t = α + ρ·y_{t-1} + Σᵢ βᵢ · (Xᵢ @ wᵢ) + εₜ

    Parameters
    ----------
    params : np.ndarray
        [α, ρ, β₁, θ₁₁, θ₂₁, β₂, θ₁₂, θ₂₂, …]
    y : np.ndarray
        Dependent variable (GDP growth, T observations).
    X_list : list[np.ndarray]
        HF lag matrices, one per predictor.
    m_list : list[int]
        Number of monthly lags per predictor.
    y_lag : np.ndarray
        Lagged dependent variable (for AR component).

    Returns
    -------
    float
        Sum of squared residuals.
    """
    alpha, rho = params[0], params[1]
    y_pred = alpha + rho * y_lag
    idx = 2
    for X_i, m_i in zip(X_list, m_list):
        beta_i = params[idx]
        w = almon_weights(m_i, params[idx + 1], params[idx + 2])
        y_pred = y_pred + beta_i * (X_i @ w)
        idx += 3
    return np.sum((y - y_pred) ** 2)


# ============================================================
# MIDAS ESTIMATION  (from fix_part2.py — estimate_midas_fixed)
# ============================================================

def estimate_midas(y: np.ndarray,
                   hf_pred_names: list[str],
                   hf_monthly: dict[str, pd.Series],
                   q_dates: pd.DatetimeIndex,
                   best_m: dict[str, int]):
    """Estimate MIDAS with bounded grid search and Nelder-Mead refinement.

    Implements the fix_part2 strategy:
    1. 49-point grid search over (θ₁, θ₂) starting values.
    2. L-BFGS-B optimisation with parameter bounds.
    3. Degeneracy guard: reject if any max(weight) > 0.90.
    4. Random restarts as last resort; final fallback with Nelder-Mead.

    Parameters
    ----------
    y : np.ndarray
        Quarterly GDP growth (dependent variable).
    hf_pred_names : list[str]
        Names of the HF predictors (keys into hf_monthly / best_m).
    hf_monthly : dict[str, pd.Series]
        Monthly predictor series.
    q_dates : pd.DatetimeIndex
        Quarterly dates aligned with y.
    best_m : dict[str, int]
        Number of monthly lags per predictor.

    Returns
    -------
    params : np.ndarray
        Estimated parameter vector.
    y_pred : np.ndarray
        Fitted values.
    resid : np.ndarray
        Residuals.
    result : scipy.optimize.OptimizeResult
        Optimisation result object.
    weight_profiles : dict
        Per-predictor weight profiles (weights, beta, theta1, theta2).
    X_list : list[np.ndarray]
        HF lag matrices used.
    m_list : list[int]
        Monthly lag counts used.
    """
    n = len(y)
    y_lag = np.concatenate([[0], y[:-1]])

    X_list, m_list = [], []
    for pn in hf_pred_names:
        X_list.append(build_midas_lag_matrix(hf_monthly[pn], q_dates, best_m[pn]))
        m_list.append(best_m[pn])

    n_preds = len(hf_pred_names)

    # Bounds: α unbounded, ρ ∈ [-1, 1], βᵢ unbounded,
    #         θ₁ ∈ [-2, 0.5], θ₂ ∈ [-0.5, 0.1]
    bounds = [(None, None), (-1.0, 1.0)]
    for _ in range(n_preds):
        bounds.extend([(None, None), (-2.0, 0.5), (-0.5, 0.1)])

    # 49-point grid search over starting values
    theta1_grid = np.linspace(-1.5, 0.3, 7)
    theta2_grid = np.linspace(-0.4, 0.05, 7)

    best_result = None
    best_ssr = np.inf

    for t1_init in theta1_grid:
        for t2_init in theta2_grid:
            x0 = [0.0, 0.3]
            for _ in range(n_preds):
                x0.extend([0.1, t1_init, t2_init])
            x0 = np.array(x0)

            try:
                res = optimize.minimize(
                    midas_objective, x0,
                    args=(y, X_list, m_list, y_lag),
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-12},
                )
                if res.fun < best_ssr:
                    # Degeneracy check
                    degenerate = False
                    idx_c = 2
                    for i in range(n_preds):
                        w = almon_weights(m_list[i], res.x[idx_c + 1], res.x[idx_c + 2])
                        if np.max(w) > 0.90:
                            degenerate = True
                        idx_c += 3
                    if not degenerate:
                        best_ssr = res.fun
                        best_result = res
            except Exception:
                pass

    # Random restarts if all grid starts were degenerate
    if best_result is None:
        print("       ⚠ All grid starts degenerate — trying random restarts …")
        for _ in range(10):
            x0 = [0.0, 0.3]
            for _ in range(n_preds):
                x0.extend([
                    0.1,
                    np.random.uniform(-1.5, 0.3),
                    np.random.uniform(-0.4, 0.05),
                ])
            x0 = np.array(x0)
            try:
                res = optimize.minimize(
                    midas_objective, x0,
                    args=(y, X_list, m_list, y_lag),
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-12},
                )
                if res.fun < best_ssr:
                    best_ssr = res.fun
                    best_result = res
            except Exception:
                pass

    # Final fallback: Nelder-Mead (unbounded)
    if best_result is None:
        x0 = np.array([0.0, 0.3] + [0.1, -0.3, -0.05] * n_preds)
        best_result = optimize.minimize(
            midas_objective, x0,
            args=(y, X_list, m_list, y_lag),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )

    # Extract fitted values and weight profiles
    params = best_result.x
    y_pred = params[0] + params[1] * y_lag
    idx = 2
    weight_profiles = {}
    for i, pn in enumerate(hf_pred_names):
        beta_i = params[idx]
        t1_i = params[idx + 1]
        t2_i = params[idx + 2]
        w = almon_weights(m_list[i], t1_i, t2_i)
        y_pred += beta_i * (X_list[i] @ w)
        weight_profiles[pn] = {
            "weights": w,
            "beta": beta_i,
            "theta1": t1_i,
            "theta2": t2_i,
        }
        idx += 3

    resid = y - y_pred
    return params, y_pred, resid, best_result, weight_profiles, X_list, m_list


# ============================================================
# MONTHLY SERIES RECONSTRUCTION
# ============================================================

def reconstruct_monthly_from_raw(raw_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Reconstruct monthly predictor series from raw quarterly aggregates.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw quarterly data with columns matching PREDICTORS spec.

    Returns
    -------
    dict[str, pd.Series]
        Monthly series keyed by predictor name.
    """
    monthly_idx = pd.date_range("2010-01", "2025-12", freq="MS")
    hf_monthly = {}

    for pred_name, spec in PREDICTORS.items():
        col = spec["raw_col"]
        s = pd.Series(np.nan, index=monthly_idx, name=pred_name)
        for qdate, row in raw_df.iterrows():
            val = row[col]
            month0 = pd.Timestamp(qdate)
            for offset in range(3):
                m = month0 + pd.DateOffset(months=offset)
                if m in s.index:
                    if spec["agg"] == "sum":
                        s.loc[m] = val / 3.0
                    else:
                        s.loc[m] = val
        hf_monthly[pred_name] = s.ffill().bfill()

    return hf_monthly


# ============================================================
# ROLLING EVALUATION
# ============================================================

def rolling_midas(df: pd.DataFrame,
                  hf_monthly: dict[str, pd.Series],
                  best_m: dict[str, int],
                  train_end: str = "2020-10-01") -> pd.DataFrame:
    """Run recursive rolling MIDAS nowcast/backcast evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Clean quarterly matrix.
    hf_monthly : dict[str, pd.Series]
        Monthly predictor series.
    best_m : dict[str, int]
        Optimal lag counts per predictor.
    train_end : str
        Last date of the initial training window.

    Returns
    -------
    pd.DataFrame
        MIDAS rolling forecast results indexed by date.
    """
    hf_pred_names = list(PREDICTORS.keys())
    eval_start = df.index.get_loc(df.index[df.index > train_end][0])
    eval_dates = df.index[eval_start:]
    print(f"[MIDAS] Rolling evaluation: {eval_dates[0]} → {eval_dates[-1]}  ({len(eval_dates)} quarters)")

    rows = []
    for t_idx in range(eval_start, len(df)):
        t_date = df.index[t_idx]
        train = df.iloc[:t_idx]
        if len(train) < 15:
            continue

        actual = df["gdp_dlsa"].iloc[t_idx]
        actual_tm1 = df["gdp_dlsa"].iloc[t_idx - 1] if t_idx > 0 else np.nan

        try:
            train_y = train["gdp_dlsa"].values
            train_dates = train.index

            params_t, fitted_t, _, res_t, _, X_list_t, m_list_t = estimate_midas(
                train_y, hf_pred_names, hf_monthly, train_dates, best_m
            )

            # 1-step-ahead nowcast
            y_lag_new = train_y[-1]
            midas_nowcast = params_t[0] + params_t[1] * y_lag_new

            idx_p = 2
            for i, pn in enumerate(hf_pred_names):
                beta_i = params_t[idx_p]
                t1_i = params_t[idx_p + 1]
                t2_i = params_t[idx_p + 2]
                m_i = best_m[pn]
                w_i = almon_weights(m_i, t1_i, t2_i)
                x_hf = build_midas_lag_matrix(
                    hf_monthly[pn], pd.DatetimeIndex([t_date]), m_i
                )
                midas_nowcast += beta_i * (x_hf @ w_i)[0]
                idx_p += 3

            # Backcast: last fitted value
            midas_backcast = fitted_t[-1] if len(fitted_t) > 0 else np.nan

        except Exception as exc:
            midas_nowcast = midas_backcast = np.nan
            print(f"       ⚠ MIDAS failed at {t_date}: {str(exc)[:60]}")

        rows.append({
            "date": t_date,
            "actual": actual,
            "midas_nowcast": midas_nowcast,
            "midas_backcast": midas_backcast,
            "actual_tm1": actual_tm1,
        })
        print(f"       [{t_date.strftime('%Y')}-Q{(t_date.month - 1) // 3 + 1}]  "
              f"Actual={actual:.4f}  MIDAS={midas_nowcast:.4f}")

    return pd.DataFrame(rows).set_index("date")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the full MIDAS estimation and rolling evaluation pipeline."""
    print("=" * 70)
    print("  MIDAS MODEL — Exponential Almon Weights")
    print("=" * 70)

    # Load data
    print("[MIDAS] Loading data …")
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
    raw_df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    print(f"       Clean shape: {df.shape}  |  Raw shape: {raw_df.shape}")

    # Reconstruct monthly series from raw quarterly
    print("[MIDAS] Reconstructing monthly predictors …")
    hf_monthly = reconstruct_monthly_from_raw(raw_df)
    for pn, s in hf_monthly.items():
        print(f"       {pn}: {len(s)} months, {s.index[0]} → {s.index[-1]}")

    best_m = {pn: spec["m"] for pn, spec in PREDICTORS.items()}

    # Full-sample estimation (for weight profiles)
    print("[MIDAS] Full-sample estimation …")
    hf_pred_names = list(PREDICTORS.keys())
    y_full = df["gdp_dlsa"].values
    q_dates = df.index

    params, y_fitted, resid, result, weight_profiles, X_list, m_list = estimate_midas(
        y_full, hf_pred_names, hf_monthly, q_dates, best_m
    )
    print(f"       Converged: {result.success}  |  SSR: {result.fun:.6f}")
    print(f"       Parameters: {params}")

    # Identify dominant predictor
    dominant = max(weight_profiles, key=lambda k: abs(weight_profiles[k]["beta"]))
    print(f"       Dominant predictor: {dominant}  |β|={abs(weight_profiles[dominant]['beta']):.4f}")

    # Save weight profiles
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for pn, wp in weight_profiles.items():
        wp_df = pd.DataFrame({
            "lag": range(1, len(wp["weights"]) + 1),
            "weight": wp["weights"],
        })
        wp_path = os.path.join(RESULTS_DIR, f"weights_{pn}.csv")
        wp_df.to_csv(wp_path, index=False)
        print(f"       Weights saved → {wp_path}")

    # Rolling evaluation
    print("[MIDAS] Starting rolling evaluation …")
    rolling_df = rolling_midas(df, hf_monthly, best_m)

    # Merge with existing BVAR rolling forecasts if available
    roll_path = os.path.join(RESULTS_DIR, "rolling_forecasts.csv")
    if os.path.exists(roll_path):
        bvar_df = pd.read_csv(roll_path, index_col=0, parse_dates=True)
        # Add MIDAS columns
        for col in ["midas_nowcast", "midas_backcast"]:
            if col in rolling_df.columns:
                bvar_df[col] = rolling_df[col]
        bvar_df.to_csv(roll_path)
        print(f"       Merged MIDAS results into {roll_path}")
    else:
        rolling_df.to_csv(roll_path)
        print(f"       Rolling forecasts saved → {roll_path}")

    print("=" * 70)
    print("  MIDAS STAGE COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
