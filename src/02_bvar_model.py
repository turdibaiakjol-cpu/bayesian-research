"""
02_bvar_model.py — Minnesota Prior Bayesian VAR for Kazakhstan GDP Nowcasting
==============================================================================

This module implements a K-variable, p-lag Bayesian Vector Autoregression with
the Minnesota (Litterman) prior for nowcasting Kazakhstan's GDP growth rate.

Key specifications (from legacy models.py and fix_part1.py):
    - K = 7 endogenous variables (core block + Granger-selected)
    - p = 2 lags
    - λ₁ = 0.1  (overall tightness)
    - λ₂ = 0.5  (cross-variable shrinkage relative to own lags)
    - λ₃ = 1.0  (harmonic lag decay exponent)
    - White-noise prior (prior mean = 0) for stationary log-differenced data
    - Ridge regularisation when condition number > 1e12

Functions:
    estimate_bvar()  — Posterior mean & variance via equation-by-equation updating
    compute_irf()    — Structural Cholesky IRFs over an arbitrary horizon
    rolling_bvar()   — Recursive pseudo-OOS evaluation loop (Q1 2021 – Q4 2025)

Outputs saved to data/results/:
    rolling_forecasts.csv   — date, actual, bvar_nowcast, bvar_backcast, CIs
    irf_data.csv            — IRF point estimates + 68 % bootstrap bands

Usage:
    python src/02_bvar_model.py

Dependencies:
    pandas, numpy, statsmodels
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean", "clean_quarterly.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

# ============================================================
# HYPERPARAMETERS
# ============================================================
LAMBDA1 = 0.1   # Overall tightness
LAMBDA2 = 0.5   # Cross-variable shrinkage
LAMBDA3 = 1.0   # Harmonic lag decay
P_LAGS = 2      # Number of lags
N_BOOT = 500    # Bootstrap replications for IRF bands
IRF_HORIZON = 12  # Quarters


# ============================================================
# VARIABLE SELECTION
# ============================================================

# Core endogenous block (must always be included)
CORE_VARS = ["gdp_dlsa", "ipi_dl", "freight_dl", "stei_dl", "brent_dl", "credit_dl"]

# Structural IRF ordering (Cholesky: most exogenous first)
IRF_ORDER = ["brent_dl", "usdkzt_dl", "ipi_dl", "freight_dl", "credit_dl", "gdp_dlsa"]


def select_endogenous_vars(df: pd.DataFrame,
                           core: list[str] = CORE_VARS,
                           max_total: int = 9) -> list[str]:
    """Select endogenous variables via Granger-causality pre-screening.

    Starts with the core block and adds up to ``max_total - len(core)``
    additional variables that Granger-cause ``gdp_dlsa`` at the 10 % level.

    Parameters
    ----------
    df : pd.DataFrame
        Clean stationary quarterly matrix.
    core : list[str]
        Core variables always included.
    max_total : int
        Maximum number of endogenous variables (caps for DOF).

    Returns
    -------
    list[str]
        Ordered list of endogenous variable names.
    """
    print("[BVAR] Granger-causality pre-screening (10 % level)")
    candidates = [c for c in df.columns if c not in core]
    passed = []

    for cand in candidates:
        try:
            test_df = df[["gdp_dlsa", cand]].dropna()
            if len(test_df) < 20:
                continue
            result = grangercausalitytests(test_df, maxlag=4, verbose=False)
            min_p = min(result[lag][0]["ssr_ftest"][1] for lag in result)
            tag = "✓ INCLUDE" if min_p < 0.10 else "✗ exclude"
            print(f"       {cand:20s} → gdp_dlsa: p={min_p:.4f}  [{tag}]")
            if min_p < 0.10:
                passed.append(cand)
        except Exception as exc:
            print(f"       {cand:20s} → error: {str(exc)[:50]}")

    endo = list(core) + passed[: max_total - len(core)]

    # Ensure usdkzt_dl is present for IRF ordering
    if "usdkzt_dl" not in endo and "usdkzt_dl" in df.columns:
        endo.insert(1, "usdkzt_dl")

    print(f"       Final endogenous block ({len(endo)}): {endo}")
    return endo


# ============================================================
# B-VAR ESTIMATION  (from fix_part1.py — estimate_bvar_fixed)
# ============================================================

def estimate_bvar(data: pd.DataFrame,
                  endo_vars: list[str],
                  p: int = P_LAGS,
                  lambda1: float = LAMBDA1,
                  lambda2: float = LAMBDA2,
                  lambda3: float = LAMBDA3):
    """Estimate B-VAR with Minnesota prior and ridge regularisation.

    Uses a **white-noise prior** (prior mean = 0 on all coefficients) because
    the data are already log-differenced and therefore stationary.  A random-walk
    prior (mean = 1 on own lag-1) would be inappropriate and propagate extreme
    observations such as the COVID-era collapse.

    Parameters
    ----------
    data : pd.DataFrame
        Stationary quarterly data containing at least ``endo_vars`` columns.
    endo_vars : list[str]
        Endogenous variable names in desired ordering.
    p : int
        Number of lags.
    lambda1, lambda2, lambda3 : float
        Minnesota prior hyperparameters.

    Returns
    -------
    B_post : np.ndarray, shape (1 + K*p, K)
        Posterior coefficient matrix (row 0 = intercept).
    V_post_list : list[np.ndarray]
        Per-equation posterior variance matrices.
    Sigma_post : np.ndarray, shape (K, K)
        Posterior residual covariance.
    X : np.ndarray, shape (T-p, 1+K*p)
        Regressor matrix used in estimation.
    Y : np.ndarray, shape (T-p, K)
        Dependent variable matrix.
    resid : np.ndarray, shape (T-p, K)
        Posterior residuals.
    """
    K = len(endo_vars)
    Y_raw = data[endo_vars].values
    T = len(Y_raw)

    # Construct Y and X (intercept + p lags)
    Y = Y_raw[p:]                                      # (T-p) x K
    X = np.ones((T - p, 1))                             # intercept column
    for lag in range(1, p + 1):
        X = np.hstack([X, Y_raw[p - lag: T - lag]])     # append lagged block

    n_params = 1 + K * p

    # --- OLS as starting point ---
    B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid_ols = Y - X @ B_ols
    dof = max(T - p - n_params, 1)
    Sigma_ols = (resid_ols.T @ resid_ols) / dof
    sigma2 = np.maximum(np.diag(Sigma_ols), 1e-10)

    XtX = X.T @ X
    XtY = X.T @ Y

    # --- Prior: WHITE NOISE (mean = 0 everywhere) ---
    B_prior = np.zeros((n_params, K))
    # Do NOT set own-lag to 1.0 — data is log-differenced / stationary

    # --- Equation-by-equation posterior ---
    B_post = np.zeros((n_params, K))
    V_post_list = []

    for i in range(K):
        # Build diagonal prior variance for equation i
        v_diag = np.ones(n_params) * 100.0          # loose intercept default
        v_diag[0] = 100.0                            # intercept
        for lag in range(1, p + 1):
            for j in range(K):
                idx = 1 + (lag - 1) * K + j
                if idx < n_params:
                    if i == j:  # own variable
                        v_diag[idx] = (lambda1 / (lag ** lambda3)) ** 2
                    else:       # cross variable
                        v_diag[idx] = (
                            (lambda1 * lambda2 / (lag ** lambda3)) ** 2
                            * (sigma2[i] / sigma2[j])
                        )

        V_prior_inv_i = np.diag(1.0 / np.maximum(v_diag, 1e-12))
        precision = XtX / sigma2[i] + V_prior_inv_i

        # Ridge regularisation if ill-conditioned (Issue 1 fix)
        cond = np.linalg.cond(precision)
        if cond > 1e12:
            precision += np.eye(n_params) * 1e-6
            print(f"       ⚠ Ridge applied: eq {i} ({endo_vars[i]}), cond={cond:.1e}")

        V_post_i = np.linalg.inv(precision)
        B_post[:, i] = V_post_i @ (
            XtY[:, i] / sigma2[i] + V_prior_inv_i @ B_prior[:, i]
        )
        V_post_list.append(V_post_i)

    resid_post = Y - X @ B_post
    Sigma_post = (resid_post.T @ resid_post) / (T - p)

    return B_post, V_post_list, Sigma_post, X, Y, resid_post


# ============================================================
# IMPULSE RESPONSE FUNCTIONS  (from fix_part1.py — compute_irf_fixed)
# ============================================================

def compute_irf(B_post: np.ndarray,
                Sigma: np.ndarray,
                p: int,
                K: int,
                horizon: int,
                irf_order_indices: list[int]) -> np.ndarray:
    """Compute Cholesky structural impulse response functions.

    Parameters
    ----------
    B_post : np.ndarray, shape (1+K*p, K)
        Posterior coefficient matrix.
    Sigma : np.ndarray, shape (K, K)
        Posterior residual covariance matrix.
    p : int
        Number of lags.
    K : int
        Number of endogenous variables.
    horizon : int
        Number of quarters ahead for the IRF.
    irf_order_indices : list[int]
        Indices mapping the structural ordering into the estimation ordering.
        E.g. if estimation order is [gdp, ipi, brent, …] and IRF order puts
        brent first, the index of brent in the estimation list is placed first.

    Returns
    -------
    irf : np.ndarray, shape (horizon+1, n_irf, n_irf)
        Structural IRF array.  irf[h, i, j] is the response of variable i to
        a 1-SD structural shock in variable j at horizon h.
    """
    n_irf = len(irf_order_indices)

    # Extract VAR(1), …, VAR(p) companion sub-matrices from B_post
    A_mats = []
    for lag in range(p):
        A_l = B_post[1 + lag * K: 1 + (lag + 1) * K, :].T   # K×K
        A_mats.append(A_l)

    # Reorder to structural ordering
    A_mats_r = [A[np.ix_(irf_order_indices, irf_order_indices)] for A in A_mats]
    Sigma_r = Sigma[np.ix_(irf_order_indices, irf_order_indices)]

    # Ensure positive-definite
    eigvals = np.linalg.eigvalsh(Sigma_r)
    if np.any(eigvals <= 0):
        Sigma_r += np.eye(n_irf) * (abs(eigvals.min()) + 1e-10)

    P = np.linalg.cholesky(Sigma_r)   # Lower-triangular Cholesky factor

    # Recursive MA representation  Φ(h)
    Phi = np.zeros((horizon + 1, n_irf, n_irf))
    Phi[0] = np.eye(n_irf)
    for h in range(1, horizon + 1):
        for j in range(min(h, p)):
            Phi[h] += A_mats_r[j] @ Phi[h - 1 - j]

    # Structural IRF = Φ(h) @ P
    irf = np.zeros((horizon + 1, n_irf, n_irf))
    for h in range(horizon + 1):
        irf[h] = Phi[h] @ P

    return irf


def bootstrap_irf(B_post: np.ndarray,
                   Sigma: np.ndarray,
                   X: np.ndarray,
                   resid: np.ndarray,
                   var_data: pd.DataFrame,
                   endo_vars: list[str],
                   p: int,
                   K: int,
                   horizon: int,
                   irf_order_indices: list[int],
                   n_boot: int = N_BOOT):
    """Bootstrap 68 % confidence bands for the IRF.

    Uses a residual-resampling bootstrap: resample rows of the posterior
    residuals with replacement, reconstruct Y*, re-estimate the B-VAR,
    and re-compute the IRF.

    Parameters
    ----------
    B_post, Sigma, X, resid : np.ndarray
        Outputs from ``estimate_bvar``.
    var_data : pd.DataFrame
        Original estimation DataFrame (needed for column names).
    endo_vars : list[str]
        Endogenous variable names.
    p, K, horizon : int
        Lag order, variable count, IRF horizon.
    irf_order_indices : list[int]
        Structural ordering indices.
    n_boot : int
        Number of bootstrap replications.

    Returns
    -------
    irf_lo, irf_hi : np.ndarray
        16th and 84th percentile IRF arrays (68 % bands).
    """
    print(f"       Bootstrapping IRF ({n_boot} replications) …")
    n_irf = len(irf_order_indices)
    T_resid = resid.shape[0]

    irf_base = compute_irf(B_post, Sigma, p, K, horizon, irf_order_indices)
    irf_boot = np.zeros((n_boot, horizon + 1, n_irf, n_irf))

    for b in range(n_boot):
        boot_idx = np.random.choice(T_resid, T_resid, replace=True)
        resid_b = resid[boot_idx]

        Y_boot = X @ B_post + resid_b
        boot_data = var_data.copy()
        boot_data.iloc[p:] = Y_boot

        try:
            B_b, _, Sigma_b, _, _, _ = estimate_bvar(boot_data, endo_vars, p)
            irf_b = compute_irf(B_b, Sigma_b, p, K, horizon, irf_order_indices)
            irf_boot[b] = irf_b[:, :n_irf, :n_irf]
        except Exception:
            irf_boot[b] = irf_base[:, :n_irf, :n_irf]

    irf_lo = np.percentile(irf_boot, 16, axis=0)
    irf_hi = np.percentile(irf_boot, 84, axis=0)
    print("       Bootstrap complete ✓")
    return irf_lo, irf_hi


# ============================================================
# ROLLING PSEUDO OUT-OF-SAMPLE EVALUATION
# ============================================================

def rolling_bvar(df: pd.DataFrame,
                 endo_vars: list[str],
                 p: int = P_LAGS,
                 lambda1: float = LAMBDA1,
                 train_end: str = "2020-10-01") -> pd.DataFrame:
    """Run recursive rolling B-VAR nowcast/backcast evaluation.

    Re-estimates the model at each quarter using only data available up to t−1,
    produces a 1-step-ahead GDP nowcast and a fitted backcast for t−1.

    Parameters
    ----------
    df : pd.DataFrame
        Clean stationary matrix (full sample).
    endo_vars : list[str]
        Endogenous variable names.
    p : int
        Lag order.
    lambda1 : float
        Overall tightness.
    train_end : str
        Last date of the initial training window (inclusive).

    Returns
    -------
    pd.DataFrame
        Rolling forecast results indexed by date.
    """
    K = len(endo_vars)
    eval_start = df.index.get_loc(df.index[df.index > train_end][0])
    eval_dates = df.index[eval_start:]
    print(f"[BVAR] Rolling evaluation: {eval_dates[0]} → {eval_dates[-1]}  ({len(eval_dates)} quarters)")

    rows = []
    for t_idx in range(eval_start, len(df)):
        t_date = df.index[t_idx]
        train = df.iloc[:t_idx]
        if len(train) < 20:
            continue

        actual = df["gdp_dlsa"].iloc[t_idx]
        actual_tm1 = df["gdp_dlsa"].iloc[t_idx - 1] if t_idx > 0 else np.nan

        try:
            B_t, V_t, Sigma_t, X_t, Y_t, _ = estimate_bvar(
                train[endo_vars].dropna(), endo_vars, p, lambda1
            )

            # 1-step-ahead nowcast
            last_obs = train[endo_vars].dropna().values
            x_new = np.ones(1)
            for lag in range(p):
                if len(last_obs) > lag:
                    x_new = np.concatenate([x_new, last_obs[-(lag + 1)]])
                else:
                    x_new = np.concatenate([x_new, np.zeros(K)])
            x_new = x_new[: B_t.shape[0]]

            gdp_idx = endo_vars.index("gdp_dlsa")
            bvar_nowcast = (x_new @ B_t)[gdp_idx]

            # Backcast: fitted value at last training observation
            bvar_backcast = (X_t @ B_t)[-1, gdp_idx] if len(Y_t) > 0 else np.nan

            # Credible intervals (1 000 posterior draws)
            n_draws = 1000
            draws = np.zeros(n_draws)
            for d in range(n_draws):
                B_draw = B_t.copy()
                for eq in range(K):
                    if eq < len(V_t):
                        try:
                            B_draw[:, eq] = np.random.multivariate_normal(B_t[:, eq], V_t[eq])
                        except Exception:
                            B_draw[:, eq] = B_t[:, eq] + np.random.normal(0, 0.01, B_t.shape[0])
                draws[d] = (x_new @ B_draw[:, gdp_idx]
                            if len(x_new) == B_draw.shape[0]
                            else bvar_nowcast)

            ci68_lo, ci68_hi = np.percentile(draws, 16), np.percentile(draws, 84)
            ci90_lo, ci90_hi = np.percentile(draws, 5), np.percentile(draws, 95)

        except Exception as exc:
            bvar_nowcast = bvar_backcast = np.nan
            ci68_lo = ci68_hi = ci90_lo = ci90_hi = np.nan
            print(f"       ⚠ B-VAR failed at {t_date}: {str(exc)[:60]}")

        rows.append({
            "date": t_date,
            "actual": actual,
            "bvar_nowcast": bvar_nowcast,
            "bvar_backcast": bvar_backcast,
            "bvar_ci68_lo": ci68_lo,
            "bvar_ci68_hi": ci68_hi,
            "bvar_ci90_lo": ci90_lo,
            "bvar_ci90_hi": ci90_hi,
            "actual_tm1": actual_tm1,
        })
        print(f"       [{t_date.strftime('%Y')}-Q{(t_date.month - 1) // 3 + 1}]  "
              f"Actual={actual:.4f}  BVAR={bvar_nowcast:.4f}")

    result = pd.DataFrame(rows).set_index("date")
    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the full B-VAR estimation, IRF, and rolling evaluation pipeline."""
    print("=" * 70)
    print("  B-VAR MODEL — Minnesota Prior")
    print("=" * 70)

    # Load clean data
    print("[BVAR] Loading clean data …")
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
    print(f"       Shape: {df.shape}")

    # Variable selection
    endo_vars = select_endogenous_vars(df)
    K = len(endo_vars)
    p = P_LAGS
    var_data = df[endo_vars].dropna()
    print(f"       K={K}, p={p}, T={len(var_data)}")

    # Full-sample estimation
    print("[BVAR] Estimating full-sample model …")
    B_post, V_post, Sigma_post, X_var, Y_var, resid = estimate_bvar(
        var_data, endo_vars, p
    )
    print(f"       B_post shape: {B_post.shape}  |  Σ shape: {Sigma_post.shape}")

    # IRF
    print("[BVAR] Computing impulse response functions …")
    irf_idx = []
    for v in IRF_ORDER:
        if v in endo_vars:
            irf_idx.append(endo_vars.index(v))
    if len(irf_idx) < len(IRF_ORDER):
        irf_idx = list(range(K))

    irf_result = compute_irf(B_post, Sigma_post, p, K, IRF_HORIZON, irf_idx)
    irf_lo, irf_hi = bootstrap_irf(
        B_post, Sigma_post, X_var, resid, var_data,
        endo_vars, p, K, IRF_HORIZON, irf_idx
    )

    # Save IRF data
    gdp_idx_irf = (irf_idx.index(endo_vars.index("gdp_dlsa"))
                   if endo_vars.index("gdp_dlsa") in irf_idx else -1)
    brent_idx_irf = (irf_idx.index(endo_vars.index("brent_dl"))
                     if endo_vars.index("brent_dl") in irf_idx else 0)
    ipi_idx_irf = (irf_idx.index(endo_vars.index("ipi_dl"))
                   if endo_vars.index("ipi_dl") in irf_idx else 2)

    if gdp_idx_irf >= 0:
        irf_df = pd.DataFrame({
            "h": range(IRF_HORIZON + 1),
            "brent_gdp": irf_result[:, gdp_idx_irf, brent_idx_irf],
            "brent_lo": irf_lo[:, gdp_idx_irf, brent_idx_irf],
            "brent_hi": irf_hi[:, gdp_idx_irf, brent_idx_irf],
            "ipi_gdp": irf_result[:, gdp_idx_irf, ipi_idx_irf],
            "ipi_lo": irf_lo[:, gdp_idx_irf, ipi_idx_irf],
            "ipi_hi": irf_hi[:, gdp_idx_irf, ipi_idx_irf],
        })
        os.makedirs(RESULTS_DIR, exist_ok=True)
        irf_path = os.path.join(RESULTS_DIR, "irf_data.csv")
        irf_df.to_csv(irf_path, index=False)
        print(f"       IRF data saved → {irf_path}")

    # Rolling evaluation
    print("[BVAR] Starting rolling evaluation …")
    rolling_df = rolling_bvar(df, endo_vars, p)

    # Save rolling forecasts
    os.makedirs(RESULTS_DIR, exist_ok=True)
    roll_path = os.path.join(RESULTS_DIR, "rolling_forecasts.csv")
    rolling_df.to_csv(roll_path)
    print(f"       Rolling forecasts saved → {roll_path}")

    print("=" * 70)
    print("  B-VAR STAGE COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
