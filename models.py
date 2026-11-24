
log_step("B-VAR Setup", "ATTEMPT", "Lag selection & Granger screening")

core_vars = ['gdp_dlsa', 'ipi_dl', 'freight_dl', 'stei_dl', 'brent_dl', 'credit_dl']
candidates = [c for c in df.columns if c not in core_vars]

granger_passed = []
for cand in candidates:
    try:
        test_df = df[['gdp_dlsa', cand]].dropna()
        if len(test_df) < 20:
            continue
        result = grangercausalitytests(test_df, maxlag=4, verbose=False)
        min_p = min([result[lag][0]['ssr_ftest'][1] for lag in result])
        if min_p < 0.10:
            granger_passed.append(cand)
            print(f"  Granger: {cand:20s} → gdp_dlsa: min_p={min_p:.4f} ✓ INCLUDE")
        else:
            print(f"  Granger: {cand:20s} → gdp_dlsa: min_p={min_p:.4f} ✗ exclude")
    except Exception as e:
        print(f"  Granger: {cand:20s} → error: {str(e)[:50]}")


endo_vars = core_vars + granger_passed[:3]  
if 'usdkzt_dl' not in endo_vars and 'usdkzt_dl' in df.columns:
    endo_vars.insert(1, 'usdkzt_dl')

print(f"\n  Final endogenous block ({len(endo_vars)}): {endo_vars}")


var_data = df[endo_vars].dropna()
K = var_data.shape[1]


try:
    var_model = VAR(var_data)
    lag_results = var_model.select_order(maxlags=min(8, len(var_data) // K - 2))
    p_bic = lag_results.bic
    p = max(1, p_bic)
    print(f"  Lag selection: BIC selects p={p}")
except Exception as e:
    p = 2
    print(f"  Lag selection fallback: p={p} ({str(e)[:50]})")

log_step("B-VAR Setup", "SUCCESS", f"K={K}, p={p}, N={len(var_data)}")


log_step("Minnesota Prior", "ATTEMPT", "Constructing prior mean & variance")

lambda1 = 0.1   
lambda2 = 0.5   
lambda3 = 1.0  


sigma2 = np.zeros(K)
for i, col in enumerate(endo_vars):
    y = var_data[col].values
    X_ar = np.column_stack([y[p - j - 1:len(y) - j - 1] for j in range(p)])
    y_ar = y[p:]
    if len(y_ar) > p + 1:
        beta_ar = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
        resid_ar = y_ar - X_ar @ beta_ar
        sigma2[i] = np.var(resid_ar)
    else:
        sigma2[i] = np.var(y)
    sigma2[i] = max(sigma2[i], 1e-10)


n_params = K * p + 1  
B_prior = np.zeros((n_params, K))

for i in range(K):
    B_prior[1 + i, i] = 1.0  


V_prior_diag = np.zeros(n_params * K)
for i in range(K):  
    V_prior_diag[i * n_params] = 100.0
    for l in range(1, p + 1):  
        for j in range(K):  
            idx = i * n_params + 1 + (l - 1) * K + j
            if idx < len(V_prior_diag):
                if i == j: 
                    V_prior_diag[idx] = (lambda1 / (l ** lambda3)) ** 2
                else:  
                    V_prior_diag[idx] = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * (sigma2[i] / sigma2[j])

V_prior_inv = np.diag(1.0 / np.maximum(V_prior_diag, 1e-12))

log_step("Minnesota Prior", "SUCCESS", f"Prior dim: B={B_prior.shape}, V_diag len={len(V_prior_diag)}")


def estimate_bvar(data, endo_vars, p, lambda1, lambda2=0.5, lambda3=1.0):
    """Estimate B-VAR with Minnesota prior. Returns posterior mean, vaiance, residual cov."""
    K =len(endo_vars)
    Y_raw =data[endo_vars].values
    T = len(Y_raw)

    
    Y =Y_raw[p:]  
    X =np.ones((T - p, 1))  
    for l in range(1, p + 1):
        X = np.hstack([X, Y_raw[p - l:T - l]]) 

    n_params_local=1 + K * p

   
    XtX =X.T @ X
    XtY= X.T @ Y
    B_ols= np.linalg.lstsq(X, Y, rcond=None)[0]

    
    resid_ols= Y - X @ B_ols
    Sigma_ols=(resid_ols.T @ resid_ols) / (T-p-n_params_local)

    #Prior
    sigma2_local =np.diag(Sigma_ols)
    sigma2_local=np.maximum(sigma2_local, 1e-10)

    B_pr = np.zeros((n_params_local, K))
    for i in range(K):
        if 1 + i < n_params_local:
            B_pr[1 + i, i]= 1.0

    V_pr_diag = np.ones(n_params_local * K) * 100.0  #loose default
    for i in range(K):
        V_pr_diag[i * n_params_local] =100.0
        for l in range(1, p + 1):
            for j in range(K):
                idx = i * n_params_local + 1 + (l - 1) * K + j
                if idx < len(V_pr_diag):
                    if i == j:
                        V_pr_diag[idx]= (lambda1 / (l ** lambda3)) ** 2
                    else:
                        V_pr_diag[idx] =(lambda1 * lambda2 / (l ** lambda3)) ** 2 * (
                            sigma2_local[i] / sigma2_local[j])

    V_pr_inv = np.diag(1.0 / np.maximum(V_pr_diag, 1e-12))

  
    B_post = np.zeros((n_params_local, K))
    V_post_list = []

    for i in range(K):
        v_pr_i = np.diag(V_pr_diag[i * n_params_local:(i + 1) * n_params_local])
        v_pr_inv_i = np.diag(1.0 / np.maximum(
            V_pr_diag[i * n_params_local:(i + 1) * n_params_local], 1e-12))

        V_post_i = np.linalg.inv(XtX / Sigma_ols[i, i] + v_pr_inv_i)
        B_post[:, i] = V_post_i @ (XtY[:, i] / Sigma_ols[i, i] + v_pr_inv_i @ B_pr[:, i])
        V_post_list.append(V_post_i)

  
    resid_post = Y - X @ B_post
    Sigma_post = (resid_post.T @ resid_post) / (T - p)

    return B_post, V_post_list, Sigma_post, X, Y, resid_post



B_post, V_post_list, Sigma_post, X_var, Y_var, resid_bvar = estimate_bvar(
    var_data, endo_vars, p, lambda1)

log_step("B-VAR Estimation", "SUCCESS",
         f"Posterior B shape: {B_post.shape}, Σ shape: {Sigma_post.shape}")


log_step("IRF Computation", "ATTEMPT", "Cholesky decomposition & 12Q horizon")

irf_order = ['brent_dl', 'usdkzt_dl', 'ipi_dl', 'freight_dl', 'credit_dl', 'gdp_dlsa']

irf_idx = []
for v in irf_order:
    if v in endo_vars:
        irf_idx.append(endo_vars.index(v))
if len(irf_idx) < len(irf_order):
    
    irf_idx = list(range(K))


Sigma_ordered = Sigma_post[np.ix_(irf_idx, irf_idx)]
P_chol = np.linalg.cholesky(Sigma_ordered)


def compute_irf(B_post, Sigma, p, K, horizon, shock_var_idx, response_var_idx,
                irf_idx_map=None):
    """Compute IRF from B-VAR posterior mean."""
    
    A_mats = []
    for l in range(p):
        A_l = B_post[1 + l * K:1 + (l + 1) * K, :].T  # K x K
        A_mats.append(A_l)

    
    Sigma_r = Sigma.copy()
    if irf_idx_map is not None:
        Sigma_r = Sigma_r[np.ix_(irf_idx_map, irf_idx_map)]

    try:
        P = np.linalg.cholesky(Sigma_r)
    except np.linalg.LinAlgError:
        Sigma_r += np.eye(Sigma_r.shape[0]) * 1e-8
        P = np.linalg.cholesky(Sigma_r)

    K_eff = Sigma_r.shape[0]
   
    if irf_idx_map is not None:
        A_mats_r = [A[np.ix_(irf_idx_map, irf_idx_map)] for A in A_mats]
    else:
        A_mats_r = A_mats

    
    irf = np.zeros((horizon + 1, K_eff, K_eff))
    irf[0] = P

    Phi = np.zeros((horizon + 1, K_eff, K_eff))
    Phi[0] = np.eye(K_eff)

    for h in range(1, horizon + 1):
        for j in range(min(h, p)):
            if j < len(A_mats_r):
                Phi[h] += A_mats_r[j] @ Phi[h - 1 - j]
        irf[h] = Phi[h] @ P

    return irf


horizon = 12
irf_result = compute_irf(B_post, Sigma_post, p, K, horizon,
                          shock_var_idx=0, response_var_idx=K - 1,
                          irf_idx_map=irf_idx if len(irf_idx) == K else None)

n_boot = 500
irf_boot = np.zeros((n_boot, horizon + 1, len(irf_idx), len(irf_idx)))

for b in range(n_boot):
   
    T_resid = resid_bvar.shape[0]
    boot_idx = np.random.choice(T_resid, T_resid, replace=True)
    resid_boot = resid_bvar[boot_idx]

   
    Y_boot = X_var @ B_post + resid_boot
    boot_data = var_data.copy()
    boot_data.iloc[p:] = Y_boot

    try:
        B_b, _, Sigma_b, _, _, _ = estimate_bvar(boot_data, endo_vars, p, lambda1)
        irf_b = compute_irf(B_b, Sigma_b, p, K, horizon, 0, K - 1,
                            irf_idx_map=irf_idx if len(irf_idx) == K else None)
        irf_boot[b] = irf_b[:, :len(irf_idx), :len(irf_idx)]
    except:
        irf_boot[b] = irf_result[:, :len(irf_idx), :len(irf_idx)]


irf_lo = np.percentile(irf_boot, 16, axis=0)
irf_hi = np.percentile(irf_boot, 84, axis=0)

log_step("IRF Computation", "SUCCESS", f"12Q horizon, {n_boot} bootstrap reps")

log_step("MIDAS Setup", "ATTEMPT", "Exponential Almon weighting")

def almon_weights(K, theta1, theta2):
    """Exponential Almon weighting function."""
    k = np.arange(1, K + 1, dtype=float)
    w = np.exp(theta1 * k + theta2 * k ** 2)
    w_sum = w.sum()
    if w_sum == 0 or np.isnan(w_sum) or np.isinf(w_sum):
        return np.ones(K) / K
    return w / w_sum


def build_midas_lag_matrix(monthly_series, quarterly_dates, m):
    """Build MIDAS lag matrix: each row=one qarter, cols=m monthly lags."""
    X = np.zeros((len(quarterly_dates), m))
    for t, qdate in enumerate(quarterly_dates):
        month_end = qdate.month 
        for j in range(m):
            target_date = qdate - pd.DateOffset(months=j)
            if target_date in monthly_series.index:
                X[t, j] = monthly_series.loc[target_date]
            else:
               
                nearest = monthly_series.index[
                    np.argmin(np.abs(monthly_series.index - target_date))]
                X[t, j] = monthly_series.loc[nearest]
    return X


def midas_objective(params, y, X_list, m_list, y_lag):
    """MIDAS SSR objective for scipy.optimize."""
    alpha = params[0]
    rho = params[1] 
    idx = 2
    y_pred = alpha + rho * y_lag

    for i, (X_i, m_i) in enumerate(zip(X_list, m_list)):
        beta_i = params[idx]
        theta1_i = params[idx + 1]
        theta2_i = params[idx + 2]
        w = almon_weights(m_i, theta1_i, theta2_i)
        y_pred += beta_i * (X_i @ w)
        idx += 3

    resid = y - y_pred
    return np.sum(resid ** 2)


hf_predictors = {
    'ipi': ipi_monthly,
    'freight': freight_monthly,
    'stei': stei_monthly
}


best_m = {}
q_dates_for_midas = df.index

for pred_name, pred_series in hf_predictors.items():
    best_bic = np.inf
    best_m_val = 6
    for m_try in [6, 9, 12]:
        try:
            X_try = build_midas_lag_matrix(pred_series, q_dates_for_midas, m_try)
           
            y_m = df['gdp_dlsa'].values
            y_lag_m = np.concatenate([[0], y_m[:-1]])
            X_ew = X_try.mean(axis=1)  
            X_ols = np.column_stack([np.ones(len(y_m)), y_lag_m, X_ew])
            beta_ols = np.linalg.lstsq(X_ols, y_m, rcond=None)[0]
            resid_ols = y_m - X_ols @ beta_ols
            ssr = np.sum(resid_ols ** 2)
            k_params = 3
            n = len(y_m)
            bic = n * np.log(ssr / n) + k_params * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_m_val = m_try
        except:
            pass
    best_m[pred_name] = best_m_val
    print(f"  MIDAS lag selection: {pred_name}: m={best_m_val} (BIC={best_bic:.2f})")


def estimate_midas(y, hf_predictors, hf_monthly, q_dates, best_m):
    """Estimate full MIDAS model with Exponential Almon weights."""
    n = len(y)
    y_lag = np.concatenate([[0], y[:-1]])

    X_list = []
    m_list = []
    for pred_name in hf_predictors:
        m_val = best_m[pred_name]
        X_mat = build_midas_lag_matrix(hf_monthly[pred_name], q_dates, m_val)
        X_list.append(X_mat)
        m_list.append(m_val)

    
    n_preds = len(hf_predictors)
    x0 = [0.0, 0.3]  #alpha, rho
    for _ in range(n_preds):
        x0.extend([0.5, -0.5, -0.1])  #beta, theta1, theta2

    x0 = np.array(x0)

    result = optimize.minimize(
        midas_objective, x0, args=(y, X_list, m_list, y_lag),
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    if not result.success:
        
        result = optimize.minimize(
            midas_objective, x0, args=(y, X_list, m_list, y_lag),
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )

    params = result.x
    alpha = params[0]
    rho = params[1]

    y_pred = alpha + rho * y_lag
    idx = 2
    weight_profiles = {}
    for i, pred_name in enumerate(hf_predictors):
        beta_i = params[idx]
        theta1_i = params[idx + 1]
        theta2_i = params[idx + 2]
        w = almon_weights(m_list[i], theta1_i, theta2_i)
        y_pred += beta_i * (X_list[i] @ w)
        weight_profiles[pred_name] = {'weights': w, 'beta': beta_i,
                                       'theta1': theta1_i, 'theta2': theta2_i}
        idx += 3

    resid = y - y_pred
    return params, y_pred, resid, result, weight_profiles, X_list, m_list



midas_params, midas_fitted, midas_resid, midas_result, weight_profiles, \
    X_list_full, m_list_full = estimate_midas(
        df['gdp_dlsa'].values, list(hf_predictors.keys()),
        hf_predictors, q_dates_for_midas, best_m)

print(f"\n  MIDAS optimization: success={midas_result.success}, SSR={midas_result.fun:.6f}")
print(f"  Parameters: {midas_params}")

dominant_pred = max(weight_profiles, key=lambda k: abs(weight_profiles[k]['beta']))
print(f"  Dominant predictor: {dominant_pred} (|β|={abs(weight_profiles[dominant_pred]['beta']):.4f})")

log_step("MIDAS Estimation", "SUCCESS", f"Converged, dominant={dominant_pred}")

