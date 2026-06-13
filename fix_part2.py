# FIX PART 2: MIDAS with bounded optimization and grid search (Issue 3)

def almon_weights(K, theta1, theta2):
    k = np.arange(1, K + 1, dtype=float)
    w = np.exp(theta1 * k + theta2 * k ** 2)
    w_sum = w.sum()
    if w_sum == 0 or np.isnan(w_sum) or np.isinf(w_sum):
        return np.ones(K) / K
    return w / w_sum

def build_midas_lag_matrix(monthly_series, quarterly_dates, m):
    X = np.zeros((len(quarterly_dates), m))
    for t, qdate in enumerate(quarterly_dates):
        for j in range(m):
            target_date = qdate - pd.DateOffset(months=j)
            if target_date in monthly_series.index:
                X[t, j] = monthly_series.loc[target_date]
            else:
                nearest = monthly_series.index[np.argmin(np.abs(monthly_series.index - target_date))]
                X[t, j] = monthly_series.loc[nearest]
    return X

def midas_objective(params, y, X_list, m_list, y_lag):
    alpha, rho = params[0], params[1]
    idx = 2
    y_pred = alpha + rho * y_lag
    for X_i, m_i in zip(X_list, m_list):
        w = almon_weights(m_i, params[idx+1], params[idx+2])
        y_pred = y_pred + params[idx] * (X_i @ w)
        idx += 3
    return np.sum((y - y_pred) ** 2)

def estimate_midas_fixed(y, hf_pred_names, hf_monthly, q_dates, best_m):
    """MIDAS with parameter bounds + multi-start grid search to prevent degeneracy."""
    n = len(y)
    y_lag = np.concatenate([[0], y[:-1]])
    X_list, m_list = [], []
    for pn in hf_pred_names:
        X_list.append(build_midas_lag_matrix(hf_monthly[pn], q_dates, best_m[pn]))
        m_list.append(best_m[pn])

    n_preds = len(hf_pred_names)
    # Bounds: alpha unbounded, rho in [-1,1], beta unbounded,
    # theta1 in [-2, 0.5], theta2 in [-0.5, 0.1]
    bounds = [(None, None), (-1, 1)]
    for _ in range(n_preds):
        bounds.extend([(None, None), (-2.0, 0.5), (-0.5, 0.1)])

    # Grid search over starting values
    theta1_grid = [-1.0, -0.5, -0.1, 0.1]
    theta2_grid = [-0.3, -0.1, 0.05]
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
                    midas_objective, x0, args=(y, X_list, m_list, y_lag),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-12})
                if res.fun < best_ssr:
                    # Check for degeneracy
                    params_check = res.x
                    degenerate = False
                    idx_c = 2
                    for i in range(n_preds):
                        t1 = params_check[idx_c + 1]
                        t2 = params_check[idx_c + 2]
                        w = almon_weights(m_list[i], t1, t2)
                        if np.max(w) > 0.90:
                            degenerate = True
                        idx_c += 3
                    if not degenerate:
                        best_ssr = res.fun
                        best_result = res
            except:
                pass

    # If all grid results were degenerate, use best anyway with random restarts
    if best_result is None:
        print("  ⚠ All grid starts degenerate, trying random restarts...")
        for attempt in range(10):
            x0 = [0.0, 0.3]
            for _ in range(n_preds):
                x0.extend([0.1, np.random.uniform(-1.5, 0.3), np.random.uniform(-0.4, 0.05)])
            x0 = np.array(x0)
            try:
                res = optimize.minimize(
                    midas_objective, x0, args=(y, X_list, m_list, y_lag),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-12})
                if res.fun < best_ssr:
                    best_ssr = res.fun
                    best_result = res
            except:
                pass

    if best_result is None:
        # Final fallback: use original unbounded
        x0 = np.array([0.0, 0.3] + [0.1, -0.3, -0.05] * n_preds)
        best_result = optimize.minimize(
            midas_objective, x0, args=(y, X_list, m_list, y_lag),
            method='Nelder-Mead', options={'maxiter': 10000})

    params = best_result.x
    y_pred = params[0] + params[1] * y_lag
    idx = 2
    weight_profiles = {}
    for i, pn in enumerate(hf_pred_names):
        beta_i, t1_i, t2_i = params[idx], params[idx+1], params[idx+2]
        w = almon_weights(m_list[i], t1_i, t2_i)
        y_pred += beta_i * (X_list[i] @ w)
        weight_profiles[pn] = {'weights': w, 'beta': beta_i, 'theta1': t1_i, 'theta2': t2_i}
        idx += 3
    resid = y - y_pred
    return params, y_pred, resid, best_result, weight_profiles, X_list, m_list
