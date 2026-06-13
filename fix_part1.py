# FIX PART 1: Patched estimate_bvar with ridge regularization (Issue 1)
# and corrected IRF computation (Issue 2)

def estimate_bvar_fixed(data, endo_vars, p, lambda1, lambda2=0.5, lambda3=1.0):
    """B-VAR with Minnesota prior + ridge regularization for numerical stability.
    Uses WHITE NOISE prior (prior mean = 0) since data is stationary (log-differenced).
    The random walk prior (mean = 1 on own lag) is inappropriate for stationary series
    as it propagates extreme observations (e.g., COVID-era Q1 2021 collapse)."""
    K = len(endo_vars)
    Y_raw = data[endo_vars].values
    T = len(Y_raw)
    Y = Y_raw[p:]
    X = np.ones((T - p, 1))
    for l in range(1, p + 1):
        X = np.hstack([X, Y_raw[p - l:T - l]])
    n_params_local = 1 + K * p

    B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid_ols = Y - X @ B_ols
    dof = max(T - p - n_params_local, 1)
    Sigma_ols = (resid_ols.T @ resid_ols) / dof
    XtX = X.T @ X
    XtY = X.T @ Y

    sigma2_local = np.maximum(np.diag(Sigma_ols), 1e-10)

    B_pr = np.zeros((n_params_local, K))
    # WHITE NOISE PRIOR: prior mean = 0 for all coefficients (stationary data)
    # Do NOT set own-lag to 1.0 — data is already log-differenced
    for i in range(K):
        if 1 + i < n_params_local:
            B_pr[1 + i, i] = 0.0  # White noise prior, not random walk

    B_post = np.zeros((n_params_local, K))
    V_post_list = []

    for i in range(K):
        v_diag_i = np.ones(n_params_local) * 100.0
        v_diag_i[0] = 100.0
        for l in range(1, p + 1):
            for j in range(K):
                idx = 1 + (l - 1) * K + j
                if idx < n_params_local:
                    if i == j:
                        v_diag_i[idx] = (lambda1 / (l ** lambda3)) ** 2
                    else:
                        v_diag_i[idx] = (lambda1 * lambda2 / (l ** lambda3)) ** 2 * (
                            sigma2_local[i] / sigma2_local[j])

        v_pr_inv_i = np.diag(1.0 / np.maximum(v_diag_i, 1e-12))
        precision = XtX / sigma2_local[i] + v_pr_inv_i

        # ISSUE 1 FIX: Ridge regularization if ill-conditioned
        cond = np.linalg.cond(precision)
        if cond > 1e12:
            precision += np.eye(n_params_local) * 1e-6
            print(f"  ⚠ Ridge applied: eq {i}, cond={cond:.1e}")

        V_post_i = np.linalg.inv(precision)
        B_post[:, i] = V_post_i @ (XtY[:, i] / sigma2_local[i] + v_pr_inv_i @ B_pr[:, i])
        V_post_list.append(V_post_i)

    resid_post = Y - X @ B_post
    Sigma_post = (resid_post.T @ resid_post) / (T - p)
    return B_post, V_post_list, Sigma_post, X, Y, resid_post


def compute_irf_fixed(B_post, Sigma, p, K, horizon, irf_order_indices):
    """Corrected IRF: properly reorders variables for Cholesky decomposition."""
    n_irf = len(irf_order_indices)
    A_mats = []
    for l in range(p):
        A_l = B_post[1 + l * K:1 + (l + 1) * K, :].T
        A_mats.append(A_l)

    # Reorder everything to IRF ordering
    A_mats_r = [A[np.ix_(irf_order_indices, irf_order_indices)] for A in A_mats]
    Sigma_r = Sigma[np.ix_(irf_order_indices, irf_order_indices)]

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(Sigma_r)
    if np.any(eigvals <= 0):
        Sigma_r += np.eye(n_irf) * (abs(eigvals.min()) + 1e-10)
    P = np.linalg.cholesky(Sigma_r)

    Phi = np.zeros((horizon + 1, n_irf, n_irf))
    Phi[0] = np.eye(n_irf)
    for h in range(1, horizon + 1):
        for j in range(min(h, p)):
            Phi[h] += A_mats_r[j] @ Phi[h - 1 - j]

    irf = np.zeros((horizon + 1, n_irf, n_irf))
    for h in range(horizon + 1):
        irf[h] = Phi[h] @ P
    return irf
