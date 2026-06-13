# %% [markdown]
# ## Cell 6: Rolling Pseudo Out-of-Sample Evaluation

# %%
# ============================================================
# ROLLING EVALUATION: Q1 2021 – Q4 2025
# ============================================================
log_step("Rolling Evaluation", "ATTEMPT", "20 quarters, recursive re-estimation")

train_end = '2020-10-01'  # Q4 2020
eval_start_idx = df.index.get_loc(df.index[df.index > train_end][0])
eval_dates = df.index[eval_start_idx:]
n_eval = len(eval_dates)
print(f"  Training: {df.index[0]} to {train_end}")
print(f"  Evaluation: {eval_dates[0]} to {eval_dates[-1]} ({n_eval} quarters)")

results_rows = []

for t_idx in range(eval_start_idx, len(df)):
    t_date = df.index[t_idx]
    train_data = df.iloc[:t_idx]  # Data up to t-1

    if len(train_data) < 20:
        continue

    actual_t = df['gdp_dlsa'].iloc[t_idx]
    actual_tm1 = df['gdp_dlsa'].iloc[t_idx - 1] if t_idx > 0 else np.nan

    # --- B-VAR Nowcast ---
    try:
        B_t, V_t, Sigma_t, X_t, Y_t, resid_t = estimate_bvar(
            train_data[endo_vars].dropna(), endo_vars, p, lambda1)

        # Point nowcast: 1-step ahead
        last_obs = train_data[endo_vars].dropna().values
        x_new = np.ones(1)
        for l in range(p):
            if len(last_obs) > l:
                x_new = np.concatenate([x_new, last_obs[-(l + 1)]])
            else:
                x_new = np.concatenate([x_new, np.zeros(K)])
        x_new = x_new[:B_t.shape[0]]

        bvar_nowcast = x_new @ B_t
        bvar_gdp_nowcast = bvar_nowcast[endo_vars.index('gdp_dlsa')]

        # Backcast (t-1): just the fitted value at t-1
        if len(Y_t) > 0:
            bvar_backcast = (X_t @ B_t)[-1, endo_vars.index('gdp_dlsa')]
        else:
            bvar_backcast = np.nan

        # Credible intervals from posterior draws
        n_draws = 1000
        ci_draws = np.zeros(n_draws)
        gdp_idx = endo_vars.index('gdp_dlsa')
        for d in range(n_draws):
            B_draw = B_t.copy()
            for eq in range(K):
                if eq < len(V_t):
                    try:
                        B_draw[:, eq] = np.random.multivariate_normal(
                            B_t[:, eq], V_t[eq])
                    except:
                        B_draw[:, eq] = B_t[:, eq] + np.random.normal(0, 0.01, B_t.shape[0])
            ci_draws[d] = x_new @ B_draw[:, gdp_idx] if len(x_new) == B_draw.shape[0] else bvar_gdp_nowcast

        bvar_ci68_lo = np.percentile(ci_draws, 16)
        bvar_ci68_hi = np.percentile(ci_draws, 84)
        bvar_ci90_lo = np.percentile(ci_draws, 5)
        bvar_ci90_hi = np.percentile(ci_draws, 95)

    except Exception as e:
        bvar_gdp_nowcast = np.nan
        bvar_backcast = np.nan
        bvar_ci68_lo = bvar_ci68_hi = bvar_ci90_lo = bvar_ci90_hi = np.nan
        print(f"  B-VAR failed at {t_date}: {str(e)[:60]}")

    # --- MIDAS Nowcast ---
    try:
        train_y = train_data['gdp_dlsa'].values
        train_dates = train_data.index

        midas_params_t, midas_fitted_t, _, midas_res_t, _, X_list_t, m_list_t = \
            estimate_midas(train_y, list(hf_predictors.keys()),
                          hf_predictors, train_dates, best_m)

        # 1-step ahead nowcast
        y_lag_new = train_y[-1]
        alpha_t = midas_params_t[0]
        rho_t = midas_params_t[1]
        midas_nowcast = alpha_t + rho_t * y_lag_new

        idx_p = 2
        for i, pred_name in enumerate(hf_predictors):
            beta_i = midas_params_t[idx_p]
            theta1_i = midas_params_t[idx_p + 1]
            theta2_i = midas_params_t[idx_p + 2]
            m_i = best_m[pred_name]
            w_i = almon_weights(m_i, theta1_i, theta2_i)
            # Get HF data for current quarter
            x_hf = build_midas_lag_matrix(hf_predictors[pred_name],
                                          pd.DatetimeIndex([t_date]), m_i)
            midas_nowcast += beta_i * (x_hf @ w_i)[0]
            idx_p += 3

        # Backcast
        if len(midas_fitted_t) > 0:
            midas_backcast = midas_fitted_t[-1]
        else:
            midas_backcast = np.nan

    except Exception as e:
        midas_nowcast = np.nan
        midas_backcast = np.nan
        print(f"  MIDAS failed at {t_date}: {str(e)[:60]}")

    results_rows.append({
        'date': t_date, 'actual': actual_t,
        'bvar_nowcast': bvar_gdp_nowcast, 'bvar_backcast': bvar_backcast,
        'bvar_ci68_lo': bvar_ci68_lo, 'bvar_ci68_hi': bvar_ci68_hi,
        'bvar_ci90_lo': bvar_ci90_lo, 'bvar_ci90_hi': bvar_ci90_hi,
        'midas_nowcast': midas_nowcast, 'midas_backcast': midas_backcast,
        'actual_tm1': actual_tm1
    })

    print(f"  [{t_date.strftime('%Y-Q')}{(t_date.month-1)//3+1}] "
          f"Actual={actual_t:.4f}  BVAR={bvar_gdp_nowcast:.4f}  MIDAS={midas_nowcast:.4f}")

results_df = pd.DataFrame(results_rows)
results_df.set_index('date', inplace=True)

log_step("Rolling Evaluation", "SUCCESS", f"{len(results_df)} quarters evaluated")

# %% [markdown]
# ## Cell 7: Accuracy Metrics & Diebold-Mariano Test

# %%
# ============================================================
# ACCURACY METRICS
# ============================================================
log_step("Accuracy Metrics", "ATTEMPT", "RMSE, MAE, MAPE + DM test")

def compute_metrics(actual, forecast, label):
    """Compute RMSE, MAE, MAPE."""
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    a, f = actual[mask], forecast[mask]
    if len(a) == 0:
        return {'Model': label, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
    rmse = np.sqrt(np.mean((a - f) ** 2))
    mae = np.mean(np.abs(a - f))
    mape = np.mean(np.abs((a - f) / np.where(a == 0, 1e-8, a))) * 100
    return {'Model': label, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def diebold_mariano(actual, f1, f2, horizon=1):
    """Diebold-Mariano test. H0: equal predictive accuracy. f1=benchmark."""
    mask = ~(np.isnan(actual) | np.isnan(f1) | np.isnan(f2))
    a, p1, p2 = actual[mask], f1[mask], f2[mask]
    e1 = (a - p1) ** 2
    e2 = (a - p2) ** 2
    d = e1 - e2
    d_bar = np.mean(d)
    # Newey-West variance
    T = len(d)
    gamma_0 = np.var(d)
    gamma_sum = 0
    for h_lag in range(1, horizon):
        gamma_h = np.cov(d[h_lag:], d[:-h_lag])[0, 1] if len(d) > h_lag + 1 else 0
        gamma_sum += 2 * gamma_h
    var_d = (gamma_0 + gamma_sum) / T
    var_d = max(var_d, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d)
    dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, dm_pval


# Nowcast metrics
actual_vals = results_df['actual'].values
metrics_list = []

# B-VAR Nowcast
metrics_list.append(compute_metrics(actual_vals, results_df['bvar_nowcast'].values, 'B-VAR Nowcast'))
# MIDAS Nowcast
metrics_list.append(compute_metrics(actual_vals, results_df['midas_nowcast'].values, 'MIDAS Nowcast'))
# B-VAR Backcast
metrics_list.append(compute_metrics(results_df['actual_tm1'].values,
                                     results_df['bvar_backcast'].values, 'B-VAR Backcast'))
# MIDAS Backcast
metrics_list.append(compute_metrics(results_df['actual_tm1'].values,
                                     results_df['midas_backcast'].values, 'MIDAS Backcast'))

metrics_df = pd.DataFrame(metrics_list)
print("\n" + "=" * 60)
print("FORECAST ACCURACY METRICS")
print("=" * 60)
print(metrics_df.to_string(index=False))

# Diebold-Mariano Tests
dm_nowcast_stat, dm_nowcast_p = diebold_mariano(
    actual_vals, results_df['bvar_nowcast'].values, results_df['midas_nowcast'].values)
dm_backcast_stat, dm_backcast_p = diebold_mariano(
    results_df['actual_tm1'].values,
    results_df['bvar_backcast'].values, results_df['midas_backcast'].values)

print(f"\nDiebold-Mariano Test (benchmark = B-VAR):")
print(f"  Nowcast:  DM={dm_nowcast_stat:.3f}, p={dm_nowcast_p:.4f}")
print(f"  Backcast: DM={dm_backcast_stat:.3f}, p={dm_backcast_p:.4f}")

log_step("Accuracy Metrics", "SUCCESS", "All metrics computed")

# %% [markdown]
# ## Cell 8: Robustness — Minnesota Prior Sensitivity

# %%
# ============================================================
# ROBUSTNESS: λ₁ SENSITIVITY
# ============================================================
log_step("Robustness", "ATTEMPT", "λ₁ grid: [0.10, 0.15, 0.20]")

lambda_grid = [0.10, 0.15, 0.20]
robustness_rows = []
baseline_rmse = metrics_df.loc[metrics_df['Model'] == 'B-VAR Nowcast', 'RMSE'].values[0]

for lam1 in lambda_grid:
    rob_forecasts = []
    rob_actuals = []

    for t_idx in range(eval_start_idx, len(df)):
        train_data = df.iloc[:t_idx]
        if len(train_data) < 20:
            continue
        try:
            B_rob, V_rob, Sigma_rob, X_rob, Y_rob, _ = estimate_bvar(
                train_data[endo_vars].dropna(), endo_vars, p, lam1)
            last_obs = train_data[endo_vars].dropna().values
            x_new = np.ones(1)
            for l in range(p):
                if len(last_obs) > l:
                    x_new = np.concatenate([x_new, last_obs[-(l + 1)]])
                else:
                    x_new = np.concatenate([x_new, np.zeros(K)])
            x_new = x_new[:B_rob.shape[0]]
            fc = x_new @ B_rob[:, endo_vars.index('gdp_dlsa')]
            rob_forecasts.append(fc)
            rob_actuals.append(df['gdp_dlsa'].iloc[t_idx])
        except:
            pass

    rob_actuals = np.array(rob_actuals)
    rob_forecasts = np.array(rob_forecasts)
    mask = ~(np.isnan(rob_actuals) | np.isnan(rob_forecasts))
    if mask.sum() > 0:
        rmse_rob = np.sqrt(np.mean((rob_actuals[mask] - rob_forecasts[mask]) ** 2))
    else:
        rmse_rob = np.nan

    pct_change = ((rmse_rob - baseline_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else np.nan
    robustness_rows.append({
        'lambda1': lam1, 'RMSE': rmse_rob, 'pct_change_vs_baseline': pct_change
    })
    print(f"  λ₁={lam1:.2f}: RMSE={rmse_rob:.6f} (Δ={pct_change:+.1f}%)")

robustness_df = pd.DataFrame(robustness_rows)
log_step("Robustness", "SUCCESS", "3 λ₁ values tested")

# ============================================================
# EXPORT ALL RESULTS TO EXCEL
# ============================================================
log_step("Excel Export", "ATTEMPT", "Saving all results")

excel_results_path = os.path.join(DATA_DIR, 'model_results.xlsx')
with pd.ExcelWriter(excel_results_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Rolling_Forecasts')
    metrics_df.to_excel(writer, sheet_name='Accuracy_Metrics', index=False)
    robustness_df.to_excel(writer, sheet_name='Robustness_Lambda', index=False)

    # DM test results
    dm_df = pd.DataFrame([
        {'Horizon': 'Nowcast', 'DM_stat': dm_nowcast_stat, 'p_value': dm_nowcast_p},
        {'Horizon': 'Backcast', 'DM_stat': dm_backcast_stat, 'p_value': dm_backcast_p}
    ])
    dm_df.to_excel(writer, sheet_name='DM_Test', index=False)

    # IRF data
    gdp_idx_in_irf = irf_idx.index(endo_vars.index('gdp_dlsa')) if endo_vars.index('gdp_dlsa') in irf_idx else -1
    brent_idx_in_irf = irf_idx.index(endo_vars.index('brent_dl')) if endo_vars.index('brent_dl') in irf_idx else 0
    ipi_idx_in_irf = irf_idx.index(endo_vars.index('ipi_dl')) if endo_vars.index('ipi_dl') in irf_idx else 2

    if gdp_idx_in_irf >= 0:
        irf_brent_gdp = irf_result[:, gdp_idx_in_irf, brent_idx_in_irf]
        irf_ipi_gdp = irf_result[:, gdp_idx_in_irf, ipi_idx_in_irf]
        irf_df = pd.DataFrame({
            'horizon': range(horizon + 1),
            'irf_brent_to_gdp': irf_brent_gdp,
            'irf_brent_lo68': irf_lo[:, gdp_idx_in_irf, brent_idx_in_irf],
            'irf_brent_hi68': irf_hi[:, gdp_idx_in_irf, brent_idx_in_irf],
            'irf_ipi_to_gdp': irf_ipi_gdp,
            'irf_ipi_lo68': irf_lo[:, gdp_idx_in_irf, ipi_idx_in_irf],
            'irf_ipi_hi68': irf_hi[:, gdp_idx_in_irf, ipi_idx_in_irf],
        })
        irf_df.to_excel(writer, sheet_name='IRF_Data', index=False)

    # Weight profiles
    for pred_name, wp in weight_profiles.items():
        wp_df = pd.DataFrame({
            'lag': range(1, len(wp['weights']) + 1),
            'weight': wp['weights']
        })
        wp_df.to_excel(writer, sheet_name=f'Weights_{pred_name}', index=False)

print(f"  ✓ Results saved: {excel_results_path}")
log_step("Excel Export", "SUCCESS", "All sheets written")

# %% [markdown]
# ## Cell 9: Visualizations

# %%
# ============================================================
# FIGURE 1: Actual vs Nowcast Overlay
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(results_df.index, results_df['actual'], 'k-', linewidth=2, label='Actual GDP Growth')
ax1.plot(results_df.index, results_df['bvar_nowcast'], color='#1a5276', linewidth=1.8,
         label='B-VAR Nowcast', marker='o', markersize=4)
ax1.fill_between(results_df.index, results_df['bvar_ci68_lo'], results_df['bvar_ci68_hi'],
                 alpha=0.2, color='#2980b9', label='B-VAR 68% CI')
ax1.plot(results_df.index, results_df['midas_nowcast'], color='#922b21', linewidth=1.8,
         label='MIDAS Nowcast', marker='s', markersize=4)
# Q1 2022 geopolitical shock
ax1.axvline(pd.Timestamp('2022-01-01'), color='gray', linestyle='--', alpha=0.7)
ax1.annotate('Russia-Ukraine\nConflict', xy=(pd.Timestamp('2022-01-01'), ax1.get_ylim()[1] * 0.9),
             fontsize=8, ha='center', color='gray')
ax1.set_title('Rolling Nowcast Comparison: B-VAR vs MIDAS (Q1 2021 – Q4 2025)', fontsize=14)
ax1.set_xlabel('Quarter')
ax1.set_ylabel('GDP Growth Rate (log-diff, SA)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'fig1_nowcast_comparison.png'), dpi=300, bbox_inches='tight')
print("  ✓ Figure 1 saved")

# ============================================================
# FIGURE 2: IRF — Brent → GDP
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))
if gdp_idx_in_irf >= 0:
    h_range = range(horizon + 1)
    ax2.plot(h_range, irf_result[:, gdp_idx_in_irf, brent_idx_in_irf],
             'b-', linewidth=2, label='IRF: Brent → GDP')
    ax2.fill_between(h_range,
                     irf_lo[:, gdp_idx_in_irf, brent_idx_in_irf],
                     irf_hi[:, gdp_idx_in_irf, brent_idx_in_irf],
                     alpha=0.25, color='#85c1e9', label='68% CI')
    ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Impulse Response: Brent Crude Oil → GDP Growth', fontsize=14)
ax2.set_xlabel('Quarters after shock')
ax2.set_ylabel('Response of GDP Growth')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig2_irf_brent_gdp.png'), dpi=300, bbox_inches='tight')
print("  ✓ Figure 2 saved")

# ============================================================
# FIGURE 3: IRF — IPI → GDP
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
if gdp_idx_in_irf >= 0:
    ax3.plot(h_range, irf_result[:, gdp_idx_in_irf, ipi_idx_in_irf],
             color='#d35400', linewidth=2, label='IRF: IPI → GDP')
    ax3.fill_between(h_range,
                     irf_lo[:, gdp_idx_in_irf, ipi_idx_in_irf],
                     irf_hi[:, gdp_idx_in_irf, ipi_idx_in_irf],
                     alpha=0.25, color='#f0b27a', label='68% CI')
    ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title('Impulse Response: Industrial Production → GDP Growth', fontsize=14)
ax3.set_xlabel('Quarters after shock')
ax3.set_ylabel('Response of GDP Growth')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'fig3_irf_ipi_gdp.png'), dpi=300, bbox_inches='tight')
print("  ✓ Figure 3 saved")

# ============================================================
# FIGURE 4: Exponential Almon Weight Profile
# ============================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))
dom_w = weight_profiles[dominant_pred]
lags = range(1, len(dom_w['weights']) + 1)
ax4.bar(lags, dom_w['weights'], color='#2e86c1', alpha=0.8, edgecolor='#1a5276')
ax4.set_title(f'Exponential Almon Weight Profile: {dominant_pred.upper()}', fontsize=14)
ax4.set_xlabel('Monthly Lag')
ax4.set_ylabel('Weight')
ax4.set_xticks(list(lags))
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'fig4_almon_weights.png'), dpi=300, bbox_inches='tight')
print("  ✓ Figure 4 saved")

plt.close('all')
log_step("Visualization", "SUCCESS", "4 figures saved at 300 DPI")

# ============================================================
# STAGE TWO SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STAGE TWO — RESULTS SUMMARY")
print("=" * 60)
print(metrics_df.to_string(index=False))
print(f"\nDiebold-Mariano (Nowcast):  DM={dm_nowcast_stat:.3f}, p={dm_nowcast_p:.4f}")
print(f"Diebold-Mariano (Backcast): DM={dm_backcast_stat:.3f}, p={dm_backcast_p:.4f}")
print(f"\nRobustness (λ₁ sensitivity):")
print(robustness_df.to_string(index=False))
print("\nExcel outputs:")
print(f"  {os.path.join(DATA_DIR, 'clean_quarterly_matrix.xlsx')}")
print(f"  {os.path.join(DATA_DIR, 'raw_quarterly_aggregates.xlsx')}")
print(f"  {os.path.join(DATA_DIR, 'data_provenance.xlsx')}")
print(f"  {os.path.join(DATA_DIR, 'model_results.xlsx')}")
print("=" * 60)
print("  PIPELINE COMPLETE ✓")
print("=" * 60)

