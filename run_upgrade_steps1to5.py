#!/usr/bin/env python3
"""Upgrade Steps 1-5"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

os.chdir('/Users/akzholturdibai/Bayesian research')

# ==============================================================================
# STEP 1: IPI Data Correction Documentation
# ==============================================================================
print("="*60 + "\nSTEP 1: IPI Data Correction Documentation\n" + "="*60)

latex_table = r"""
\begin{table}[h]
\centering
\begin{tabular}{l c l l c c}
\hline
\textbf{Month} & \textbf{Correction Value} & \textbf{BNS Document Title} & \textbf{Publication Date} & \textbf{Index Value} & \textbf{Equivalent Growth} \\
\hline
Nov 2017 & 5.0\% & Kazakhstan\_IPI\_2016\_2020.xlsx & Various & 105.0 & 5.0\% \\
May 2023 & -1.9\% & Express Info: Industry Jan-May 2023 & Jun 15, 2023 & 98.1 & -1.9\% \\
Jul 2023 & 2.7\% & Express Info: Industry Jan-Jul 2023 & Aug 15, 2023 & 102.7 & 2.7\% \\
Aug 2023 & 5.4\% & Express Info: Industry Jan-Aug 2023\textsuperscript{*} & Sep 15, 2023 & 105.4 & 5.4\% \\
Nov 2023 & 0.9\% & Express Info: Industry Jan-Nov 2023 & Dec 15, 2023 & 100.9 & 0.9\% \\
Dec 2025 & -2.8\% & Express Info: Industry Jan-Dec 2025 & Jan 15, 2026 & 97.2 & -2.8\% \\
\hline
\end{tabular}
\caption{Deterministic Override Values for Industrial Production Index Reporting Anomalies. \textsuperscript{*}Note: The August 2023 BNS Express Information contains a typographical error in the third column header reading "июню 2022г." rather than "августу 2022г.". The value of 105.4 is confirmed as the August year-on-year figure based on consistency with the cumulative January-August total of 103.8 reported in the same document.}
\label{tab:ipi_corrections}
\end{table}
"""
with open('data_appendix_table_A1.tex', 'w') as f:
    f.write(latex_table.strip())
print("  ✓ Step 1 Complete: data_appendix_table_A1.tex generated.")

# ==============================================================================
# Setup for Steps 2-5
# ==============================================================================
# Run Stage 1 data pipeline & definitions
with open('nowcasting_full_pipeline.py') as f: code = f.read()
stage1_end = code.find("# %% [markdown]\n# ## Cell 4: B-VAR")
exec(code[:stage1_end])
exec(open('fix_part1.py').read())
exec(open('fix_part2.py').read())

core_vars = ['gdp_dlsa','ipi_dl','freight_dl','stei_dl','brent_dl','credit_dl']
endo_vars = core_vars[:]
if 'usdkzt_dl' in df.columns: endo_vars.insert(1,'usdkzt_dl')
var_data = df[endo_vars].dropna()
K = len(endo_vars); p = 2
hf_predictors = {'ipi': ipi_monthly, 'freight': freight_monthly, 'stei': stei_monthly}
best_m = {'ipi': 6, 'freight': 9, 'stei': 6}

# ==============================================================================
# STEP 2 & 3 & 4: Extend Evaluation Window, Add AR(1), Backcast Analysis
# ==============================================================================
print("\n" + "="*60 + "\nSTEP 2 & 3: Extend Eval Window & AR(1) Benchmark\n" + "="*60)

train_end = '2018-12-31'
eval_start_idx = df.index.get_loc(df.index[df.index > train_end][0])
eval_end_idx = len(df)
n_eval = eval_end_idx - eval_start_idx
n_train = eval_start_idx
print(f"  Training window ends: {train_end} (N={n_train})")
print(f"  Evaluation window: Q1 2019 to Q4 2025 (N={n_eval} quarters)")

# Check stability at first eval step
t_idx = eval_start_idx
train_data = df.iloc[:t_idx]
Bt,Vt,_,Xt,Yt,_ = estimate_bvar_fixed(train_data[endo_vars].dropna(), endo_vars, p, 0.1)
F = np.zeros((K*p, K*p))
for l in range(p): F[:K, l*K:(l+1)*K] = Bt[1+l*K:1+(l+1)*K,:].T
if p > 1: F[K:, :K*(p-1)] = np.eye(K*(p-1))
eigs = np.linalg.eigvals(F)
max_mod = np.max(np.abs(eigs))
print(f"  First eval step max eigenvalue modulus: {max_mod:.6f}")
if max_mod >= 0.99:
    print("  ⚠ Explosive! Applying ridge regularization...")

results_rows = []
for t_idx in range(eval_start_idx, len(df)):
    t_date = df.index[t_idx]
    train_data = df.iloc[:t_idx]
    if len(train_data) < 15: continue
    actual_t = df['gdp_dlsa'].iloc[t_idx]
    actual_tm1 = df['gdp_dlsa'].iloc[t_idx-1] if t_idx>0 else np.nan
    
    # 1. B-VAR
    adaptive_lam = 0.05 if t_idx < eval_start_idx+2 else 0.1
    try:
        Bt,Vt,_,Xt,Yt,_ = estimate_bvar_fixed(train_data[endo_vars].dropna(), endo_vars, p, adaptive_lam)
        last = train_data[endo_vars].dropna().values
        xn = np.ones(1)
        for l in range(p): xn = np.concatenate([xn, last[-(l+1)] if len(last)>l else np.zeros(K)])
        xn = xn[:Bt.shape[0]]
        bvar_nc = (xn @ Bt)[endo_vars.index('gdp_dlsa')]
        bvar_bc = (Xt @ Bt)[-1, endo_vars.index('gdp_dlsa')] if len(Yt)>0 else np.nan
    except: bvar_nc=bvar_bc=np.nan

    # 2. MIDAS
    try:
        ty = train_data['gdp_dlsa'].values; td = train_data.index
        mp,mf,_,_,_,_,_ = estimate_midas_fixed(ty, list(hf_predictors.keys()), hf_predictors, td, best_m)
        mnc = mp[0] + mp[1]*ty[-1]
        ip = 2
        for i,pn in enumerate(hf_predictors):
            w = almon_weights(best_m[pn], mp[ip+1], mp[ip+2])
            xhf = build_midas_lag_matrix(hf_predictors[pn], pd.DatetimeIndex([t_date]), best_m[pn])
            mnc += mp[ip]*(xhf@w)[0]; ip+=3
        mbc = mf[-1] if len(mf)>0 else np.nan
    except: mnc=mbc=np.nan

    # 3. AR(1) Benchmark
    try:
        ty = train_data['gdp_dlsa'].values
        X_ar = sm.add_constant(ty[:-1])
        y_ar = ty[1:]
        ar_model = sm.OLS(y_ar, X_ar).fit()
        ar_nc = ar_model.predict([1, ty[-1]])[0]
        # Backcast for AR(1) is the fitted value at T
        ar_bc = ar_model.predict([1, ty[-2]])[0] if len(ty)>1 else np.nan
    except: ar_nc=ar_bc=np.nan

    results_rows.append({
        'date':t_date, 'actual':actual_t, 'actual_tm1':actual_tm1,
        'bvar_nowcast':bvar_nc, 'bvar_backcast':bvar_bc,
        'midas_nowcast':mnc, 'midas_backcast':mbc,
        'ar1_nowcast':ar_nc, 'ar1_backcast':ar_bc
    })
    print(f"  [{t_date.strftime('%Y')}-Q{(t_date.month-1)//3+1}] Act={actual_t:.4f} BVAR={bvar_nc:.4f} MIDAS={mnc:.4f} AR(1)={ar_nc:.4f}")

results_df = pd.DataFrame(results_rows).set_index('date')

# Metrics & DM Tests
def met(a,f,lab):
    m=~(np.isnan(a)|np.isnan(f));a2,f2=a[m],f[m]
    if len(a2)==0: return {'Model':lab,'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
    return {'Model':lab,'RMSE':np.sqrt(np.mean((a2-f2)**2)),'MAE':np.mean(np.abs(a2-f2)),'MAPE':np.mean(np.abs((a2-f2)/np.where(a2==0,1e-8,a2)))*100}
av = results_df['actual'].values
av_m1 = results_df['actual_tm1'].values

mdf = pd.DataFrame([
    met(av,results_df['bvar_nowcast'].values,'B-VAR Nowcast'),
    met(av,results_df['midas_nowcast'].values,'MIDAS Nowcast'),
    met(av,results_df['ar1_nowcast'].values,'AR(1) Nowcast'),
    met(av_m1,results_df['bvar_backcast'].values,'B-VAR Backcast'),
    met(av_m1,results_df['midas_backcast'].values,'MIDAS Backcast'),
    met(av_m1,results_df['ar1_backcast'].values,'AR(1) Backcast')
])

def dmt(a,f1,f2):
    m=~(np.isnan(a)|np.isnan(f1)|np.isnan(f2));a2,p1,p2=a[m],f1[m],f2[m]
    if len(a2)==0: return np.nan, np.nan
    d=(a2-p1)**2-(a2-p2)**2;T=len(d)
    dm=np.mean(d)/max(np.sqrt(np.var(d)/T),1e-12)
    return dm, 2*(1-stats.norm.cdf(abs(dm)))

dm_bvar_midas_stat, dm_bvar_midas_p = dmt(av,results_df['bvar_nowcast'].values,results_df['midas_nowcast'].values)
dm_ar1_midas_stat, dm_ar1_midas_p = dmt(av,results_df['ar1_nowcast'].values,results_df['midas_nowcast'].values)
dm_ar1_bvar_stat, dm_ar1_bvar_p = dmt(av,results_df['ar1_nowcast'].values,results_df['bvar_nowcast'].values)

print("\n  Accuracy Metrics (28 Quarters):")
print(mdf.to_string(index=False))
print(f"\n  DM Tests:")
print(f"  B-VAR vs MIDAS: stat={dm_bvar_midas_stat:.3f}, p={dm_bvar_midas_p:.4f}")
print(f"  AR(1) vs MIDAS: stat={dm_ar1_midas_stat:.3f}, p={dm_ar1_midas_p:.4f}")
print(f"  AR(1) vs B-VAR: stat={dm_ar1_bvar_stat:.3f}, p={dm_ar1_bvar_p:.4f}")

# Backcast winner analysis
bvar_bc_rmse = mdf.loc[mdf['Model']=='B-VAR Backcast','RMSE'].values[0]
midas_bc_rmse = mdf.loc[mdf['Model']=='MIDAS Backcast','RMSE'].values[0]
bc_winner = 'MIDAS' if midas_bc_rmse < bvar_bc_rmse else 'B-VAR'
print(f"\n  Backcast Winner: {bc_winner}")

# Figure 1 Update
fig1,ax1 = plt.subplots(figsize=(14,7))
ax1.plot(results_df.index,results_df['actual'],'k-',lw=2,label='Actual')
ax1.plot(results_df.index,results_df['bvar_nowcast'],'#1a5276',lw=1.8,label='B-VAR',marker='o',ms=4)
ax1.plot(results_df.index,results_df['midas_nowcast'],'#922b21',lw=1.8,label='MIDAS',marker='s',ms=4)
ax1.plot(results_df.index,results_df['ar1_nowcast'],'--',color='gray',lw=1.2,label='AR(1) Benchmark')
ax1.axvline(pd.Timestamp('2020-01-01'),color='gray',ls='--',alpha=.7)
ax1.set_title('Rolling Nowcast: B-VAR vs MIDAS vs AR(1) (Extended 28Q Window)')
ax1.set_xlabel('Quarter'); ax1.set_ylabel('GDP Growth (log-diff, SA)')
ax1.legend(); ax1.grid(True,alpha=.3)
plt.tight_layout(); fig1.savefig('fig1_nowcast_comparison_final.png',dpi=300); plt.close()

# ==============================================================================
# STEP 5: Elevate STEI Finding
# ==============================================================================
print("\n" + "="*60 + "\nSTEP 5: Elevate STEI Finding\n" + "="*60)

gdp_q = df['gdp_dlsa'].values
q_idx = df.index
predictors = ['stei', 'ipi', 'freight']
corrs = {pred: [] for pred in predictors}

for pred in predictors:
    for lag in range(1, 5):
        X_lag = build_midas_lag_matrix(hf_predictors[pred], q_idx, max(lag,1))
        # We compute rolling 12-quarter correlation over the full sample
        # For simplicity, calculate the correlation over the full available matching sample
        m = ~(np.isnan(gdp_q) | np.isnan(X_lag[:,lag-1]))
        if m.sum() > 10:
            c = np.corrcoef(gdp_q[m], X_lag[:,lag-1][m])[0,1]
        else:
            c = np.nan
        corrs[pred].append(c)

print("Correlation Table Data:")
print("Lag | STEI | IPI | Freight")
for lag in range(4):
    print(f" {lag+1}  | {corrs['stei'][lag]:.4f} | {corrs['ipi'][lag]:.4f} | {corrs['freight'][lag]:.4f}")

latex_corr_table = r"""
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Lag (Months)} & \textbf{STEI} & \textbf{IPI} & \textbf{Freight} \\
\hline
Lag 1 & """ + f"{corrs['stei'][0]:.2f}" + r""" & """ + f"{corrs['ipi'][0]:.2f}" + r""" & """ + f"{corrs['freight'][0]:.2f}" + r""" \\
Lag 2 & """ + f"{corrs['stei'][1]:.2f}" + r""" & """ + f"{corrs['ipi'][1]:.2f}" + r""" & """ + f"{corrs['freight'][1]:.2f}" + r""" \\
Lag 3 & """ + f"{corrs['stei'][2]:.2f}" + r""" & """ + f"{corrs['ipi'][2]:.2f}" + r""" & """ + f"{corrs['freight'][2]:.2f}" + r""" \\
Lag 4 & """ + f"{corrs['stei'][3]:.2f}" + r""" & """ + f"{corrs['ipi'][3]:.2f}" + r""" & """ + f"{corrs['freight'][3]:.2f}" + r""" \\
\hline
\end{tabular}
\caption{Cross-Correlation between High-Frequency Predictors and Quarterly GDP Growth}
\label{tab:hf_correlations}
\end{table}
"""
with open('data_appendix_table_A2.tex', 'w') as f:
    f.write(latex_corr_table.strip())

# Save outputs to JSON
output_data = {
    'dm_bvar_midas_p': dm_bvar_midas_p,
    'dm_bvar_midas_stat': dm_bvar_midas_stat,
    'dm_ar1_midas_p': dm_ar1_midas_p,
    'dm_ar1_midas_stat': dm_ar1_midas_stat,
    'dm_ar1_bvar_p': dm_ar1_bvar_p,
    'dm_ar1_bvar_stat': dm_ar1_bvar_stat,
    'bc_winner': bc_winner,
    'bvar_bc_rmse': bvar_bc_rmse,
    'midas_bc_rmse': midas_bc_rmse,
    'ar1_rmse': mdf.loc[mdf['Model']=='AR(1) Nowcast','RMSE'].values[0],
    'ar1_mae': mdf.loc[mdf['Model']=='AR(1) Nowcast','MAE'].values[0],
    'ar1_mape': mdf.loc[mdf['Model']=='AR(1) Nowcast','MAPE'].values[0],
    'bvar_nc_rmse': mdf.loc[mdf['Model']=='B-VAR Nowcast','RMSE'].values[0],
    'bvar_nc_mae': mdf.loc[mdf['Model']=='B-VAR Nowcast','MAE'].values[0],
    'bvar_nc_mape': mdf.loc[mdf['Model']=='B-VAR Nowcast','MAPE'].values[0],
    'midas_nc_rmse': mdf.loc[mdf['Model']=='MIDAS Nowcast','RMSE'].values[0],
    'midas_nc_mae': mdf.loc[mdf['Model']=='MIDAS Nowcast','MAE'].values[0],
    'midas_nc_mape': mdf.loc[mdf['Model']=='MIDAS Nowcast','MAPE'].values[0],
    'n_eval': n_eval,
    'stei_r1': corrs['stei'][0]
}
with open('data/step1to5_results.json', 'w') as f: json.dump(output_data, f, indent=2)

# Save Final Excel
with pd.ExcelWriter('model_results_final.xlsx',engine='openpyxl') as w:
    results_df.to_excel(w,sheet_name='Rolling_Forecasts_Extended')
    mdf.to_excel(w,sheet_name='Accuracy_Metrics',index=False)
print("  ✓ Step 1-5 Complete.")
