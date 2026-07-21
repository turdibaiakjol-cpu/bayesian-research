#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd

os.chdir('/Users/akzholturdibai/Bayesian research')

# Run Stage 1 & fixes
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

train_end = '2018-12-31'
eval_start_idx = df.index.get_loc(df.index[df.index > train_end][0])

def bvar_rmse_for_lambda(lam1):
    errs = []
    for t_idx in range(eval_start_idx, len(df)):
        train_data = df.iloc[:t_idx]
        actual_t = df['gdp_dlsa'].iloc[t_idx]
        
        # apply same adaptive rule as in our run_upgrade_steps1to5.py
        # Wait, the prompt says "reconcile the lambda sensitivity baseline RMSE with the main results table RMSE".
        # In run_upgrade_steps1to5.py we used: adaptive_lam = 0.05 if t_idx < eval_start_idx+2 else 0.1
        # Let's use the requested lam1
        if lam1 == 0.1:
            lam = 0.05 if t_idx < eval_start_idx+2 else 0.1
        elif lam1 == 0.15:
            lam = 0.10 if t_idx < eval_start_idx+2 else 0.15
        elif lam1 == 0.20:
            lam = 0.15 if t_idx < eval_start_idx+2 else 0.20
        else:
            lam = lam1
            
        try:
            Bt,Vt,_,Xt,Yt,_ = estimate_bvar_fixed(train_data[endo_vars].dropna(), endo_vars, p, lam)
            last = train_data[endo_vars].dropna().values
            xn = np.ones(1)
            for l in range(p): xn = np.concatenate([xn, last[-(l+1)] if len(last)>l else np.zeros(K)])
            xn = xn[:Bt.shape[0]]
            bvar_nc = (xn @ Bt)[endo_vars.index('gdp_dlsa')]
            errs.append((actual_t - bvar_nc)**2)
        except: pass
    return np.sqrt(np.mean(errs))

rmse_10 = bvar_rmse_for_lambda(0.10)
rmse_15 = bvar_rmse_for_lambda(0.15)
rmse_20 = bvar_rmse_for_lambda(0.20)

print(f"rob_10: {rmse_10}")
print(f"rob_15: {rmse_15}")
print(f"rob_20: {rmse_20}")
