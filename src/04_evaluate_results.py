"""
04_evaluate_results.py — Model Evaluation & Figure Generation
==============================================================
Loads rolling forecast results from data/results/, computes accuracy
metrics (RMSE, MAE, MAPE), runs the Diebold-Mariano test with
Harvey-Leybourne-Newbold small-sample correction, and generates
all publication-ready figures.

Outputs:
  - outputs/metrics/accuracy_metrics.csv
  - outputs/metrics/dm_test.csv
  - outputs/figures/fig1_nowcast_comparison.png
  - outputs/figures/fig2_irf_brent_gdp.png
  - outputs/figures/fig3_irf_ipi_gdp.png
  - outputs/figures/fig4_almon_weights.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'data', 'results')
METRICS = os.path.join(ROOT, 'outputs', 'metrics')
FIGURES = os.path.join(ROOT, 'outputs', 'figures')

os.makedirs(METRICS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ─── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
NAVY   = '#1a3557'
STEEL  = '#2c5f8a'
GOLD   = '#d4a017'
RED    = '#c0392b'
GREEN  = '#27ae60'
GREY   = '#7f8c8d'


def compute_metrics(actual, predicted, label=''):
    """Compute RMSE, MAE, MAPE for a forecast series."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    n = len(a)
    if n == 0:
        return {'model': label, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'n': 0}
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae = np.mean(np.abs(a - p))
    mape = np.mean(np.abs((a - p) / np.where(np.abs(a) < 1e-10, 1e-10, a))) * 100
    return {'model': label, 'rmse': rmse, 'mae': mae, 'mape': mape, 'n': n}


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold correction.

    Parameters
    ----------
    e1 : np.ndarray — forecast errors from model 1 (B-VAR)
    e2 : np.ndarray — forecast errors from model 2 (MIDAS)
    h : int — forecast horizon (1 for nowcast)

    Returns
    -------
    dict with: dm_stat, dm_pvalue, hln_stat, hln_pvalue, T
    """
    from scipy.stats import t as t_dist

    d = e1 ** 2 - e2 ** 2
    T = len(d)
    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=0)

    # Autocovariance correction for h > 1
    gamma_sum = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * gamma_k

    var_d_bar = gamma_sum / T
    if var_d_bar <= 0:
        var_d_bar = 1e-15

    # Standard DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)

    # HLN small-sample correction
    hln_correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    hln_stat = hln_correction * dm_stat

    # p-values (two-sided, using t_{T-1} for HLN)
    from scipy.stats import norm
    dm_pvalue = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    hln_pvalue = 2 * (1 - t_dist.cdf(np.abs(hln_stat), df=T - 1))

    return {
        'dm_stat': dm_stat, 'dm_pvalue': dm_pvalue,
        'hln_stat': hln_stat, 'hln_pvalue': hln_pvalue,
        'T': T, 'd_bar': d_bar
    }


def fig1_nowcast_comparison(merged, save_path):
    """Generate Figure 1: Rolling Nowcast Comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))

    dates = pd.to_datetime(merged['date'])
    ax.plot(dates, merged['actual'], color=NAVY, linewidth=2.2,
            label='Actual GDP Growth', zorder=5)
    ax.plot(dates, merged['midas_nowcast'], color=GOLD, linewidth=1.6,
            label='MIDAS Nowcast', linestyle='-', marker='o', markersize=3)
    ax.plot(dates, merged['bvar_nowcast'], color=RED, linewidth=1.4,
            label='B-VAR Nowcast', linestyle='--', marker='s', markersize=3)
    if 'ar1_nowcast' in merged.columns:
        ax.plot(dates, merged['ar1_nowcast'], color=GREY, linewidth=1.0,
                label='AR(1) Benchmark', linestyle=':', alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Log-Differenced SA GDP Growth')
    ax.set_title('Figure 1. Recursive Pseudo Out-of-Sample Nowcast Comparison',
                 fontsize=12, fontweight='bold', color=NAVY)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(save_path)}')


def fig_irf(irf_df, shock_var, title, fig_num, save_path):
    """Generate an IRF figure."""
    subset = irf_df[(irf_df['shock'] == shock_var) & (irf_df['response'] == 'gdp_dlsa')]
    if subset.empty:
        print(f'  ⚠ No IRF data for {shock_var}')
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(subset['horizon'], subset['value'], color=NAVY, linewidth=2.2,
            marker='o', markersize=4)
    ax.fill_between(subset['horizon'], 0, subset['value'], alpha=0.15, color=STEEL)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Quarters After Shock')
    ax.set_ylabel('Response (log points)')
    ax.set_title(f'Figure {fig_num}. {title}',
                 fontsize=11, fontweight='bold', color=NAVY)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(save_path)}')


def fig4_almon_weights(save_path):
    """Generate Figure 4: Almon Weight Profiles."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = [NAVY, STEEL, GOLD]
    labels = ['IPI', 'Freight', 'STEI']

    for i, (label, color) in enumerate(zip(labels, colors)):
        fpath = os.path.join(RESULTS, f'weights_{label.lower()}.csv')
        if os.path.exists(fpath):
            w_df = pd.read_csv(fpath)
            axes[i].bar(w_df['lag'], w_df['weight'], color=color, alpha=0.85, width=0.7)
            axes[i].set_title(label, fontsize=11, fontweight='bold', color=NAVY)
            axes[i].set_xlabel('Monthly Lag')
            axes[i].set_ylabel('Weight')
            axes[i].set_ylim(0, 1.0)
            axes[i].grid(True, alpha=0.15)
        else:
            axes[i].text(0.5, 0.5, f'{label}\n(no data)', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=10, color=GREY)

    fig.suptitle('Figure 4. Exponential Almon Weight Profiles by Predictor',
                 fontsize=12, fontweight='bold', color=NAVY, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  ✓ {os.path.basename(save_path)}')


if __name__ == '__main__':
    print('═' * 60)
    print('  MODEL EVALUATION & FIGURE GENERATION')
    print('═' * 60)

    # --- Load rolling forecast data ---
    print('\n[1/4] Loading forecast data...')

    # Try extended results first (from the pipeline), then modular results
    ext_path = os.path.join(RESULTS, 'rolling_forecasts_extended.csv')
    bvar_path = os.path.join(RESULTS, 'bvar_rolling_forecasts.csv')
    midas_path = os.path.join(RESULTS, 'midas_rolling_forecasts.csv')
    ar1_path = os.path.join(RESULTS, 'ar1_rolling_forecasts.csv')

    if os.path.exists(ext_path):
        merged = pd.read_csv(ext_path)
        print(f'  ✓ Loaded extended rolling forecasts: {len(merged)} quarters')
    elif os.path.exists(bvar_path) and os.path.exists(midas_path):
        bvar_df = pd.read_csv(bvar_path)
        midas_df = pd.read_csv(midas_path)
        merged = bvar_df.merge(midas_df[['date', 'midas_nowcast', 'midas_backcast']],
                               on='date', how='outer')
        if os.path.exists(ar1_path):
            ar1_df = pd.read_csv(ar1_path)
            merged = merged.merge(ar1_df[['date', 'ar1_nowcast']], on='date', how='left')
        print(f'  ✓ Merged modular forecasts: {len(merged)} quarters')
    else:
        print('  ✗ No forecast data found. Run 02_bvar_model.py and 03_midas_model.py first.')
        exit(1)

    # --- Compute accuracy metrics ---
    print('\n[2/4] Computing accuracy metrics...')
    metrics = []

    for col, label in [('midas_nowcast', 'MIDAS Nowcast'),
                       ('bvar_nowcast', 'B-VAR Nowcast'),
                       ('ar1_nowcast', 'AR(1) Nowcast'),
                       ('midas_backcast', 'MIDAS Backcast'),
                       ('bvar_backcast', 'B-VAR Backcast'),
                       ('ar1_backcast', 'AR(1) Backcast')]:
        if col in merged.columns:
            actual_col = 'actual_tm1' if 'backcast' in col else 'actual'
            if actual_col in merged.columns:
                m = compute_metrics(merged[actual_col].values,
                                    merged[col].values, label)
                metrics.append(m)
                print(f'  {label:20s}: RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}, '
                      f'MAPE={m["mape"]:.1f}%')

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(METRICS, 'accuracy_metrics.csv'), index=False)
    print(f'  ✓ Saved to outputs/metrics/accuracy_metrics.csv')

    # --- Diebold-Mariano test ---
    print('\n[3/4] Diebold-Mariano test...')
    if 'bvar_nowcast' in merged.columns and 'midas_nowcast' in merged.columns:
        valid = merged.dropna(subset=['actual', 'bvar_nowcast', 'midas_nowcast'])
        e_bvar = (valid['actual'] - valid['bvar_nowcast']).values
        e_midas = (valid['actual'] - valid['midas_nowcast']).values

        dm = diebold_mariano_test(e_bvar, e_midas, h=1)
        print(f'  Standard DM stat:  {dm["dm_stat"]:.4f} (p={dm["dm_pvalue"]:.4f})')
        print(f'  HLN-corrected:     {dm["hln_stat"]:.4f} (p={dm["hln_pvalue"]:.4f})')
        print(f'  Evaluation periods: {dm["T"]}')

        dm_df = pd.DataFrame([{
            'comparison': 'B-VAR vs MIDAS',
            'dm_statistic': dm['dm_stat'],
            'dm_pvalue': dm['dm_pvalue'],
            'hln_statistic': dm['hln_stat'],
            'hln_pvalue': dm['hln_pvalue'],
            'n_periods': dm['T'],
            'mean_loss_diff': dm['d_bar'],
        }])
        dm_df.to_csv(os.path.join(METRICS, 'dm_test.csv'), index=False)
        print(f'  ✓ Saved to outputs/metrics/dm_test.csv')

    # --- Generate figures ---
    print('\n[4/4] Generating figures...')
    fig1_nowcast_comparison(merged, os.path.join(FIGURES, 'fig1_nowcast_comparison.png'))

    irf_path = os.path.join(RESULTS, 'irf_data.csv')
    if os.path.exists(irf_path):
        irf_df = pd.read_csv(irf_path)
        fig_irf(irf_df, 'brent_dl',
                'Response of GDP Growth to Brent Crude Oil Shock', 2,
                os.path.join(FIGURES, 'fig2_irf_brent_gdp.png'))
        fig_irf(irf_df, 'ipi_dl',
                'Response of GDP Growth to IPI Shock', 3,
                os.path.join(FIGURES, 'fig3_irf_ipi_gdp.png'))
    else:
        print('  ⚠ No IRF data found — run 02_bvar_model.py first')

    fig4_almon_weights(os.path.join(FIGURES, 'fig4_almon_weights.png'))

    print('\n═ EVALUATION COMPLETE ═')
    print(f'  Figures → {FIGURES}/')
    print(f'  Metrics → {METRICS}/')
