# %% [markdown]
# ## Cell 3: Quarterly Aggregation & Stationarity Transformations

# %%
# ============================================================
# QUARTERLY AGGREGATION
# ============================================================
log_step("Quarterly Aggregation", "ATTEMPT", "Aggregating all monthly series")

def to_quarterly_mean(series):
    """Aggregate monthly to quarterly by arithmetic mean."""
    return series.resample('QS').mean()

def to_quarterly_sum(series):
    """Aggregate monthly to quarterly by sum (flow variables)."""
    return series.resample('QS').sum()

def to_quarterly_end(series):
    """Aggregate monthly to quarterly by end-of-quarter observation."""
    return series.resample('QS').last()

# Apply aggregation per variable specification
q_ipi = to_quarterly_mean(ipi_monthly)
q_stei = to_quarterly_mean(stei_monthly)
q_freight = to_quarterly_sum(freight_monthly)
q_construction = to_quarterly_sum(constr_monthly)
q_trade = to_quarterly_sum(trade_monthly)
q_nbk_rate = to_quarterly_end(nbk_monthly)
q_credit = to_quarterly_end(credit_monthly)
q_m2 = to_quarterly_end(m2_monthly)
q_wages = to_quarterly_mean(wages_monthly)
q_brent = to_quarterly_mean(brent_monthly)
q_copper = to_quarterly_mean(copper_monthly)
q_zinc = to_quarterly_mean(zinc_monthly)
q_uranium = to_quarterly_mean(uranium_monthly)
q_usdkzt = to_quarterly_mean(usdkzt_monthly)
q_rubkzt = to_quarterly_mean(rubkzt_monthly)
q_cpi_food = to_quarterly_end(cpi_food_monthly)
q_cpi_nonfood = to_quarterly_end(cpi_nonfood_monthly)
q_cpi_services = to_quarterly_end(cpi_services_monthly)

log_step("Quarterly Aggregation", "SUCCESS", "18 variables aggregated")

# ============================================================
# SEASONAL ADJUSTMENT OF GDP (STL)
# ============================================================
log_step("Seasonal Adjustment", "ATTEMPT", "STL decomposition of log GDP")

# Convert GDP PVI to level index (cumulative)
gdp_level = (gdp_series / 100).cumprod() * 100  # Cumulative index, base=100
gdp_log = np.log(gdp_level)

# STL decomposition (period=4 for quarterly)
try:
    stl_result = STL(gdp_log, period=4, robust=True).fit()
    gdp_sa = gdp_log - stl_result.seasonal  # Remove seasonal component
    gdp_dlsa = gdp_sa.diff().dropna()
    gdp_dlsa.name = 'gdp_dlsa'
    log_step("Seasonal Adjustment", "SUCCESS",
             f"STL applied, gdp_dlsa has {len(gdp_dlsa)} obs")
except Exception as e:
    log_step("Seasonal Adjustment", "FAILURE", str(e))
    # Fallback: simple log diff without SA
    gdp_dlsa = np.log(gdp_level).diff().dropna()
    gdp_dlsa.name = 'gdp_dlsa'
    log_step("Seasonal Adjustment", "FALLBACK", "Using log-diff without SA")

# ============================================================
# STATIONARITY TRANSFORMATIONS
# ============================================================
log_step("Stationarity Transforms", "ATTEMPT", "Log-differencing all variables")

def log_diff(series, name):
    """Apply log first-difference transformation."""
    result = np.log(series.clip(lower=0.01)).diff().dropna()
    result.name = name
    return result

def level_diff(series, name):
    """Apply first-difference in levels."""
    result = series.diff().dropna()
    result.name = name
    return result

# Apply transformations
ipi_dl = log_diff(q_ipi.clip(lower=-50) + 100, 'ipi_dl')  # Shift to positive for log
stei_dl = log_diff(q_stei.clip(lower=-50) + 100, 'stei_dl')
freight_dl = log_diff(q_freight, 'freight_dl')
construction_dl = log_diff(q_construction, 'construction_dl')
trade_dl = log_diff(q_trade, 'trade_dl')
nbk_rate_d = level_diff(q_nbk_rate, 'nbk_rate_d')
credit_dl = log_diff(q_credit, 'credit_dl')
m2_dl = log_diff(q_m2, 'm2_dl')
wages_dl = log_diff(q_wages, 'wages_dl')
brent_dl = log_diff(q_brent, 'brent_dl')
copper_dl = log_diff(q_copper, 'copper_dl')
zinc_dl = log_diff(q_zinc, 'zinc_dl')
uranium_dl = log_diff(q_uranium, 'uranium_dl')
usdkzt_dl = log_diff(q_usdkzt, 'usdkzt_dl')
rubkzt_dl = log_diff(q_rubkzt, 'rubkzt_dl')
cpi_food_dl = log_diff(q_cpi_food, 'cpi_food_dl')
cpi_nonfood_dl = log_diff(q_cpi_nonfood, 'cpi_nonfood_dl')
cpi_services_dl = log_diff(q_cpi_services, 'cpi_services_dl')

log_step("Stationarity Transforms", "SUCCESS", "18 variables transformed")

# ============================================================
# ASSEMBLE FINAL MATRIX
# ============================================================
log_step("Matrix Assembly", "ATTEMPT", "Aligning on quarterly index")

all_vars = [
    gdp_dlsa, ipi_dl, stei_dl, freight_dl, construction_dl, trade_dl,
    nbk_rate_d, credit_dl, m2_dl, wages_dl, brent_dl, copper_dl,
    zinc_dl, uranium_dl, usdkzt_dl, rubkzt_dl,
    cpi_food_dl, cpi_nonfood_dl, cpi_services_dl
]

# Combine into DataFrame
df = pd.concat(all_vars, axis=1)

# Restrict to Q1 2010 – Q4 2025
mask = (df.index >= '2010-01-01') & (df.index <= '2025-12-31')
df = df.loc[mask]

# Interior interpolation for ≤2 consecutive missing quarters
df = df.interpolate(method='linear', limit=2, limit_direction='both')

# Check missingness
miss_report = df.isnull().sum()
miss_pct = (miss_report / len(df) * 100).round(1)
print("\n" + "=" * 60)
print("MISSINGNESS REPORT")
print("=" * 60)
for col in df.columns:
    status = "OK" if miss_pct[col] < 15 else "DROP"
    print(f"  {col:20s}: {miss_report[col]:3d} missing ({miss_pct[col]:5.1f}%) [{status}]")

# Drop variables with >15% missing
drop_cols = miss_pct[miss_pct >= 15].index.tolist()
if drop_cols:
    print(f"\nDropping variables with >15% missing: {drop_cols}")
    df = df.drop(columns=drop_cols)

# Drop remaining NaN rows at boundaries
df = df.dropna()

# Final summary
print("\n" + "=" * 60)
print("FINAL DATA MATRIX SUMMARY")
print("=" * 60)
print(f"  Shape: {df.shape}")
print(f"  Date range: {df.index[0].strftime('%Y-Q%q') if hasattr(df.index[0], 'strftime') else df.index[0]} to {df.index[-1]}")
print(f"  N observations: {len(df)}")
print(f"  Variables ({df.shape[1]}): {list(df.columns)}")
print(f"  Any NaN remaining: {df.isnull().any().any()}")

# ADF stationarity check
print("\n" + "=" * 60)
print("ADF STATIONARITY TESTS (5% level)")
print("=" * 60)
for col in df.columns:
    try:
        adf_stat, adf_p, _, _, _, _ = adfuller(df[col].dropna(), maxlag=8)
        status = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"
        print(f"  {col:20s}: ADF={adf_stat:7.3f}, p={adf_p:.4f} [{status}]")
    except Exception as e:
        print(f"  {col:20s}: ADF test failed - {str(e)[:50]}")

log_step("Matrix Assembly", "SUCCESS", f"Final matrix: {df.shape}")

# ============================================================
# EXPORT CLEAN DATA TO EXCEL
# ============================================================
excel_path = os.path.join(DATA_DIR, 'clean_quarterly_matrix.xlsx')
df.to_excel(excel_path, index=True, sheet_name='Transformed_Data')

# Also save raw quarterly aggregates
raw_q = pd.DataFrame({
    'gdp_pvi': gdp_series, 'ipi_q_mean': q_ipi, 'stei_q_mean': q_stei,
    'freight_q_sum': q_freight, 'construction_q_sum': q_construction,
    'trade_q_sum': q_trade, 'nbk_rate_eop': q_nbk_rate,
    'credit_eop': q_credit, 'm2_eop': q_m2, 'wages_q_mean': q_wages,
    'brent_q_mean': q_brent, 'copper_q_mean': q_copper,
    'zinc_q_mean': q_zinc, 'uranium_q_mean': q_uranium,
    'usdkzt_q_mean': q_usdkzt, 'rubkzt_q_mean': q_rubkzt,
    'cpi_food_eop': q_cpi_food, 'cpi_nonfood_eop': q_cpi_nonfood,
    'cpi_services_eop': q_cpi_services,
})
raw_q.to_excel(os.path.join(DATA_DIR, 'raw_quarterly_aggregates.xlsx'),
               sheet_name='Raw_Quarterly')

# Save provenance table
prov_df = pd.DataFrame(provenance)
prov_df.to_excel(os.path.join(DATA_DIR, 'data_provenance.xlsx'),
                 index=False, sheet_name='Provenance')

print(f"\n  ✓ Clean matrix saved: {excel_path}")
print(f"  ✓ Raw quarterly saved: {os.path.join(DATA_DIR, 'raw_quarterly_aggregates.xlsx')}")
print(f"  ✓ Provenance table saved: {os.path.join(DATA_DIR, 'data_provenance.xlsx')}")

# ============================================================
# PIPELINE HEALTH REPORT — STAGE ONE
# ============================================================
print("\n" + "=" * 60)
print("STAGE ONE — PIPELINE HEALTH REPORT")
print("=" * 60)
print(f"  Variables successfully collected: {len(all_vars)}")
print(f"  Variables requiring interpolation: {int((miss_report > 0).sum())}")
print(f"  Variables dropped (>15% missing): {len(drop_cols)}")
print(f"  Final effective N: {len(df)}")
print(f"  Final variable count: {df.shape[1]}")
print("  Status: STAGE ONE COMPLETE ✓")
print("=" * 60)

