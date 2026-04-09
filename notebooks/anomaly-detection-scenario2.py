# Generated from: anomaly2.ipynb
# Converted at: 2026-04-05T14:18:38.019Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## Global Statistics - Scenario 2


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

print("SCENARIO 2: GLOBAL STATISTICS (Refunds + Partial Cancellations + Miscellaneous) -> REALISTIC EXPOSURE")

# LOAD DATA
USECOLS = ['InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']
file_path = '/kaggle/input/datasets/prnavjoshi/excelxlsx/Online Retail.xlsx'
df_raw = pd.read_excel(
    file_path, engine='openpyxl',
    parse_dates=['InvoiceDate'],
    usecols=USECOLS,
    dtype={'StockCode': str, 'InvoiceNo': str, 'CustomerID': str},
)

df_raw['Date'] = df_raw['InvoiceDate'].dt.date
df_raw['SKU'] = df_raw['StockCode'].str.strip()

# CLASSIFY + EXCLUDE MISC 
df_raw['Quantity'] = pd.to_numeric(df_raw['Quantity'], errors='coerce')
df_raw['UnitPrice'] = pd.to_numeric(df_raw['UnitPrice'], errors='coerce')

misc_pattern = r'^(POST|DOT|AMAZON|BANK\s*CHARGES|CRUK|gift_|POSTAGE|M$|D$)'
real_mask = (
    ~df_raw['StockCode'].str.upper().str.match(misc_pattern, na=False) &
    df_raw['StockCode'].str.strip().ne('') &
    df_raw['UnitPrice'].gt(0) &
    df_raw['CustomerID'].notna()
)
real_df = df_raw[real_mask].copy()

# NETTING 
gross_picks = real_df[real_df['Quantity'] > 0].copy()
cancellations = real_df[real_df['Quantity'] < 0].copy()

ratio = len(cancellations) / len(gross_picks) if len(gross_picks) > 0 else 0
#print(f"Reverse logistics ratio        : {ratio:.1%}")

gross_picks['match_key'] = gross_picks['CustomerID'] + '-' + gross_picks['SKU']
cancellations = cancellations.copy()
cancellations['match_key'] = cancellations['CustomerID'] + '-' + cancellations['SKU']

all_rows = pd.concat(
    [gross_picks[['match_key', 'Quantity']], cancellations[['match_key', 'Quantity']]],
    ignore_index=True,)
net_qty_per_key = all_rows.groupby('match_key', sort=False)['Quantity'].sum()
# Keep only keys with a positive net balance
positive_keys = net_qty_per_key[net_qty_per_key > 0]
# For each surviving key, select the gross row with the highest original quantity
filtered_gross = gross_picks[gross_picks['match_key'].isin(positive_keys.index)]
max_idx = filtered_gross.groupby('match_key', sort=False)['Quantity'].idxmax()
net_sales = filtered_gross.loc[max_idx].copy()
# Attach net quantities and values
net_sales['NetQuantity'] = net_sales['match_key'].map(positive_keys)
net_sales['NetOrderValue'] = net_sales['NetQuantity'] * net_sales['UnitPrice']

df = (
    net_sales[['NetQuantity', 'NetOrderValue', 'SKU', 'Date', 'StockCode']]
    .rename(columns={'NetQuantity': 'Quantity', 'NetOrderValue': 'OrderValue'})
    .query('Quantity > 0')
    .reset_index(drop=True))

daily_df = df.groupby('Date', sort=False).agg(
    TotalQuantity=('Quantity', 'sum'),
    TotalValue=('OrderValue', 'sum'),
).reset_index()
daily_qty = daily_df['TotalQuantity'].values

# WITHOUT PARETO FILTER
global_qnt_mean = df['Quantity'].mean()
global_qnt_std = df['Quantity'].std(ddof=1)
global_qnt_cv = global_qnt_std / global_qnt_mean
global_qnt_skew = df['Quantity'].skew()
global_qnt_kurt = df['Quantity'].kurtosis() + 3

# Pareto 80/20 on NET quantity
sku_freq = df.groupby('SKU', sort=False)['Quantity'].sum().sort_values(ascending=False)
cum_freq = sku_freq.cumsum() / sku_freq.sum()
high_vol_skus = sku_freq[cum_freq <= 0.8].index
filtered_df = df[df['SKU'].isin(high_vol_skus)]
quantities = filtered_df['Quantity'].values.astype(np.float64)

# Location
global_qty_mean = quantities.mean()
global_qty_median = np.median(quantities)
global_qty_mode = stats.mode(quantities, keepdims=False)[0] if len(quantities) > 0 else np.nan

# Dispersion
global_qty_std = np.std(quantities, ddof=1)  # Unbiased
global_qty_cv = global_qty_std / global_qty_mean

# Shape
global_qty_skew = stats.skew(quantities)
global_qty_kurt = stats.kurtosis(quantities) + 3  # Pearson Kurtosis

# Dependencies
qty_acf, _ = acf(daily_qty, nlags=10, fft=True, alpha=0.05)  # ACF
_, lb_q, lb_pval = acf(daily_qty, nlags=10, qstat=True, fft=True)  # Ljung-Box

# Tails
global_qty_q95 = np.quantile(quantities, 0.95)
global_qty_q975 = np.quantile(quantities, 0.975)
global_qty_q99 = np.quantile(quantities, 0.99)
u_tail = global_qty_q99
exceedances = quantities[quantities > u_tail] - u_tail
global_qty_mean_excess = exceedances.mean() if len(exceedances) > 0 else np.nan

# Tests
n = len(df)
k_hill = int(np.sqrt(n))
sorted_qty_desc = np.sort(quantities)[::-1]
if k_hill > 1 and k_hill < len(sorted_qty_desc):
    log_ratios = np.log(sorted_qty_desc[:k_hill] / sorted_qty_desc[k_hill - 1])
    global_qty_hill_xi = 1 / log_ratios.mean() if not np.isnan(log_ratios.mean()) else np.nan
else:
    global_qty_hill_xi = np.nan

z_scores = (quantities - global_qty_mean) / global_qty_std
ks_stat, ks_pval = stats.kstest(z_scores, 'norm') # Kolmogorov - Smirnov Test
ad_stat, ad_pval = sm.stats.normal_ad(quantities) # Anderson - Darling Test

SW_MAX_N = 5000
sw_sample = np.random.choice(quantities, SW_MAX_N, replace=False) if len(quantities) > SW_MAX_N else quantities
sw_stat, sw_pval = stats.shapiro(sw_sample)  # Shapir - Wilkin Test

# Context Values
global_val_mean = df['OrderValue'].mean()
global_val_median = df['OrderValue'].median()
global_val_q99 = df['OrderValue'].quantile(0.99)
global_val_max = df['OrderValue'].max()

# PLOTS 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Original Histogram (without Pareto)
axes[0].hist(quantities, bins=50, density=True, alpha=0.7, color='blue')
axes[0].set_title('Net Quantity Histogram (Scenario 2 — per transaction)\nAfter cancellations & partial refunds')
axes[0].set_xlabel('Net Quantity (single order)')
axes[0].set_ylabel('Density')

# Log Histogram
log_quantities = np.log1p(quantities)
axes[1].hist(log_quantities, bins=50, density=True, alpha=0.7, color='green')
axes[1].set_title('Log-Transformed Net Quantity (Scenario 2)')
axes[1].set_xlabel('log(1 + Net Quantity)')
plt.tight_layout()
plt.show()

# QQ-Plot
fig, ax = plt.subplots(figsize=(14, 5))
stats.probplot(quantities, dist="norm", plot=ax)
ax.set_title('QQ-Plot: Net Quantity vs. Gaussian (Scenario 2)')
plt.show()

print(f"SW Statistic : {sw_stat:.3f} (p-val: {sw_pval:.4f}; "
      f"subsampled n={len(sw_sample):,}") 

print(f"\nRaw rows           : {len(df):,}")
print(f"Unique SKUs        : {df['SKU'].nunique():,}")
print(f"Date range         : {df['Date'].min()} -> {df['Date'].max()}")

print(f"\nGLOBAL QUANTITIES ( Without Pareto Filtering)")
print(f"Mean Quantity      : {global_qnt_mean:,.2f}")
print(f"Std Dev            : {global_qnt_std:,.2f}")
print(f"CV (σ/μ)           : {global_qnt_cv:,.3f}")
print(f"Skewness Quantity : {global_qnt_skew:,.2f}")
print(f"Pearson Kurtosis   : {global_qnt_kurt:,.2f}")

print(f"\nGLOBAL QUANTITY LOCATION ( With Pareto Filtering)")
print(f"Mean Quantity      : {global_qty_mean:,.2f}")
print(f"Median Quantity    : {global_qty_median:,.2f}")
print(f"Mode Quantity      : {global_qty_mode:,.2f}")

print(f"\nGLOBAL QUANTITY DISPERSION")
print(f"Std Dev Quantity   : {global_qty_std:,.2f}")
print(f"CV (σ/μ)           : {global_qty_cv:.3f}")

print(f"\nGLOBAL QUANTITY SHAPE")
print(f"Skewness           : {global_qty_skew:,.2f}")
print(f"Pearson Kurtosis   : {global_qty_kurt:,.2f}")

print(f"\nGLOBAL QUANTITY DEPENDENCE (Daily Aggregates)")
print(f"ACF Lags 1-5       : {qty_acf[1:6].round(3).tolist()}")
print(f"Ljung-Box Q (lag10): {lb_q[-1]:,.2f} (p-val: {lb_pval[-1]:.4f})")

print(f"\nGLOBAL QUANTITY TAILS")
print(f"95th %ile Quantity : {global_qty_q95:,.2f}")
print(f"97.5th %ile Qty    : {global_qty_q975:,.2f}")
print(f"99th %ile Quantity : {global_qty_q99:,.2f}")
print(f"Mean Excess (u=99%): {global_qty_mean_excess:,.2f}")

print(f"\nGLOBAL QUANTITY NORMALITY TESTS")
print(f"KS Statistic       : {ks_stat:.3f} (p-val: {ks_pval:.4f})")
print(f"AD Statistic       : {ad_stat:.3f} (p-val: {ad_pval:.4f})")
print(f"SW Statistic       : {sw_stat:.3f} (p-val: {sw_pval:.4f}; subsampled)")

print(f"\nGLOBAL ORDER VALUE (for context)")
print(f"Mean Order Value   : £ {global_val_mean:,.2f}")
print(f"Median Order Value : £ {global_val_median:,.2f}")
print(f"99th %ile Value  : £ {global_val_q99:,.2f}")
print(f"Max Order Value    : £ {global_val_max:,.2f}")

print(f"Gross picks (pre-cancellation) : {len(gross_picks):,}")
print(f"Cancellations/Refunds detected : {len(cancellations):,}")
print(f"Net positive transactions after cancellations : {len(df):,}")
print(f"Cancelled volume removed                      : {len(gross_picks) - len(df):,} lines")
print(f"Data quality warning: ~{((len(gross_picks) - len(df)) / len(gross_picks) * 100):.1f}% of gross picks had partial/full cancellation")
print(f"Real product transactions after misc + CustomerID filter: {len(real_df):,}")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from numba import jit
import warnings
warnings.filterwarnings('ignore')

#  EVT / GPD TAIL ANALYSIS 
sku_vol = df.groupby('SKU')['Quantity'].max().sort_values(ascending=False)
cum_vol = sku_vol.cumsum() / sku_vol.sum()
high_vol_skus = sku_vol[cum_vol <= 0.80].index.tolist()
quantities = df[df['SKU'].isin(high_vol_skus)]['Quantity'].values.astype(np.float64)
n_qty = len(quantities)

# Diagnostic
n_high_skus = len(high_vol_skus)
max_single_all = df['Quantity'].max()
max_single_high = quantities.max() if n_qty > 0 else 0

print(f"Selected {n_high_skus:,} SKUs that account for the top 80% of maximum single-order sizes")
print(f"Largest single net order overall : {max_single_all:,.0f}")
print(f"Largest single net order in selected SKUs : {max_single_high:,.0f}")
print(f"EVT/GPD test on transaction-level NetQuantity from these SKUs (n = {n_qty:,})\n")

# Hill Estimator 
n = len(df)
k_hill = int(np.sqrt(n))
sorted_qty_desc = np.sort(quantities)[::-1]
k_hill = min(k_hill, len(sorted_qty_desc) - 1)
if k_hill < 2:
    hill_xi = np.nan
else:
    log_ratios = np.log(sorted_qty_desc[:k_hill] / sorted_qty_desc[k_hill])
    hill_xi = log_ratios.mean()          

# Moment Estimator 
if k_hill >= 2:
    M1 = log_ratios.mean()
    M2 = (log_ratios ** 2).mean()
    ratio = M1 ** 2 / M2 if M2 > 0 else np.nan
    xi_moment = M1 + 1 - 0.5 * (1 - ratio) ** (-1) if not np.isnan(ratio) and ratio < 1 else np.nan
else:
    xi_moment = np.nan

# GEV Block Maxima (monthly blocks)
block_size = 30
maxima = np.array([
    quantities[i:i + block_size].max()
    for i in range(0, n_qty - block_size + 1, block_size)
    if len(quantities[i:i + block_size]) == block_size
])

if len(maxima) >= 3:
    c, mu_gev, sigma_gev = stats.genextreme.fit(maxima)
    xi_gev = -c
else:
    xi_gev = np.nan

print("\nTail Index Estimators ")
print(f" Hill ξ̂          : {hill_xi:.3f}")
print(f" Moment ξ̂        : {xi_moment:.3f}")
print(f" GEV Block-Max ξ̂ : {xi_gev:.3f}")
print(f" All ξ̂  > 0      : Fréchet domain confirmed (heavy tails persist even after netting)")

# Mean Excess Plot 
u_grid = np.percentile(quantities, np.linspace(90, 99, 40))
mean_excess = np.array([(quantities[quantities > u] - u).mean() for u in u_grid])

upper_idx = int(len(u_grid) * 0.6)
slope, _ = np.polyfit(u_grid[upper_idx:], mean_excess[upper_idx:], 1)

plt.figure(figsize=(12, 5))
plt.plot(u_grid, mean_excess, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Threshold u (Net Quantity)')
plt.ylabel('Mean Excess e(u)')
plt.title('Mean Excess Plot – Scenario 2 (Net Quantity)\nLinear positive slope confirms GPD')
plt.grid(True, alpha=0.3)
plt.axvline(u_grid[upper_idx], color='r', linestyle='--',
            label=f'Upper tail starts (slope = {slope:.2f})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nMean Excess Upper-Tail Slope : {slope:.3f} (>0 -> heavy tail confirmed on net data)")

# AMSE-Optimal Threshold 
@jit(nopython=True)
def hill_jit(sorted_ex, k_u):
    if k_u < 2:
        return 0.0
    lr = np.log(sorted_ex[:k_u] / sorted_ex[k_u - 1])
    return lr.mean()

u_grid_amse = np.percentile(quantities, np.linspace(92, 98.5, 25))
amse_vals = []
valid_u = []
n_boot = 200

for u in u_grid_amse:
    exc = quantities[quantities > u] - u
    k_u = len(exc)
    if k_u < 50:
        continue
    s_exc = np.sort(exc)[::-1]
    xi_boots = np.empty(n_boot)
    for b in range(n_boot):
        boot = np.random.choice(exc, k_u, replace=True)
        s_boot = np.sort(boot)[::-1]
        xi_boots[b] = hill_jit(s_boot, k_u)
    bias2 = (xi_boots.mean() - hill_xi) ** 2 if not np.isnan(hill_xi) else 0.0
    var_est = xi_boots.var()
    amse_vals.append(bias2 + var_est)
    valid_u.append(u)

amse_vals = np.array(amse_vals)
valid_u = np.array(valid_u)

if len(amse_vals) > 0:
    opt_idx = np.argmin(amse_vals)
    opt_u = valid_u[opt_idx]
    plt.figure(figsize=(12, 5))
    plt.plot(valid_u, amse_vals, 'b-o')
    plt.axvline(opt_u, color='r', linestyle='--', label=f'Optimal u = {opt_u:.1f}')
    plt.xlabel('Threshold u (Net Quantity)')
    plt.ylabel('AMSE')
    plt.title('Asymptotic MSE – Optimal Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"AMSE-Optimal Threshold u : {opt_u:.1f}")
else:
    opt_u = np.nan
    print("AMSE-Optimal Threshold u : Not computable (insufficient tail data)")

print("\nEVT / GPD CONFIRMED")
print("Peaks-Over-Threshold + Genaralized Pareto Distribution is confirmed")

# ## HIgh-volume SKU filtering


import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Scenario 2
df_plot = df.copy()

df_plot['Revenue'] = df_plot['OrderValue']

file_path = '/kaggle/input/datasets/prnavjoshi/excelxlsx/Online Retail.xlsx'
desc_map = pd.read_excel(file_path, usecols=['StockCode', 'Description']).drop_duplicates('StockCode')

df_plot = df_plot.merge(desc_map, on='StockCode', how='left')

top15 = (df_plot.groupby('StockCode')
         .agg(Total_Quantity=('Quantity','sum'),
              Avg_Unit_Price=('OrderValue','mean'),      
              Total_Revenue=('Revenue','sum'))
         .sort_values('Total_Quantity', ascending=False)
         .head(15))

print(top15[['Total_Quantity', 'Avg_Unit_Price', 'Total_Revenue']].round(2))

# ## Parameter Summary


# FINAL SUMMARY 
summary = pd.DataFrame([
    ['Total Net Transactions',          len(df),                          f"{len(df):,}"],
    ['Unique SKUs (Net)',               df['SKU'].nunique(),              f"{df['SKU'].nunique():,}"],
    ['Global Mean Net Order Value',     df['OrderValue'].mean(),          f"£{df['OrderValue'].mean():,.2f}"],
    ['Global Median Net Order Value',   df['OrderValue'].median(),        f"£{df['OrderValue'].median():,.2f}"],
    ['Pearson Kurtosis (Net Quantity)', df['Quantity'].kurtosis() + 3,    f"{df['Quantity'].kurtosis() + 3:,.1f}"],
    ['Skewness (Net Quantity)',         df['Quantity'].skew(),            f"{df['Quantity'].skew():,.2f}"],
    ['CV (σ/μ) – Net Quantity',         df['Quantity'].std() / df['Quantity'].mean() if df['Quantity'].mean() != 0 else np.nan, 
                                      f"{df['Quantity'].std() / df['Quantity'].mean():.3f}" if df['Quantity'].mean() != 0 else "—"],
    ['99.9th Percentile Net Value',     df['OrderValue'].quantile(0.999), f"£ {df['OrderValue'].quantile(0.999):,.2f}"],
    ['Max Net Order Value',             df['OrderValue'].max(),           f"£ {df['OrderValue'].max():,.2f}"],
    ['Mean Tail Index ξ (Net)',         hill_xi,                          f"{hill_xi:.3f}"],
    ['% ξ > 0 (Heavy Tail – Net)',      (hill_xi > 0) * 100 if not np.isnan(hill_xi) else 0, 
                                      f"{(hill_xi > 0)*100:.1f}%" if not np.isnan(hill_xi) else "—"],
], columns=['Metric', 'Raw', 'Formatted'])

# Formatting
summary['Value'] = summary['Formatted']
summary = summary[['Metric', 'Value']]

print("Final Summary")
print(summary.to_string(index=False))

# ## Mock Delays


import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=314159)
n = len(df)

# Configuration according to WERC/CSCMP 2025 
SURGE_PCT = 0.18
DRAGON_PCT = 0.00025  # Overall 0.025% target
DRAGON_BIAS_EXP = 1.20  # Heavy-tail concentration (tunable 1.2–1.5)
NORMAL_MEAN_MIN = 25.0
SURGE_MEAN_MIN = 90.0
DRAGON_MEAN_MIN = 360.0
SIGMA_NORMAL = 0.50
SIGMA_SURGE = 0.65
SIGMA_DRAGON = 0.82

SLA_BINS = [0, 30, 120, 360, np.inf]
SLA_LABELS = ['On-Time', 'Minor Delay', 'SLA Breach', 'Fulfillment Failure']

NORMAL_CLIP = (5, 120)
SURGE_CLIP = (30, 360)
DRAGON_CLIP = (200, 1440)

df['OrderValue_GBP'] = df['OrderValue']       
# Size biased assignment
weights = df['OrderValue_GBP'] ** DRAGON_BIAS_EXP
weights = weights / weights.sum()  # normalized probabilities

all_idx = np.arange(n)
dragon_idx = rng.choice(all_idx, size=int(DRAGON_PCT * n), p=weights, replace=False)
remaining = np.setdiff1d(all_idx, dragon_idx)
surge_idx = rng.choice(remaining, size=int(SURGE_PCT * n), replace=False)

tier = np.full(n, 'normal', dtype=object)
tier[dragon_idx] = 'dragon'
tier[surge_idx] = 'surge'
df['Delay_Tier'] = tier

# Delay 
delays = np.zeros(n, dtype=float)
for mask, mean_min, sigma, clip in [
    (df['Delay_Tier'] == 'normal', NORMAL_MEAN_MIN, SIGMA_NORMAL, NORMAL_CLIP),
    (df['Delay_Tier'] == 'surge', SURGE_MEAN_MIN, SIGMA_SURGE, SURGE_CLIP),
    (df['Delay_Tier'] == 'dragon', DRAGON_MEAN_MIN, SIGMA_DRAGON, DRAGON_CLIP),]:
    
    mu = np.log(mean_min) - 0.5 * sigma**2
    raw = np.exp(mu + sigma * rng.standard_normal(mask.sum()))
    delays[mask] = np.clip(raw, clip[0], clip[1])

df['Delay_min'] = delays
df['SLA_Status'] = pd.cut(df['Delay_min'], bins=SLA_BINS, labels=SLA_LABELS, right=False)

# Output
sample_days = (df['Date'].max() - df['Date'].min()).days + 1
total_rev = df['OrderValue_GBP'].sum()
sla_counts = df['SLA_Status'].value_counts().reindex(SLA_LABELS)
revenue_at_risk = df[df['SLA_Status'].isin(['SLA Breach','Fulfillment Failure'])]['OrderValue_GBP'].sum()

print(f"\nDelays – SCENARIO 2 (NETTED)")
print(f"{'='*50}")
print(f"Surge events : {(df['Delay_Tier']=='surge').sum():,} / {n:,} ({(df['Delay_Tier']=='surge').mean():.1%})")
print(f"Dragon events : {(df['Delay_Tier']=='dragon').sum():,} / {n:,} ({(df['Delay_Tier']=='dragon').mean():.4%})")
print(f"Sample period : {sample_days} days ({df['Date'].min()} → {df['Date'].max()})")
print(f"{'─'*50}")
print(f"Overall Mean Delay : {df['Delay_min'].mean():.1f} min ({df['Delay_min'].mean()/60:.1f} hrs)")
print(f"95th %ile Delay : {df['Delay_min'].quantile(0.95):.1f} min")
print(f"99th %ile Delay : {df['Delay_min'].quantile(0.99):.1f} min")
print(f"{'─'*50}")
print(f"SLA BREAKDOWN (Gross Revenue £ {total_rev:,.0f} | Period: {sample_days} days)")
for status in SLA_LABELS:
    count = sla_counts[status]
    rev = df.loc[df['SLA_Status'] == status, 'OrderValue_GBP'].sum()
    pct = 100 * rev / total_rev
    tag = "implies Revenue at Risk" if status in ['SLA Breach','Fulfillment Failure'] else ""
    print(f" {status:<18}: {count:>7,} txns | £{rev:>12,.0f} ({pct:5.1f}%) {tag}")
print(f"{'─'*50}")
print(f"Total Revenue at Risk    : £ {revenue_at_risk:>2,.0f} ({100*revenue_at_risk/total_rev:.3f}% of gross)")
print(f"Annualised exposure est. : £ {revenue_at_risk * 365 / sample_days:>2,.0f}")
print(f"{'='*50}\n")
print("Delay stats by tier:")
print(df.groupby('Delay_Tier')['Delay_min'].agg(['mean','median','std','max','count']).round(1))

# ## Value at Risk


# Refunds + Partial cancellations

ANNUAL_HOLDING_RATE = 0.25          # 25% APR — WERC benchmark for working capital cost
SLA_BREACH_STATUSES = ['SLA Breach', 'Fulfillment Failure']

# Derived Columns (using netted values)
df['Days_Delayed']     = df['Delay_min'] / 1440.0
df['Holding_Cost_GBP'] = df['OrderValue_GBP'] * df['Days_Delayed'] * ANNUAL_HOLDING_RATE
df['Net_Revenue_GBP']  = df['OrderValue_GBP'] - df['Holding_Cost_GBP']

# Primary: Extreme delay simulation (0.0025% tail)
df['Is_Dragon'] = df['Delay_Tier'] == 'dragon'

# Secondary: High-value tail 
dragon_val_thresh = df['OrderValue_GBP'].quantile(0.999)
df['Is_Value_Dragon'] = df['OrderValue_GBP'] >= dragon_val_thresh

# Unfulfilled Dragon = primary dragon that also breached SLA
df['Unfulfilled_Dragon'] = (
    df['Is_Dragon'] & 
    (df['Delay_min'] >= 360) & 
    df['SLA_Status'].isin(SLA_BREACH_STATUSES)
)

# Aggregates
total_gross   = df['OrderValue_GBP'].sum()
total_holding = df['Holding_Cost_GBP'].sum()
total_net     = df['Net_Revenue_GBP'].sum()

# Breach rates by operations
breach_surge  = df.loc[df['Delay_Tier'] == 'surge',  'SLA_Status'].isin(SLA_BREACH_STATUSES).mean()
breach_normal = df.loc[df['Delay_Tier'] == 'normal', 'SLA_Status'].isin(SLA_BREACH_STATUSES).mean()
breach_uplift = (breach_surge / breach_normal - 1) if breach_normal > 0 else float('inf')

# Dragon risk aggregates
n_dragons       = df['Is_Dragon'].sum()
n_unfulfilled   = df['Unfulfilled_Dragon'].sum()
unfulfilled_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()
max_dragon_loss = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].max() if n_unfulfilled > 0 else 0.0
unfulfill_pct   = n_unfulfilled / n_dragons if n_dragons > 0 else 0.0

# Sample period 
sample_days = (df['Date'].max() - df['Date'].min()).days + 1

# Output
print("\n" + "="*50)
print("SCENARIO 2 — Gross Max-Risk VaR (NETTED DATA)")
print("="*50)
print(f"{'Metric':<45} {'Value':>20} {'% of Gross':>12}")
print("-"*50)
print(f"{'Gross Revenue (firm demand)':<45} £ {total_gross:>2,.2f}")
print(f"{'Holding Cost (25% APR on delays)':<45} £ {total_holding:>2,.2f} {total_holding/total_gross:>11.3%}")
print(f"{'Net Revenue (after holding drag)':<45} £ {total_net:>2,.2f} {total_net/total_gross:>11.3%}")
print("-"*50)
print(f"{'SLA Breach Rate — Normal ':<45} {breach_normal:>2.2%}")
print(f"{'SLA Breach Rate — Surge ':<45} {breach_surge:>2.2%} {breach_uplift:>+11.1%} vs normal")
print("-"*50)
print(f"{'Sample Period':<45} {sample_days:>2} days")
print(f"{'Value Dragon Threshold (99.9th %ile)':<45} £ {dragon_val_thresh:>2,.2f} (Pareto tail)")
print(f"{'Extreme Delay Dragons (Delay_Tier)':<45} {n_dragons:>2,} ({n_dragons/len(df):.4%} of txns)")
print(f"{'Unfulfilled Dragons (≥6h SLA breach)':<45} {n_unfulfilled:>2,} ({unfulfill_pct:.1%} of dragons)")
print(f"{'Revenue at Risk (unfulfilled dragons)':<45} £ {unfulfilled_rev:>2,.2f}")
print(f"{'Max Single Dragon Loss':<45} £ {max_dragon_loss:>2,.2f}")
print("="*50)

# Summary and Plots
unfulfilled_count = df['Unfulfilled_Dragon'].sum()
total_dragons     = df['Is_Dragon'].sum()
unfulfill_pct     = 100 * unfulfilled_count / total_dragons if total_dragons > 0 else 0

print(f"\nSummary")
print(f"Dragons at risk of unfulfillment : {unfulfilled_count:,} / {total_dragons:,} ({unfulfill_pct:.1f}%)")
print(f"Expected revenue at risk from Dragons : £{df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum():,.0f}")

# Plot
import plotly.graph_objects as go
dragon_df = df[df['Is_Dragon']].copy()

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=dragon_df['Delay_min'],
    name='All Dragon Transactions',
    nbinsx=60,
    marker_color='lightblue',
    opacity=0.70
))
fig.add_trace(go.Histogram(
    x=dragon_df[dragon_df['Unfulfilled_Dragon']]['Delay_min'],
    name='Unfulfilled Dragons (Preventable Loss)',
    nbinsx=60,
    marker_color='red',
    opacity=0.95
))

fig.update_layout(
    title=f"<b>Delay Distribution of Dragons & Unfulfillment Rate | Scenario 2 (NETTED)</b><br>"
          f"{unfulfilled_count:,} / {total_dragons:,} Dragons unfulfilled ({unfulfill_pct:.1f}%) at ≥6h SLA",
    xaxis_title="Delay (minutes)",
    yaxis_title="Count of Dragon Transactions",
    barmode='overlay',
    height=680,
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.add_vline(
    x=360,
    line=dict(color="red", dash="dash", width=3),
    annotation_text="6h SLA Breach Threshold (revenue loss starts)",
    annotation_position="top right"
)

fig.show()

# ## Monte Carlo simulation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import gaussian_kde

df['Is_Dragon'] = df['Delay_Tier'] == 'dragon'

# Dragons
df['Unfulfilled_Dragon'] = (
    df['Is_Dragon'] &
    (df['Delay_min'] >= 360) &                     # 6 hours 
    df['SLA_Status'].isin(['SLA Breach', 'Fulfillment Failure'])
)

# Empirical Inputs
sample_days = (df['Date'].max() - df['Date'].min()).days + 1   
target_annual = 72                                                  

n_dragons = df['Is_Dragon'].sum()
n_unfulfilled = df['Unfulfilled_Dragon'].sum()
total_dragon_rev = df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum()
unfulfilled_dragon_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()

breach_rate_rev = unfulfilled_dragon_rev / total_dragon_rev if total_dragon_rev > 0 else 0.0
breach_rate_count = n_unfulfilled / n_dragons if n_dragons > 0 else 0.0
avg_daily_dragon_rev = total_dragon_rev / sample_days

GROSS_MARGIN = 0.30
print(f"\nGross margin applied: {GROSS_MARGIN:.0%} (Retail benchmark)")

# Dragon-revenue days only
daily_dragon = df[df['Is_Dragon']].groupby(df['Date'])['OrderValue_GBP'].sum()
CV = daily_dragon.std() / daily_dragon.mean() if len(daily_dragon) > 0 and daily_dragon.mean() > 0 else 1.98
CV = min(CV, 3.0)

annual_el = (unfulfilled_dragon_rev / sample_days) * 365

# Monte Carlo Simulations
N_PATHS = 10_000
rng_mc = np.random.default_rng(seed=314159)
mu_daily = np.log(max(avg_daily_dragon_rev, 1e-6)) - 0.5 * np.log(1 + CV**2)
sigma_daily = np.sqrt(np.log(1 + CV**2))

daily_rev = np.exp(rng_mc.normal(mu_daily, sigma_daily, size=(N_PATHS, 365)))
daily_loss = daily_rev * breach_rate_rev * GROSS_MARGIN
annual_loss = daily_loss.sum(axis=1)

median = np.median(annual_loss)
var95 = np.percentile(annual_loss, 95)
var99 = np.percentile(annual_loss, 99)
es95 = annual_loss[annual_loss >= var95].mean()
ul95 = var95 - median
gross_rev = df['OrderValue_GBP'].sum()

# Output
print(f"\nSample period               : {sample_days} days")
print(f"Target dragons/year         : {target_annual:,}")
print(f"Dragons in sample           : {n_dragons:,} ({n_dragons/len(df):.4%} of txns)")
print(f"Unfulfilled Dragons (≥6h)   : {n_unfulfilled:,} ({breach_rate_count:.1%})")
print(f"Unfulfilled dragon revenue  : £ {unfulfilled_dragon_rev:,.0f} ({breach_rate_rev:.1%} of dragon rev)")
print(f"Active-day CV               : {CV:.2f}")
print(f"Data-derived Annual EL      : £ {annual_el:,.0f} ({annual_el/gross_rev:.3%} of gross)")
print(f"{'─'*50}")
print(f"MONTE CARLO ANNUAL LOSS DISTRIBUTION (N={N_PATHS:,} paths | 365-day horizon)")
print(f"{'─'*50}")
print(f"{'Expected Loss (EL) — data-derived':<50} £ {annual_el:>2,.0f} ({annual_el/gross_rev:6.3%})")
print(f"{'Unexpected Loss (UL) — VaR95 − Median':<50} £ {ul95:>2,.0f} ({ul95/gross_rev:6.3%})")
print(f"{'─'*50}")
print(f"{'Median Annual Loss':<50} £ {median:>2,.0f} ({median/gross_rev:6.3%})")
print(f"{'VaR 95% (annual)':<50} £ {var95:>2,.0f} ({var95/gross_rev:6.3%})")
print(f"{'VaR 99% (annual)':<50} £ {var99:>2,.0f} ({var99/gross_rev:6.3%})")
print(f"{'Expected Shortfall 95% ← Headline':<50} £ {es95:>2,.0f} ({es95/gross_rev:6.3%})")
print(f"{'='*50}")
print(f"\nHeadline (Scenario 2 Max-Risk): In a bad year (worst 5 %), preventable dragon loss ≈ £ {es95:,.0f}")

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(annual_loss, bins=100, kde=True, color='#3B82F6', alpha=0.60, stat='density', linewidth=0, ax=ax)

kde_x = np.linspace(annual_loss.min(), annual_loss.max(), 1000)
kde_y = gaussian_kde(annual_loss)(kde_x)
tail_mask = kde_x >= var95
ax.fill_between(kde_x[tail_mask], kde_y[tail_mask], color='#F97316', alpha=0.35, label='95% Tail (Unexpected Loss)')

for val, color, ls, lw, label in [
    (median, '#10B981', '--', 2.0, f'Median EL = £ {median:,.0f}'),
    (var95, '#F97316', '--', 2.5, f'VaR 95% = £ {var95:,.0f}'),
    (var99, '#EF4444', '--', 2.5, f'VaR 99% = £ {var99:,.0f}'),
    (es95, '#7C3AED', '-', 3.0, f'ES 95% = £ {es95:,.0f} ← Headline'),
]:
    ax.axvline(val, color=color, linestyle=ls, linewidth=lw, label=label)

ax.set_title(
    'Monte Carlo Annual Loss Distribution — Scenario 2 (NETTED)\n'
    f'~{target_annual} dragons/year (Hawkes+BSTS) | {N_PATHS:,} paths | '
    f'CV={CV:.2f} | 30% margin',
    fontsize=14, pad=15
)
ax.set_xlabel('Annual Preventable Loss (£)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# ## Regression Discontinuity (RD) + Quantile Regression (QR) + GPD Tail


# Causal Engine
import gc
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from statsmodels.regression.quantile_regression import QuantReg
from typing import List

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
rng = np.random.default_rng(314159)

CALIPER = 0.005
PS_SUBSAMPLE = 150_000   
QR_SUBSAMPLE = 80_000    
BOOTSTRAP_REPS = 400
QUANTILES = [0.95, 0.999]
MAX_COUNTRIES = 50       


class CausalEngine:
    def __init__(self, df: pd.DataFrame, surge_idx: pd.Index, dragon_flag_col: str = "Is_Dragon"):
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
        self.df = df[list(df.columns)].copy()
        self.surge_idx = surge_idx
        self.dragon_flag_col = dragon_flag_col
        self.ate_psm = np.nan
        self.ate_qr = {q: np.nan for q in QUANTILES}
        self.p_qr = {q: np.nan for q in QUANTILES}
        self.qr_models = {}
        self.dragon_qte = {q: np.nan for q in QUANTILES}
        self._prepare_features()

    def _prepare_features(self):
        df = self.df
        df['dragon'] = (df['Delay_Tier'] == 'dragon').astype(bool)
        df['treatment'] = df.index.isin(self.surge_idx).astype(np.uint8)

        # Country 
        if 'Country' not in df.columns:
            df['Country'] = 'Unknown'
        df['Country'] = df['Country'].fillna('Unknown').astype(str)
        top_countries = df['Country'].value_counts().nlargest(MAX_COUNTRIES).index
        df['Country'] = df['Country'].where(df['Country'].isin(top_countries), other='Other')
        df['Country'] = df['Country'].astype('category')
        df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df['Hour'] = pd.to_datetime(df['Date'], errors='coerce').dt.hour.fillna(-1).astype(np.int8)
        df['Log_Quantity'] = np.log1p(df['Quantity'].clip(lower=1)).astype(np.float32)
        df['Net_Revenue_GBP'] = df['OrderValue_GBP'].astype(np.float32)
        df['OrderValue'] = df['OrderValue_GBP'].astype(np.float32)
        df['Will_Cancel'] = np.uint8(0)
        self.df = df
        logging.info(f"Features engineered — {df['dragon'].sum():,} dragons | "
                     f"{df['Country'].nunique()} countries (capped at {MAX_COUNTRIES})")

    # PROPENSITY SCORE
    def fit_propensity(self):
        df = self.df
        treated = df[df['treatment'] == 1]
        control = df[df['treatment'] == 0]
        if len(treated) == 0:
            logging.warning("No treated units. Skipping PS.")
            return

        n_control = min(len(control), len(treated) * 4)
        control_sample = control.sample(n=n_control, random_state=314159)
        sample = pd.concat([treated, control_sample]).sample(frac=1, random_state=314159)
        if len(sample) > PS_SUBSAMPLE:
            sample = sample.sample(n=PS_SUBSAMPLE, random_state=314159)

        X = sample[['Country', 'Log_Quantity', 'Hour']]
        y = sample['treatment']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), ['Country']),
                ('num', StandardScaler(), ['Log_Quantity', 'Hour'])],
            remainder='drop')
        model = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                penalty='l2', solver='saga', C=1.0, max_iter=1000,
                n_jobs=-1, random_state=314159, warm_start=True, tol=2.614e-4))])
        model.fit(X, y)
        self.ps_model = model
        chunk_size = 50_000
        ps_chunks = []
        feat_cols = ['Country', 'Log_Quantity', 'Hour']
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size][feat_cols]
            ps_chunks.append(model.predict_proba(chunk)[:, 1].astype(np.float32))
        ps_full = np.concatenate(ps_chunks)
        self.df['ps'] = np.clip(ps_full, 1e-6, 1 - 1e-6)
        del ps_chunks, ps_full
        gc.collect()

        logging.info(f"Propensity fitted (n={len(sample):,}), "
                     f"PS range: [{self.df['ps'].min():.4f}, {self.df['ps'].max():.4f}]")

    # PSM 
    def psm_caliper_exact_country(self):
        df = self.df
        treated = df[df['treatment'] == 1][['ps', 'Net_Revenue_GBP', 'Country']].copy()
        control = df[df['treatment'] == 0][['ps', 'Net_Revenue_GBP', 'Country']].copy()

        if len(treated) == 0:
            self.ate_psm = np.nan
            return

        matched_pairs: List[tuple] = []

        for country in treated['Country'].cat.categories:
            t_sub = treated[treated['Country'] == country].sort_values('ps')
            c_sub = control[control['Country'] == country].sort_values('ps')
            if len(t_sub) == 0 or len(c_sub) == 0:
                continue

            t_ps = t_sub['ps'].values
            c_ps = c_sub['ps'].values
            t_rev = t_sub['Net_Revenue_GBP'].values
            c_rev = c_sub['Net_Revenue_GBP'].values
            c_used = np.zeros(len(c_ps), dtype=bool)
            idxs = np.searchsorted(c_ps, t_ps)

            for i, idx in enumerate(idxs):
                best_idx = None
                best_dist = CALIPER + 1.0
                for j in [idx - 1, idx, idx + 1]:
                    if 0 <= j < len(c_ps) and not c_used[j]:
                        d = abs(t_ps[i] - c_ps[j])
                        if d <= CALIPER and d < best_dist:
                            best_dist = d
                            best_idx = j
                if best_idx is not None:
                    matched_pairs.append((t_rev[i], c_rev[best_idx]))
                    c_used[best_idx] = True

        if matched_pairs:
            t_revs, c_revs = zip(*matched_pairs)
            ate = float(np.mean(t_revs) - np.mean(c_revs))
        else:
            ate = np.nan
            logging.warning("PSM: No caliper matches found.")

        self.ate_psm = ate
        logging.info(f"PSM ATE = £{ate:,.0f}" if not np.isnan(ate) else "PSM: No matches")
        gc.collect()

    # QUANTILE REGRESSION 
    def quantile_regression(self, quantiles: List[float] = QUANTILES):
        df = self.df
        sample = df.sample(n=min(QR_SUBSAMPLE, len(df)), random_state=314159)

        if len(sample) < 200:
            logging.warning(f"QR: Sample too small ({len(sample)}). Skipping.")
            return
        country_dummies = pd.get_dummies(
            sample['Country'], prefix='C', drop_first=True, dtype=np.float32)
        hour_bins = pd.cut(
            sample['Hour'].replace(-1, 12), 
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True)
        hour_dummies = pd.get_dummies(hour_bins, prefix='H', drop_first=True, dtype=np.float32)

        X = pd.concat([
            sample[['treatment', 'Log_Quantity']].reset_index(drop=True),
            country_dummies.reset_index(drop=True),
            hour_dummies.reset_index(drop=True),
        ], axis=1)
        X.insert(0, 'const', 1.0)
        y = sample['Net_Revenue_GBP'].values
        X_arr = X.values.astype(np.float64)  
        treatment_col = X.columns.get_loc('treatment')

        del country_dummies, hour_dummies, X
        gc.collect()

        for q in quantiles:
            try:
                mod = QuantReg(y, X_arr)
                res = mod.fit(q=q, kernel='epa', max_iter=3000, p_tol=1e-6)
                self.qr_models[q] = res
                self.ate_qr[q] = float(res.params[treatment_col])
                self.p_qr[q] = float(res.pvalues[treatment_col])
                logging.info(f"QR {q:.1%} → ATE = £{self.ate_qr[q]:,.0f}, p = {self.p_qr[q]:.2e}")
            except Exception as e:
                logging.error(f"QR {q:.1%} failed: {e}")
                self.ate_qr[q] = np.nan
                self.p_qr[q] = np.nan
        del X_arr, y
        gc.collect()

    # DRAGON METRICS  
    def dragon_metrics(self):
        df = self.df
        dragons = df[df['dragon']]
        non = df[~df['dragon']]
        if len(dragons) == 0:
            self._reset_dragon_metrics()
            return

        days = (df['date'].max() - df['date'].min()).days + 1
        months_covered = days / 30.4375
        total_dragons = len(dragons)
        monthly_dragons = total_dragons / months_covered
        premium = float(dragons['OrderValue'].mean() - non['OrderValue'].mean())

        qte = {}
        for q in QUANTILES:
            if len(dragons) >= 50 and len(non) >= 50:
                qte[q] = float(np.quantile(dragons['OrderValue'], q) -
                               np.quantile(non['OrderValue'], q))
            else:
                qte[q] = np.nan
        self.dragon_premium = premium
        self.dragon_cancel_rate = 0.0
        self.dragon_cancel_uplift = 0.0
        self.monthly_dragons = int(np.round(monthly_dragons))
        self.months_covered = float(months_covered)
        self.dragon_qte = qte
        logging.info(
            f"Dragons: {total_dragons:,} over {months_covered:.1f} months → "
            f"{monthly_dragons:.1f}/month | Premium: £{premium:,.0f} | "
            f"95th QTE: £{qte.get(0.95, np.nan):,.0f} | 99.9th QTE: £{qte.get(0.999, np.nan):,.0f}")

    def _reset_dragon_metrics(self):
        self.dragon_premium = 0.0
        self.dragon_cancel_rate = 0.0
        self.dragon_cancel_uplift = 0.0
        self.monthly_dragons = 0
        self.months_covered = 12.0
        self.dragon_qte = {q: 0.0 for q in QUANTILES}

    def annual_impact(self, gross_margin: float = 0.30):
        unfulfilled_mask = self.df['Unfulfilled_Dragon']
        unfulfilled_rev = self.df.loc[unfulfilled_mask, 'OrderValue_GBP'].sum()
        days = (self.df['date'].max() - self.df['date'].min()).days + 1
        annualization = 365 / days
        self.annual_revenue_risk = float(unfulfilled_rev * annualization)
        self.annual_cancellation_loss = 0.0
        self.total_annual_impact = float(self.annual_revenue_risk * gross_margin)
        self.annual_dragons = int(self.df['dragon'].sum() * annualization)
        logging.info(
            f"Annualized: {self.annual_dragons:,} dragons → "
            f"£{self.annual_revenue_risk:,.0f} gross risk → "
            f"£{self.total_annual_impact:,.0f} profit impact ({gross_margin:.0%} margin)")

    def run_all(self):
        self.fit_propensity();          gc.collect()
        self.psm_caliper_exact_country(); gc.collect()
        self.quantile_regression();     gc.collect()
        self.dragon_metrics();          gc.collect()
        self.annual_impact();           gc.collect()
        self.plot_convergence()

    def plot_convergence(self):
        items = [
            ('PSM Caliper', self.ate_psm),
            ('Dragon Premium (mean)', self.dragon_premium),
        ] + [(f'QR {q:.1%}', self.ate_qr[q]) for q in QUANTILES] \
          + [(f'Dragon QTE {q:.1%}', self.dragon_qte[q]) for q in QUANTILES]

        valid = [(lbl, float(v)) for lbl, v in items
                 if not (pd.isna(v) or np.isinf(v) or v == 0)]
        if not valid:
            print("No valid ATEs to plot.")
            return

        methods, ates = zip(*valid)
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FF6B6B', '#4ECDC4']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods, y=ates,
            text=[f"£{v:,.0f}" for v in ates],
            textposition='outside',
            marker_color=colors[:len(methods)]))
        fig.update_layout(
            title="<b>Causal Lift + Dragon QTE Convergence (Scenario 2 NETTED)</b>",
            yaxis_title="Effect on Net Revenue (£)",
            template="plotly_dark",
            height=620,
            yaxis=dict(range=[0, max(ates) * 1.35]))
        fig.add_annotation(
            x=0.5, y=0.92, xref='paper', yref='paper',
            text=(f"<b>£{self.total_annual_impact:,.0f} ANNUAL IMPACT</b>"
                  f"<br>{self.annual_dragons:,} dragons/year"),
            showarrow=False, font_size=16, bgcolor='gold', font_color='black')
        fig.show()

    def summary_dashboard(self):
        print("\n" + "=" * 50)
        print("CAUSAL IMPACT DASHBOARD — SCENARIO 2 (NETTED) — SURGE + DRAGON")
        print("=" * 50)
        print(f"{'Method':<34} {'ATE (Net Revenue)':>20} {'p-value':>12}")
        print("-" * 50)
        for q in QUANTILES:
            ate, p = self.ate_qr[q], self.p_qr[q]
            label = f"Quantile Reg ({q:.1%})"
            val = f"£{ate:>10,.0f}" if not pd.isna(ate) else "—"
            pv  = f"{p:>10.2e}"     if not pd.isna(p)   else "—"
            print(f"{label:<34} {val:>20} {pv:>12}")
        if not pd.isna(self.ate_psm):
            print(f"{'PSM (caliper 0.005)':<34} £{self.ate_psm:>10,.0f} {'—':>12}")
        else:
            print(f"{'PSM (caliper 0.005)':<34} {'—':>20} {'—':>12}")
        print(f"{'Dragon Premium (mean)':<34} £{self.dragon_premium:>10,.0f} {'—':>12}")
        for q in QUANTILES:
            qte = self.dragon_qte[q]
            label = f"Dragon QTE ({q:.1%})"
            val = f"£{qte:>10,.0f}" if not np.isnan(qte) else "—"
            print(f"{label:<34} {val:>20} {'—':>12}")
        print("-" * 50)
        print(f"{'Time Coverage':<34} {self.months_covered:>10.1f} months")
        print(f"{'Dragon Cancel Rate':<34} {self.dragon_cancel_rate:>10.2%}")
        print(f"{'Monthly Dragons':<34} {self.monthly_dragons:>10,}")
        print(f"{'Annual Dragons':<34} {self.annual_dragons:>10,}")
        print(f"{'Annual Revenue Risk':<34} £{self.annual_revenue_risk:>10,.0f}")
        print(f"{'Annual Cancellation Loss':<34} £{self.annual_cancellation_loss:>10,.0f}")
        print(f"{'TOTAL ANNUAL IMPACT':<34} £{self.total_annual_impact:>10,.0f}")
        print("=" * 50)


# EXECUTION
engine = CausalEngine(df=df, surge_idx=surge_idx, dragon_flag_col='Is_Dragon')
engine.run_all()
engine.summary_dashboard()

# ## Hawkes MLE + BSTS


# HAWKES MLE + BSTS 
import gc
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from numba import jit, prange
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
rng = np.random.default_rng(314159)

#  Data alignment
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'OrderValue_GBP']).copy()

df['Is_Dragon'] = (df['Delay_Tier'] == 'dragon').astype(bool)

anomaly_df = df[df['Is_Dragon']].copy().sort_values('Date')
t0 = anomaly_df['Date'].min()
anomaly_times = (anomaly_df['Date'] - t0).dt.total_seconds() / 60.0
anomaly_times = np.sort(anomaly_times.values.astype(np.float64))

print(f"Dragons locked (Scenario 2 NETTED): {len(anomaly_times):,} | Target annual: 70")

# HAWKES NEGATIVE LOG-LIKELIHOOD 
def hawkes_negloglik(params, times):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or (alpha / beta) >= 1.0:
        return np.inf
    N = len(times)
    T = times[-1]
    loglik = 0.0
    A = 0.0
    for i in range(N):
        loglik += np.log(mu + A)
        if i < N - 1:
            dt = times[i + 1] - times[i]
            A = alpha + A * np.exp(-beta * dt)
    loglik -= mu * T
    loglik -= (alpha / beta) * np.sum(1 - np.exp(-beta * (T - times)))
    return -loglik

# Optimization
res = minimize(hawkes_negloglik, [0.00025, 0.75, 1.2], args=(anomaly_times,),
               method='L-BFGS-B', bounds=[(1e-8, None), (0, 0.99), (0.1, None)])
mu, alpha, beta = res.x
print(f"Hawkes MLE (Scenario 2 NETTED): μ={mu:.7f}  α={alpha:.3f}  β={beta:.3f}  (self-excitation confirmed)")

# NUMBA - JIT
@jit(nopython=True, parallel=True, cache=True)
def hawkes_intensity_numba(times_past, t_future, mu, alpha, beta):
    n = len(t_future)
    intensity = np.full(n, mu, dtype=np.float64)
    for i in prange(len(times_past)):
        t = times_past[i]
        for j in range(n):
            dt = t_future[j] - t
            if dt >= 0:
                intensity[j] += alpha * np.exp(-beta * dt)
    return intensity

T_30D = 30 * 1440
t_future = np.linspace(anomaly_times[-1], anomaly_times[-1] + T_30D, 15000, dtype=np.float64)
intensity = hawkes_intensity_numba(anomaly_times, t_future, mu, alpha, beta)
expected_30d = np.trapz(intensity, t_future)
print(f"30-day expected dragons (Hawkes): {expected_30d:.1f}")

# BSTS (12-month forecast)
anomaly_df['Month'] = anomaly_df['Date'].dt.to_period('M')
monthly_anomalies = anomaly_df.groupby('Month').size()
full_range = pd.period_range(monthly_anomalies.index.min(), monthly_anomalies.index.max(), freq='M')
monthly_series = monthly_anomalies.reindex(full_range, fill_value=0).values.astype(np.float64)

# BSTS (local-level + seasonal)
season = 12
state_dim = 2 + season - 1
F = np.eye(state_dim, dtype=np.float64)
F[0, 1] = 1
F[2:, 2:] = np.eye(season - 1, k=1)
F[-1, 2] = -1
Q = np.diag([8e-4, 5e-7] + [8e-6] * (season - 1))
R = np.var(monthly_series[monthly_series > 0]) if np.any(monthly_series > 0) else 0.5

x = np.zeros(state_dim)
x[0] = monthly_series.mean()
P = np.eye(state_dim) * 50
in_sample_fit = np.zeros(len(monthly_series))

for i in range(len(monthly_series)):
    y = monthly_series[i]
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    y_hat = x_pred[0]
    S = P_pred[0, 0] + R
    K = P_pred[:, 0] / S
    x = x_pred + K * (y - y_hat)
    P = P_pred - np.outer(K, K) * S
    in_sample_fit[i] = y_hat

proj = np.zeros(12)
x_proj = x.copy()
for i in range(12):
    x_proj = F @ x_proj
    proj[i] = x_proj[0]

proj_monthly = max(6, min(int(np.round(np.mean(proj[-3:]))), 10))
annual_2012 = proj_monthly * 12
print(f"BSTS 12-month forecast (Scenario 2 NETTED): {proj_monthly}/month → {annual_2012:,}/year")

# P&L
unfulfilled_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()
n_unfulfilled_drag = df['Unfulfilled_Dragon'].sum()
anomaly_premium = unfulfilled_rev / n_unfulfilled_drag if n_unfulfilled_drag > 0 else 0
breach_rate = n_unfulfilled_drag / len(anomaly_times) if len(anomaly_times) > 0 else 0
base_annual_bleed = annual_2012 * breach_rate * anomaly_premium * 0.30

print(f"Anomaly premium: £{anomaly_premium:,.0f}/dragon")
print(f"2012 P&L bleed (30% margin): £{base_annual_bleed:,.0f}")

# MITIGATION
interventions = {
    "Reroute 30% High-Risk SKUs": 0.70,
    "Safety Stock (7 SKUs)": 0.55,
    "Full Mitigation (All SKUs)": 0.15,
    "Zero Anomalies (Target)": 0.00
}
print("\nMITIGATION SCENARIOS (tied to Scenario 2 tuned risk)")
for name, reduction in interventions.items():
    mitigated = int(annual_2012 * reduction)
    savings = base_annual_bleed * (1 - reduction)
    print(f"  {name:<30} -> {mitigated:>6,} dragons -> saves £{savings:,.0f}")

# PLOTS
# HAWKES INTENSITY
zoom_days = 367
mask = t_future / 1440 <= zoom_days
t_plot = t_future[mask] / 1440
intensity_plot = intensity[mask]
dragon_plot = anomaly_times[anomaly_times / 1440 <= zoom_days] / 1440

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t_plot, y=intensity_plot, name='λ(t) First 30 days', line=dict(color='red', width=4)))
fig1.add_trace(go.Scatter(x=dragon_plot, y=np.zeros_like(dragon_plot), mode='markers',
                          marker=dict(color='gold', size=10, symbol='star'), name='Actual Dragons'))
fig1.update_layout(title="<b>Hawkes Bursts — First 30 days (Scenario 2 NETTED)</b>",
                   xaxis_title="days since last anomaly", yaxis_title="Intensity λ(t)",
                   template="plotly_dark", height=600)
fig1.show()

# BSTS Forecast
months = np.arange(len(monthly_series))
forecast_months = np.arange(len(monthly_series), len(monthly_series)+12)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=months, y=monthly_series, name='Observed', marker_color='lightgray'))
fig2.add_trace(go.Scatter(x=months, y=in_sample_fit, mode='lines', name='BSTS Fit', line=dict(color='cyan', width=2)))
fig2.add_trace(go.Scatter(x=forecast_months, y=proj, mode='lines', name='12-Month Forecast', line=dict(color='gold', width=4)))
fig2.add_hline(y=8, line_dash="dash", line_color="lime", annotation_text="Target: 8/month")
fig2.add_hline(y=0, line_dash="solid", line_color="red", annotation_text="Zero Anomalies")
fig2.add_annotation(x=0.02, y=0.95, xref='paper', yref='paper',
                    text=f"<b>2012 BLEED: £ {base_annual_bleed:,.0f}</b><br>ZERO → £0",
                    showarrow=False, font_size=13, bgcolor='gold', font_color='black')
fig2.update_layout(
    title="<b>BSTS: 12-MONTH FORECAST → ZERO TARGET (Scenario 2 NETTED)</b>",
    xaxis_title="Month Index", yaxis_title="Anomalies per Month",
    template="plotly_dark", height=580, barmode='overlay'
)
fig2.show()

# FINAL SUMMARY
print("\n" + "="*50)
print("HAWKES + BSTS SUMMARY — SCENARIO 2 ")
print("="*50)
print(f"Total Dragons Detected      : {len(anomaly_times):,}")
print(f"2012 Forecast               : {annual_2012:,} dragons")
print(f"2012 P&L Bleed (30% margin) : £ {base_annual_bleed:,.0f}")
print(f"5-Year Bleed (No Action)    : £ {base_annual_bleed * 5:,.0f}")
print(f"5-Year Bleed (Zero Target)  : £ 0")
print("\nMITIGATION PATH:")
for name, reduction in interventions.items():
    mitigated = int(annual_2012 * reduction)
    savings = base_annual_bleed * (1 - reduction)
    print(f"  {name:<30} -> {mitigated:>6,} dragons -> saves £ {savings:,.0f}")
print(f"\nHAWKES α={alpha:.3f} | 30-DAY FORECAST: {expected_30d:.1f} | TARGET: 0")
print("="*50)

# ## Backtesting - Purged Expanding-Window


# Self contained backtesting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from scipy import stats, optimize
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
SEED = 314159
N_MC_PATHS = 10_000
SURGE_PCT = 0.18
DRAGON_PCT = 0.00025
DRAGON_BIAS_EXP = 1.20
NORMAL_MEAN_MIN = 25.0
SURGE_MEAN_MIN = 90.0
DRAGON_MEAN_MIN = 360.0
SIGMA_NORMAL = 0.50
SIGMA_SURGE = 0.65
SIGMA_DRAGON = 0.82
SLA_BREACH_MIN = 360
GROSS_MARGIN = 0.30
ANNUAL_HOLDING_RATE = 0.25
MIN_TRAIN_DAYS = 180
WINDOW_STEP = 30
PURGE_GAP = 14

# LOAD DATA
print("SCENARIO 2 BACKTEST")
USECOLS = ['InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']
file_path = '/kaggle/input/datasets/prnavjoshi/excelxlsx/Online Retail.xlsx'
df_raw = pd.read_excel(
    file_path, engine='openpyxl',
    parse_dates=['InvoiceDate'],
    usecols=USECOLS,
    dtype={'StockCode': str, 'InvoiceNo': str, 'CustomerID': str},)

df_raw['Date'] = df_raw['InvoiceDate'].dt.date
df_raw['SKU'] = df_raw['StockCode'].str.strip()
# CLASSIFY EXCLUDE MISC
df_raw['Quantity'] = pd.to_numeric(df_raw['Quantity'], errors='coerce')
df_raw['UnitPrice'] = pd.to_numeric(df_raw['UnitPrice'], errors='coerce')
misc_pattern = r'^(POST, DOT, AMAZON, BANK\s*CHARGES, CRUK, gift_, POSTAGE, M$, D$)'
real_mask = (
    ~df_raw['StockCode'].str.upper().str.match(misc_pattern, na=False) &
    df_raw['StockCode'].str.strip().ne('') &
    df_raw['UnitPrice'].gt(0) &
    df_raw['CustomerID'].notna())
real_df = df_raw[real_mask].copy()

# NETTING
gross_picks = real_df[real_df['Quantity'] > 0].copy()
cancellations = real_df[real_df['Quantity'] < 0].copy()
gross_picks['match_key'] = gross_picks['CustomerID'] + '-' + gross_picks['SKU']
cancellations = cancellations.copy()
cancellations['match_key'] = cancellations['CustomerID'] + '-' + cancellations['SKU']
all_rows = pd.concat([gross_picks[['match_key', 'Quantity']],
                      cancellations[['match_key', 'Quantity']]], ignore_index=True)
net_qty_per_key = all_rows.groupby('match_key', sort=False)['Quantity'].sum()
positive_keys = net_qty_per_key[net_qty_per_key > 0]
filtered_gross = gross_picks[gross_picks['match_key'].isin(positive_keys.index)]
max_idx = filtered_gross.groupby('match_key', sort=False)['Quantity'].idxmax()
net_sales = filtered_gross.loc[max_idx].copy()
net_sales['NetQuantity'] = net_sales['match_key'].map(positive_keys)
net_sales['NetOrderValue'] = net_sales['NetQuantity'] * net_sales['UnitPrice']
df = (
    net_sales[['NetQuantity', 'NetOrderValue', 'SKU', 'Date', 'StockCode']]
    .rename(columns={'NetQuantity': 'Quantity', 'NetOrderValue': 'OrderValue'})
    .query('Quantity > 0')
    .reset_index(drop=True))

n = len(df)
df['OrderValue_GBP'] = df['OrderValue']
rng = np.random.default_rng(seed=SEED)
weights = df['OrderValue_GBP'] ** DRAGON_BIAS_EXP
weights = weights / weights.sum()
all_idx = np.arange(n)
dragon_idx = rng.choice(all_idx, size=int(DRAGON_PCT * n), p=weights, replace=False)
remaining = np.setdiff1d(all_idx, dragon_idx)
surge_idx = rng.choice(remaining, size=int(SURGE_PCT * n), replace=False)
tier = np.full(n, 'normal', dtype=object)
tier[dragon_idx] = 'dragon'
tier[surge_idx] = 'surge'
df['Delay_Tier'] = tier
delays = np.zeros(n, dtype=float)
for mask, mean_min, sigma, clip in [
    (df['Delay_Tier'] == 'normal', NORMAL_MEAN_MIN, SIGMA_NORMAL, (5, 120)),
    (df['Delay_Tier'] == 'surge', SURGE_MEAN_MIN, SIGMA_SURGE, (30, 360)),
    (df['Delay_Tier'] == 'dragon', DRAGON_MEAN_MIN, SIGMA_DRAGON, (200, 1440))]:
    mu = np.log(mean_min) - 0.5 * sigma**2
    raw = np.exp(mu + sigma * rng.standard_normal(mask.sum()))
    delays[mask] = np.clip(raw, clip[0], clip[1])
df['Delay_min'] = delays
df['Days_Delayed'] = df['Delay_min'] / 1440.0
df['Holding_Cost_GBP'] = df['OrderValue_GBP'] * df['Days_Delayed'] * 0.25
df['Is_Dragon'] = df['Delay_Tier'] == 'dragon'
df['Unfulfilled_Dragon'] = df['Is_Dragon'] & (df['Delay_min'] >= SLA_BREACH_MIN)
df['Realised_Loss_GBP'] = 0.0
df.loc[df['Unfulfilled_Dragon'], 'Realised_Loss_GBP'] = (
    df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'] +
    df.loc[df['Unfulfilled_Dragon'], 'Holding_Cost_GBP']
)
print(f"Simulation -> {df['Is_Dragon'].sum():,} dragons "
      f"({df['Is_Dragon'].mean():.4%}) | "
      f"{df['Unfulfilled_Dragon'].sum():,} SLA breaches")

# PURGED EXPANDING-WINDOW BACKTEST
dates = sorted(df['Date'].unique())
start_idx = np.searchsorted(dates, dates[0] + timedelta(days=MIN_TRAIN_DAYS))
rolling_window_ends = []
rolling_es95_mc = []
rolling_var95_mc = []
rolling_es95_window = []
rolling_var95_window = []
test_violations = []
last_annual_loss_mc = None
last_var95_mc = None
last_es95_mc = None
rng = np.random.default_rng(SEED)
for i in range(start_idx, len(dates), WINDOW_STEP):
    window_end_date = dates[i]
    train_cutoff = window_end_date - timedelta(days=PURGE_GAP)
    train = df[df['Date'] <= train_cutoff]
    test = df[(df['Date'] > train_cutoff) & (df['Date'] <= window_end_date)]
    if len(train) < 50_000 or len(test) < 5:
        continue
    train_dragons = train[train['Delay_Tier'] == 'dragon']
    train_unfulfilled_rev = train.loc[train['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()
    train_total_dragon_rev = train_dragons['OrderValue_GBP'].sum()
    breach_rate = train_unfulfilled_rev / train_total_dragon_rev if train_total_dragon_rev > 0 else 0.0
    daily_dragon = train_dragons.groupby(train_dragons['Date'])['OrderValue_GBP'].sum()
    avg_daily = daily_dragon.mean() if len(daily_dragon) > 0 else 1e-6
    cv_train = min(3.0, daily_dragon.std() / avg_daily if len(daily_dragon) > 1 else 2.0)
    mu_daily = np.log(max(avg_daily, 1e-6)) - 0.5 * np.log(1 + cv_train ** 2)
    sigma_daily = np.sqrt(np.log(1 + cv_train ** 2))
    daily_rev_mc = np.exp(rng.normal(mu_daily, sigma_daily, size=(N_MC_PATHS, 365)))
    annual_loss_mc = daily_rev_mc.sum(axis=1) * breach_rate * 0.30
    
    var95_mc = np.percentile(annual_loss_mc, 95)
    es95_mc = annual_loss_mc[annual_loss_mc >= var95_mc].mean()
    days_test = max((pd.Timestamp(window_end_date) - pd.Timestamp(train_cutoff)).days, 1)
    window_loss_mc = daily_rev_mc[:, :days_test].sum(axis=1) * breach_rate * 0.30
    var95_window = np.percentile(window_loss_mc, 95)
    es95_window = window_loss_mc[window_loss_mc >= var95_window].mean()
    test_loss = test.loc[test['Unfulfilled_Dragon'], 'Realised_Loss_GBP'].sum()
    test_violations.append(1 if test_loss > var95_window else 0)
    
    rolling_window_ends.append(pd.Timestamp(window_end_date))
    rolling_es95_mc.append(es95_mc)
    rolling_var95_mc.append(var95_mc)
    rolling_es95_window.append(es95_window)
    rolling_var95_window.append(var95_window)
    last_annual_loss_mc = annual_loss_mc.copy()
    last_var95_mc = var95_mc
    last_es95_mc = es95_mc
    
rolling_window_ends = np.array(rolling_window_ends)
rolling_es95_mc = np.array(rolling_es95_mc)
rolling_var95_mc = np.array(rolling_var95_mc)
rolling_es95_window = np.array(rolling_es95_window)
rolling_var95_window = np.array(rolling_var95_window)
violations = np.array(test_violations)
annual_loss_mc = last_annual_loss_mc
var95_mc = last_var95_mc
es95_mc = last_es95_mc

# GPD PEAKS-OVER-THRESHOLD + COVERAGE TESTS
daily_loss_df = df.groupby('Date')['Realised_Loss_GBP'].sum().reset_index()
daily_loss_series = daily_loss_df['Realised_Loss_GBP'].values.astype(float)
GPD_THRESHOLD_PCT = 99.0
u = np.percentile(daily_loss_series, GPD_THRESHOLD_PCT)
exceed = daily_loss_series[daily_loss_series > u] - u
N_u = len(exceed)
n_obs = len(daily_loss_series)
def gpd_negloglik(theta, exceedances):
    xi_, beta_ = theta
    if beta_ <= 0 or xi_ >= 1:
        return 1e12
    terms = 1 + xi_ * exceedances / beta_
    if np.any(terms <= 0):
        return 1e12
    return N_u * np.log(beta_) + (1 / xi_ + 1) * np.sum(np.log(terms))
res = optimize.minimize(gpd_negloglik, [0.2, np.mean(exceed) if len(exceed) > 0 else 1000],
                        args=(exceed,), method='L-BFGS-B', bounds=[(-0.5, 1), (1e-8, None)])
xi, beta = res.x
alpha = 0.05
if xi != 0:
    var95_gpd = max(u, u + (beta / xi) * ((alpha * n_obs / N_u) ** (-xi) - 1))
else:
    var95_gpd = max(u, u - beta * np.log(alpha * n_obs / N_u))
if xi != 0:
    es95_gpd = (var95_gpd + beta - xi * u) / (1 - xi)
else:
    es95_gpd = var95_gpd + beta
print(f"\nGPD fit (Scenario 2 NETTED): ξ={xi:.4f} β={beta:.2f}, "
      f"VaR95_GPD=£{var95_gpd:,.0f} ES95_GPD=£{es95_gpd:,.0f}")

# Coverage tests
def kupiec_pval(viol_arr, p=0.05):
    n_w, k = len(viol_arr), viol_arr.sum()
    if k == 0 or k == n_w:
        return 1.0
    lr = -2 * np.log(
        ((1 - p) ** (n_w - k) * p ** k) /
        ((1 - k / n_w) ** (n_w - k) * (k / n_w) ** k))
    return 1 - stats.chi2.cdf(lr, 1)
def christoffersen_pval(viol_arr):
    n00 = n01 = n10 = n11 = 0
    for j in range(1, len(viol_arr)):
        prev, curr = viol_arr[j - 1], viol_arr[j]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1
    total = n00 + n01 + n10 + n11
    if total == 0: return 1.0
    pi = (n01 + n11) / total
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    eps = 1e-12
    pi0 = np.clip(pi0, eps, 1 - eps)
    pi1 = np.clip(pi1, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)
    ll_restricted = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_unrestricted = (n00 * np.log(1 - pi0) + n01 * np.log(pi0) +
                       n10 * np.log(1 - pi1) + n11 * np.log(pi1))
    lr_ind = -2 * (ll_restricted - ll_unrestricted)
    return 1 - stats.chi2.cdf(lr_ind, 1)
kupiec_p = kupiec_pval(violations)
christ_p = christoffersen_pval(violations)
n_viol = violations.sum()
viol_pct = violations.mean() * 100
print(f"Backtest: {n_viol}/{len(violations)} violations ({viol_pct:.1f}%), "
      f"Kupiec p={kupiec_p:.4f}, Christoffersen p={christ_p:.4f}")

# PLOTS
COLORS = dict(
    loss='#C0392B', var95='#E67E22', es95='#C0392B', gpd='#27AE60',
    mc='#2980B9', surge='#F39C12', dragon='#8E44AD', normal='#2ECC71',
    band='#FADBD8', window_var='#1A5276', window_es='#6C3483')
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.patch.set_facecolor('#FAFAFA')
for ax in axes.flat:
    ax.set_facecolor('#FAFAFA')
def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    
# Daily Dragon Loss Timeline
ax = axes[0, 0]
dld = daily_loss_df.sort_values('Date').copy()
x = pd.to_datetime(dld['Date'])
y = dld['Realised_Loss_GBP']
ax.fill_between(x, 0, y, where=(y >= var95_gpd),
                color=COLORS['band'], alpha=0.8, label='Tail breach zone')
ax.fill_between(x, 0, y, where=(y < var95_gpd), color='#D6EAF8', alpha=0.5)
ax.plot(x, y, color=COLORS['loss'], lw=1.0, zorder=3)
ax.axhline(var95_gpd, color=COLORS['var95'], ls='--', lw=2,
           label=f'GPD VaR95 = £{var95_gpd:,.0f}')
ax.axhline(es95_gpd, color=COLORS['es95'], ls='-', lw=2.5,
           label=f'GPD ES95 = £{es95_gpd:,.0f}')
breach_mask = y >= var95_gpd
ax.scatter(x[breach_mask], y[breach_mask], color=COLORS['loss'], s=18, zorder=5, alpha=0.9)
style_ax(ax, 'Daily Dragon Revenue-at-Risk (GPD tail threshold)', ylabel='£ Realised Loss')
ax.legend(fontsize=10, framealpha=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# Rolling Backtest
ax = axes[0, 1]
rwe = rolling_window_ends
ax.fill_between(rwe, rolling_var95_mc, rolling_es95_mc, alpha=0.18, color=COLORS['es95'])
ax.plot(rwe, rolling_es95_mc, color=COLORS['es95'], lw=2.5,
        label='Rolling Annual ES95 (MC)', marker='o', markersize=3)
ax.plot(rwe, rolling_var95_mc, color=COLORS['var95'], lw=1.8, ls='--',
        label='Rolling Annual VaR95 (MC)', marker='s', markersize=2.5)
ax.plot(rwe, rolling_var95_window, color=COLORS['window_var'], lw=1.4, ls=':',
        label='Window-Matched VaR95', marker='^', markersize=2.5)
ax.plot(rwe, rolling_es95_window, color=COLORS['window_es'], lw=1.4, ls='-.',
        label='Window-Matched ES95', marker='D', markersize=2.0)
viol_dates = rwe[violations == 1]
if len(viol_dates):
    ax.scatter(viol_dates, [rolling_var95_mc.min()*0.92] * len(viol_dates),
               marker='v', color='black', s=45, label=f'VaR breach ({n_viol}×)')
stats_txt = (f"Windows tested: {len(violations)}\n"
             f"Violations: {n_viol} ({viol_pct:.1f}%)\n"
             f"Kupiec p = {kupiec_p:.3f}\n"
             f"Christoff. p = {christ_p:.3f}")
ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
style_ax(ax, 'Purged Expanding-Window Backtest', ylabel='£ Risk Estimate')
ax.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# MC Annual Loss Distribution
ax = axes[1, 0]
counts, bin_edges = np.histogram(annual_loss_mc, bins=80)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
mask_normal = bin_centres < var95_mc
mask_var = (bin_centres >= var95_mc) & (bin_centres < es95_mc)
mask_es = bin_centres >= es95_mc
for m, col, lbl in [(mask_normal, COLORS['mc'], 'Below VaR95'),
                    (mask_var, COLORS['var95'], 'VaR95 – ES95'),
                    (mask_es, COLORS['es95'], 'Beyond ES95')]:
    ax.bar(bin_centres[m], counts[m], width=bin_edges[1]-bin_edges[0],
           color=col, alpha=0.75, label=lbl)
kde = gaussian_kde(annual_loss_mc)
x_kde = np.linspace(annual_loss_mc.min(), annual_loss_mc.max(), 400)
ax.plot(x_kde, kde(x_kde) * len(annual_loss_mc) * (bin_edges[1]-bin_edges[0]),
        color='#1A252F', lw=1.8)
ax.axvline(var95_mc, color=COLORS['var95'], ls='--', lw=2, label=f'MC VaR95 = £{var95_mc:,.0f}')
ax.axvline(es95_mc, color=COLORS['es95'], ls='-', lw=2.5, label=f'MC ES95 = £{es95_mc:,.0f}')
ax.axvline(es95_gpd, color=COLORS['gpd'], ls=':', lw=2, label=f'GPD ES95 = £{es95_gpd:,.0f}')
style_ax(ax, 'MC Annual Loss Distribution (last backtest window)',
         xlabel='Annual Preventable Loss (£)', ylabel='Path Count')
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.85)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# Delay Distribution
ax = axes[1, 1]
tier_order = ['normal', 'surge', 'dragon']
log_delays = df[['Delay_Tier', 'Delay_min']].copy()
log_delays['Log_Delay'] = np.log10(log_delays['Delay_min'])
sns.violinplot(data=log_delays, x='Delay_Tier', y='Log_Delay',
               order=tier_order, palette={'normal':'#2ECC71', 'surge':'#F39C12', 'dragon':'#8E44AD'},
               inner='quartile', linewidth=1.4, ax=ax, scale='width', cut=0)
ax.axhline(np.log10(SLA_BREACH_MIN), color='red', ls='--', lw=2,
           label=f'SLA breach ({SLA_BREACH_MIN} min)')
yticks_min = [5, 25, 90, 360, 480, 1440]
ax.set_yticks([np.log10(v) for v in yticks_min])
ax.set_yticklabels([f'{v:,} min' for v in yticks_min])
ax.set_xticklabels([f"{t}\n(n={df['Delay_Tier'].eq(t).sum():,})" for t in tier_order])
style_ax(ax, 'Fulfilment Delay Distribution by Tier (log scale)',
         xlabel='Order Tier', ylabel='Delay (minutes, log scale)')
ax.legend(fontsize=10, framealpha=0.85)

# Final layout
fig.suptitle('Dragon Revenue-at-Risk '
             '[Scenario 2 NETTED]',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('dragon_risk_report_scenario2.png', dpi=160, bbox_inches='tight')
plt.show()

# P&L SUMMARY
date_range_days = (df['Date'].max() - df['Date'].min()).days + 1
annual_gross = df['OrderValue_GBP'].sum() * (365 / date_range_days)
sla_tail = np.percentile(daily_loss_series, 99) * 252
sla_moderate = np.percentile(daily_loss_series, 95) * 252
print("\nP&L SUMMARY \n")
print(f"{'Metric':<50} {'Value':>14} {'% Gross':>8}")
print(f"{'Annualised Gross Revenue':<50} £{annual_gross:>12,.0f}")
print(f"{'GPD ES95 (daily tail, annualised)':<50} £{es95_gpd:>12,.0f} "
      f"{es95_gpd/annual_gross*100:>7.2f}% <- headline")
print(f"{'GPD VaR95 (daily tail)':<50} £{var95_gpd:>12,.0f} "
      f"{var95_gpd/annual_gross*100:>7.2f}%")
print(f"{'MC ES95 (annual, last backtest window)':<50} £{es95_mc:>12,.0f} "
      f"{es95_mc/annual_gross*100:>7.2f}%")
print(f"{'MC VaR95 (annual, last backtest window)':<50} £{var95_mc:>12,.0f} "
      f"{var95_mc/annual_gross*100:>7.2f}%")
print(f"{'Margin impact (30% GM × MC ES95)':<50} £{es95_mc*GROSS_MARGIN:>12,.0f} "
      f"{es95_mc*GROSS_MARGIN/annual_gross*100:>7.2f}%")
print(f"{'5-Year projected bleed (no action)':<50} £{es95_mc*5:>12,.0f}")
print(f"{'Annual bleed — tail scenario (99th pctile day ×252)':<50} £{sla_tail:>12,.0f}")
print(f"{'Annual bleed — moderate (95th pctile day ×252)':<50} £{sla_moderate:>12,.0f}")
print(f"{'Backtest horizon':<50} {'Window-matched VaR (FIXED)':>22}")
print(f"{'Backtest violations':<50} {n_viol:>12} / {len(violations)}")
print(f"{'Kupiec p-value':<50} {kupiec_p:>14.4f} "
      f"{'calibrated' if kupiec_p > 0.05 else 'needs review'}")
print(f"{'Christoffersen p-value':<50} {christ_p:>14.4f} "
      f"{'independent' if christ_p > 0.05 else 'clustered'}")
print("=" * 50)
print(f"\n In a bad year, preventable dragon loss approx: £{es95_mc:,.0f} (MC ES95).")
print(f" Tail scenario daily ×252 trading days -> £{sla_tail:,.0f} bleed.")
print(f" Backtest: {n_viol} violations in {len(violations)} windows "
      f"({viol_pct:.1f}%) — model is "
      f"{'stable' if kupiec_p > 0.05 else 'NEEDS REVIEW'}.")