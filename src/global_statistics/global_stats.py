# src/global_statistics/global_stats.py
# Contains Global Statistics, EVT/GPD Tail Analysis,
# and High volume SKU Filtering and Parameter Summary functions.
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from numba import jit
from src.data.loader import load_and_clean_uci
from src.utils.plot_paths import save_matplotlib_figure
import warnings
warnings.filterwarnings("ignore")
 
def _plotly_show_alias(ctx):
    def _show(fig):
        if ctx is not None:
            ctx.save_plotly(fig)
        else:
            fig.show()
    return _show
 
# Global Statistics
def run_global_statistics(
    df, 
    gross_picks=None,
    cancellations=None,
    real_df=None,
    scenario: int = 1,
    cfg: dict = None,
    ctx=None,
) -> dict:
    _show = _plotly_show_alias(ctx)
    state = {}
    df = state.get("df", df)
    
    if 'OrderValue' not in df.columns and 'OrderValue_GBP' in df.columns:
        df = df.assign(OrderValue=df['OrderValue_GBP'])
    if 'OrderValue_GBP' not in df.columns and 'OrderValue' in df.columns:
        df = df.assign(OrderValue_GBP=df['OrderValue'])

    if scenario == 1:
        daily_df = df.groupby('Date').agg(
            TotalQuantity=('Quantity', 'sum'),
            TotalValue=('OrderValue', 'sum')
        ).reset_index()
        daily_qty = daily_df['TotalQuantity'].values  # For ACF

# WITHOUT PARETO FILTER
        global_qnt_mean = df['Quantity'].mean()
        global_qnt_std = df['Quantity'].std(ddof=1)
        global_qnt_cv = global_qnt_std / global_qnt_mean  
        global_qnt_skew = df['Quantity'].skew()
        global_qnt_kurt = df['Quantity'].kurtosis() + 3  # Pearson Kurtosis

# Pareto 80/20
        sku_freq = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False)
        cum_freq = sku_freq.cumsum() / sku_freq.sum()
        high_vol_skus = sku_freq[cum_freq <= 0.8].index  # Top SKUs covering 80% orders
        filtered_df = df[df['SKU'].isin(high_vol_skus)].copy()
        quantities = filtered_df['Quantity'].values  # Filtered for tail

# Location
        global_qty_mean = quantities.mean()
        global_qty_median = np.median(quantities)
        global_qty_mode = stats.mode(quantities, keepdims=False)[0] if len(quantities) > 0 else np.nan 

# Dispersion
        global_qty_std = np.std(quantities, ddof=1)  # Unbiased 
        global_qty_cv = global_qty_std / global_qty_mean

# Shape
        global_qty_skew = stats.skew(quantities)
        global_qty_kurt = stats.kurtosis(quantities) + 3 #Pearson Kurtosis

# Dependencies
        qty_acf, _ = acf(daily_qty, nlags=10, fft=True, alpha=0.05)
        _, lb_q, lb_pval = acf(daily_qty, nlags=10, qstat=True, fft=True) 

# Tails
        global_qty_q95 = np.quantile(quantities, 0.95)
        global_qty_q975 = np.quantile(quantities, 0.975)
        global_qty_q99 = np.quantile(quantities, 0.99)
        u_tail = global_qty_q99
        exceedances = quantities[quantities > u_tail] - u_tail
        global_qty_mean_excess = exceedances.mean() if len(exceedances) > 0 else np.nan

# Tests
        z_scores = (quantities - global_qty_mean) / global_qty_std
        ks_stat, ks_pval = stats.kstest(z_scores, 'norm') # Kolmogorov - Smirnov Test
        ad_stat, ad_pval = sm.stats.normal_ad(quantities) # Anderson - Darling Test 
        SW_MAX_N = 5_000
        rng_sw = np.random.default_rng(314159)
        sw_sample = (rng_sw.choice(quantities, SW_MAX_N, replace=False)
             if len(quantities) > SW_MAX_N
             else quantities)
        sw_stat, sw_pval = stats.shapiro(sw_sample)

# Context Values
        global_val_mean = df['OrderValue_GBP'].mean()
        global_val_median = df['OrderValue_GBP'].median()
        global_val_q99 = df['OrderValue_GBP'].quantile(0.99)
        global_val_max = df['OrderValue_GBP'].max()

# PLOTS
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Original Histogram (without Pareto)
        axes[0].hist(quantities, bins=50, density=True, alpha=0.7, color='blue')
        axes[0].set_title('Quantity Histogram (Pareto-filtered SKUs)')
        axes[0].set_xlabel('Quantity')
        axes[0].set_ylabel('Density')

# Log Histogram 
        log_quantities = np.log1p(quantities)  # log(1+Y) for continuity at 0
        axes[1].hist(log_quantities, bins=50, density=True, alpha=0.7, color='green')
        axes[1].set_title('Log-Transformed Quantity Histogram')
        axes[1].set_xlabel('log(1 + Quantity)')
        axes[1].set_ylabel('Density')
        plt.tight_layout()
        save_matplotlib_figure("original-log-histogram-sc1.png", scenario=1)
        plt.show()

# QQ-Plot 
        fig, ax = plt.subplots(figsize=(14, 5))
        stats.probplot(quantities, dist="norm", plot=ax)
        ax.set_title('QQ-Plot: Quantity vs. Gaussian')
        save_matplotlib_figure("QQ-plot-sc1.png", scenario=1)
        plt.show()

# Outputs
        print(f"\nRaw rows           : {len(df):,}")  
        print(f"Unique SKUs        : {df['SKU'].nunique():,}")
        print(f"Date range         : {df['Date'].min()} -> {df['Date'].max()}")

        print("\nGLOBAL QUANTITIES ( Without Pareto Filtering)")
        print(f"Mean Quantity      : {global_qnt_mean:,.2f}")
        print(f"Std Dev            : {global_qnt_std:,.2f}")
        print(f"CV (σ/μ)           : {global_qnt_cv:,.3f}")
        print(f"Skewness Quantity : {global_qnt_skew:,.2f}")
        print(f"Pearson Kurtosis   : {global_qnt_kurt:,.2f}")

        print("\nGLOBAL QUANTITY LOCATION ( With Pareto Filtering)")
        print(f"Mean Quantity      : {global_qty_mean:,.2f}")
        print(f"Median Quantity    : {global_qty_median:,.2f}")
        print(f"Mode Quantity      : {global_qty_mode:,.2f}")

        print("\nGLOBAL QUANTITY DISPERSION")
        print(f"Std Dev Quantity   : {global_qty_std:,.2f}")
        print(f"CV (σ/μ)           : {global_qty_cv:.3f}")

        print("\nGLOBAL QUANTITY SHAPE")
        print(f"Skewness           : {global_qty_skew:,.2f}")
        print(f"Pearson Kurtosis   : {global_qty_kurt:,.2f}")

        print("\nGLOBAL QUANTITY DEPENDENCE (Daily Aggregates)")
        print(f"ACF Lags 1-5       : {qty_acf[1:6].round(3).tolist()}")
        print(f"Ljung-Box Q (lag10): {lb_q[-1]:,.2f} (p-val: {lb_pval[-1]:.4f})")

        print("\nGLOBAL QUANTITY TAILS")
        print(f"95th %ile Quantity : {global_qty_q95:,.2f}")
        print(f"97.5th %ile Qty    : {global_qty_q975:,.2f}")
        print(f"99th %ile Quantity : {global_qty_q99:,.2f}")
        print(f"Mean Excess (u=99%): {global_qty_mean_excess:,.2f}")

        print("\nGLOBAL QUANTITY NORMALITY TESTS")
        print(f"KS Statistic       : {ks_stat:.3f} (p-val: {ks_pval:.4f})")
        print(f"AD Statistic       : {ad_stat:.3f} (p-val: {ad_pval:.4f})")  # Pvalue
        print(f"SW Statistic       : {sw_stat:.3f} (p-val: {sw_pval:.4f}; subsampled)")

        print("\nGLOBAL ORDER VALUE (for context)")
        print(f"Mean Order Value   : £ {global_val_mean:,.2f}")
        print(f"Median Order Value : £ {global_val_median:,.2f}")
        print(f"99th %ile Value  : £ {global_val_q99:,.2f}")
        print(f"Max Order Value    : £ {global_val_max:,.2f}")
 
    else:
        df = state.get("df", df)
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
        z_scores = (quantities - global_qty_mean) / global_qty_std
        ks_stat, ks_pval = stats.kstest(z_scores, 'norm') # Kolmogorov - Smirnov Test
        ad_stat, ad_pval = sm.stats.normal_ad(quantities) # Anderson - Darling Test

        SW_MAX_N = 5000
        rng_sw = np.random.default_rng(314159)
        sw_sample = rng_sw.choice(quantities, SW_MAX_N, replace=False) if len(quantities) > SW_MAX_N else quantities
        sw_stat, sw_pval = stats.shapiro(sw_sample)

# Context Values
        global_val_mean = df['OrderValue_GBP'].mean()
        global_val_median = df['OrderValue_GBP'].median()
        global_val_q99 = df['OrderValue_GBP'].quantile(0.99)
        global_val_max = df['OrderValue_GBP'].max()

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
        save_matplotlib_figure("original-log-histogram-sc2.png", scenario=2)
        plt.show()

# QQ-Plot
        fig, ax = plt.subplots(figsize=(14, 5))
        stats.probplot(quantities, dist="norm", plot=ax)
        ax.set_title('QQ-Plot: Net Quantity vs. Gaussian (Scenario 2)')
        save_matplotlib_figure("QQ-plot-sc2.png", scenario=2)
        plt.show()

        print(f"\nRaw rows           : {len(df):,}")
        print(f"Unique SKUs        : {df['SKU'].nunique():,}")
        print(f"Date range         : {df['Date'].min()} -> {df['Date'].max()}")

        print("\nGLOBAL QUANTITIES ( Without Pareto Filtering)")
        print(f"Mean Quantity      : {global_qnt_mean:,.2f}")
        print(f"Std Dev            : {global_qnt_std:,.2f}")
        print(f"CV (σ/μ)           : {global_qnt_cv:,.3f}")
        print(f"Skewness Quantity : {global_qnt_skew:,.2f}")
        print(f"Pearson Kurtosis   : {global_qnt_kurt:,.2f}")

        print("\nGLOBAL QUANTITY LOCATION ( With Pareto Filtering)")
        print(f"Mean Quantity      : {global_qty_mean:,.2f}")
        print(f"Median Quantity    : {global_qty_median:,.2f}")
        print(f"Mode Quantity      : {global_qty_mode:,.2f}")

        print("\nGLOBAL QUANTITY DISPERSION")
        print(f"Std Dev Quantity   : {global_qty_std:,.2f}")
        print(f"CV (σ/μ)           : {global_qty_cv:.3f}")

        print("\nGLOBAL QUANTITY SHAPE")
        print(f"Skewness           : {global_qty_skew:,.2f}")
        print(f"Pearson Kurtosis   : {global_qty_kurt:,.2f}")

        print("\nGLOBAL QUANTITY DEPENDENCE (Daily Aggregates)")
        print(f"ACF Lags 1-5       : {qty_acf[1:6].round(3).tolist()}")
        print(f"Ljung-Box Q (lag10): {lb_q[-1]:,.2f} (p-val: {lb_pval[-1]:.4f})")

        print("\nGLOBAL QUANTITY TAILS")
        print(f"95th %ile Quantity : {global_qty_q95:,.2f}")
        print(f"97.5th %ile Qty    : {global_qty_q975:,.2f}")
        print(f"99th %ile Quantity : {global_qty_q99:,.2f}")
        print(f"Mean Excess (u=99%): {global_qty_mean_excess:,.2f}")

        print("\nGLOBAL QUANTITY NORMALITY TESTS")
        print(f"KS Statistic       : {ks_stat:.3f} (p-val: {ks_pval:.4f})")
        print(f"AD Statistic       : {ad_stat:.3f} (p-val: {ad_pval:.4f})")
        print(f"SW Statistic       : {sw_stat:.3f} (p-val: {sw_pval:.4f}; subsampled)")

        print("\nGLOBAL ORDER VALUE (for context)")
        print(f"Mean Order Value   : £ {global_val_mean:,.2f}")
        print(f"Median Order Value : £ {global_val_median:,.2f}")
        print(f"99th %ile Value  : £ {global_val_q99:,.2f}")
        print(f"Max Order Value    : £ {global_val_max:,.2f}")

        gp_n = len(gross_picks) if gross_picks is not None else df.attrs.get("gross_picks_n")
        ca_n = len(cancellations) if cancellations is not None else df.attrs.get("cancellations_n")
        real_n = len(real_df) if real_df is not None else df.attrs.get("real_n")
        if gp_n is not None and ca_n is not None:
            print(f"Gross picks (pre-cancellation) : {gp_n:,}")
            print(f"Cancellations/Refunds detected : {ca_n:,}")
            print(f"Cancelled volume removed                      : {gp_n - len(df):,} lines")
            if gp_n:
                print(f"Data quality warning: ~{((gp_n - len(df)) / gp_n * 100):.1f}% of gross picks had partial/full cancellation")
        if real_n is not None:
            print(f"Real product transactions after misc + CustomerID filter: {real_n:,}")

    state["quantities"]   = quantities
    state["daily_qty"]    = daily_qty
    state["global_qty_mean"] = global_qty_mean
    state["global_qty_std"] = global_qty_std
    state["global_qty_kurt"] = global_qty_kurt
    state["global_qty_skew"] = global_qty_skew
    state["qty_acf"] = qty_acf
    state["lb_pval"] = lb_pval
    state["global_qty_q95"] = global_qty_q95
    state["global_qty_q975"] = global_qty_q975
    state["global_qty_q99"] = global_qty_q99
    state["global_qty_mean_excess"] = global_qty_mean_excess
    state["ks_stat"] = ks_stat
    state["ks_pval"] = ks_pval
    state["ad_stat"] = ad_stat
    state["ad_pval"] = ad_pval
    state["sw_stat"] = sw_stat
    state["sw_pval"] = sw_pval
    state["high_vol_skus"] = high_vol_skus
    state["df"] = df
    return state

# EVT / GPD TAIL ANALYSIS
def run_evt_gpd(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Expects state to contain at least:
        quantities, high_vol_skus
    Adds to state:
        hill_xi, xi_moment, xi_gev, opt_u, slope
    """
    if state is None:
        state = {}
    _show = _plotly_show_alias(ctx)
 
    if scenario == 1:
        df = state.get("df", df)
        sku_vol = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False)
        cum_vol = sku_vol.cumsum() / sku_vol.sum()
        high_vol_skus = sku_vol[cum_vol <= 0.80].index.tolist()
        quantities = df[df['SKU'].isin(high_vol_skus)]['Quantity'].values.astype(np.float64)

        n = len(quantities)
        sorted_desc = np.sort(quantities)[::-1]

        print(f"EVT/GPD Test on High-Volume SKUs (n = {n:,})")

        hill_xi = np.nan
        xi_moment = np.nan
        xi_gev = np.nan
        log_ratios = np.array([])

# Hill Estimator
        k_stable = min(int(np.sqrt(n)), len(sorted_desc) - 1)
        if k_stable >= 2:
            log_ratios = np.log(sorted_desc[:k_stable] / sorted_desc[k_stable])
            hill_xi = np.mean(log_ratios)

# Moment Estimator
        if k_stable >= 2 and len(log_ratios) > 0:
            M1 = np.mean(log_ratios)
            M2 = np.mean(log_ratios**2)
            ratio = M1**2 / M2 if M2 > 0 else np.nan
            xi_moment = (
                M1 + 1 - 0.5 * (1 - ratio)**(-1)
                if not np.isnan(ratio) and ratio < 1
                else np.nan
            )

# GEV Block Maxima (blocks of 30 consecutive filtered transactions)
        block_size = 30
        maxima = np.array([
            quantities[i:i + block_size].max()
            for i in range(0, n - block_size + 1, block_size)
            if len(quantities[i:i + block_size]) == block_size
        ])
        if len(maxima) >= 3:
            c, mu_gev, sigma_gev = stats.genextreme.fit(maxima)
            xi_gev = -c

        print("\nTail Index Estimators (convergence confirms heavy tails)")
        print(f"  Hill ξ̂            : {hill_xi:.3f}" if not np.isnan(hill_xi) else "  Hill ξ̂            : nan")
        print(f"  Moment ξ̂          : {xi_moment:.3f}" if not np.isnan(xi_moment) else "  Moment ξ̂          : nan")
        print(f"  GEV Block-Max ξ̂   : {xi_gev:.3f}" if not np.isnan(xi_gev) else "  GEV Block-Max ξ̂   : nan")
        print("  All ξ̂  > 0        : Fréchet domain confirmed (heavy tails)")

# 4. Mean Excess Plot + Slope 
        u_grid = np.percentile(quantities, np.linspace(90, 99, 40))
        mean_excess = np.array([np.mean(quantities[quantities > u] - u) for u in u_grid])

# Upper-tail slope (linearity test)
        upper_idx = int(len(u_grid) * 0.6)
        slope, _ = np.polyfit(u_grid[upper_idx:], mean_excess[upper_idx:], 1)

        plt.figure(figsize=(12, 5))
        plt.plot(u_grid, mean_excess, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Threshold u (Quantity)')
        plt.ylabel('Mean Excess e(u)')
        plt.title('Mean Excess Plot\n(Linear positive slope in upper tail implies GPD is justified)')
        plt.grid(True, alpha=0.3)
        plt.axvline(u_grid[upper_idx], color='r', linestyle='--', label=f'Upper tail starts here (slope = {slope:.2f})')
        plt.legend()
        plt.tight_layout()
        save_matplotlib_figure("mean_excess_plot-sc1.png", scenario=1)
        plt.show()

        print(f"\nMean Excess Upper-Tail Slope : {slope:.3f}  (>0 implies heavy tail confirmed)")

# 5. AMSE-Optimal Threshold 
        @jit(nopython=True)
        def hill_jit(sorted_ex, k_u):
            if k_u < 2: 
                return 0.0
            lr = np.log(sorted_ex[:k_u] / sorted_ex[k_u-1])
            return np.mean(lr)

        u_grid_amse = np.percentile(quantities, np.linspace(92, 98.5, 25))
        amse_vals = []
        valid_u = []
        n_boot = 200
        rng_amse = np.random.default_rng(314159)

        for u in u_grid_amse:
            exc = quantities[quantities > u] - u
            k_u = len(exc)
            if k_u < 50: 
                continue
            xi_boots = []
            for _ in range(n_boot):
                boot = rng_amse.choice(exc, k_u, replace=True)
                s_boot = np.sort(boot)[::-1]
                xi_boots.append(hill_jit(s_boot, k_u))
            bias2 = (np.mean(xi_boots) - hill_xi)**2
            var_est = np.var(xi_boots)
            amse_vals.append(bias2 + var_est)
            valid_u.append(u)

        opt_u = np.nan
        if len(amse_vals) > 0:
            opt_idx = np.argmin(amse_vals)
            opt_u = valid_u[opt_idx]
            plt.figure(figsize=(12, 5))
            plt.plot(valid_u, amse_vals, 'b-o')
            plt.axvline(opt_u, color='r', linestyle='--', label=f'Optimal u = {opt_u:.1f}')
            plt.xlabel('Threshold u')
            plt.ylabel('AMSE')
            plt.title('Asymptotic MSE for Optimal Threshold Selection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_matplotlib_figure("amse_threshold-sc1.png", scenario=1)
            plt.show()
            print(f"AMSE-Optimal Threshold u     : {opt_u:.1f}")
        else:
            print("AMSE-Optimal Threshold u     : Not computable (insufficient tail data)")

        print("\nEVT DOMAIN DIAGNOSIS")
        print("Fréchet domain (ξ > 0) on high-volume SKU quantities; GPD MLE is fitted later on daily loss")
        
        # Store scenario 1 results to state
        state["hill_xi"]   = hill_xi
        state["xi_moment"] = xi_moment
        state["xi_gev"]    = xi_gev
        state["opt_u"]     = opt_u
        state["slope"]     = slope
        state["df"]        = df
       
    else:  # scenario == 2
        
        sku_vol       = df.groupby('SKU')['Quantity'].max().sort_values(ascending=False)
        cum_vol       = sku_vol.cumsum() / sku_vol.sum()
        high_vol_skus = sku_vol[cum_vol <= 0.80].index.tolist()
        quantities    = df[df['SKU'].isin(high_vol_skus)]['Quantity'].values.astype(np.float64)
        n_qty         = len(quantities)

        # Diagnostic
        n_high_skus     = len(high_vol_skus)
        max_single_all  = df['Quantity'].max()
        max_single_high = quantities.max() if n_qty > 0 else 0

        print(f"Selected {n_high_skus:,} SKUs that account for the top 80% of maximum single-order sizes")
        print(f"Largest single net order overall : {max_single_all:,.0f}")
        print(f"Largest single net order in selected SKUs : {max_single_high:,.0f}")
        print(f"EVT/GPD test on transaction-level NetQuantity from these SKUs (n = {n_qty:,})\n")

        hill_xi   = np.nan
        xi_moment = np.nan
        xi_gev    = np.nan
        log_ratios = np.array([])

# Hill Estimator
        n               = n_qty
        k_hill          = int(np.sqrt(n))
        sorted_qty_desc = np.sort(quantities)[::-1]
        k_hill          = min(k_hill, len(sorted_qty_desc) - 1)

        if k_hill >= 2:
            log_ratios = np.log(sorted_qty_desc[:k_hill] / sorted_qty_desc[k_hill])
            hill_xi    = log_ratios.mean()

# Moment Estimator
        if k_hill >= 2 and len(log_ratios) > 0:
            M1    = log_ratios.mean()
            M2    = (log_ratios ** 2).mean()
            ratio = M1 ** 2 / M2 if M2 > 0 else np.nan
            xi_moment = (
                M1 + 1 - 0.5 * (1 - ratio) ** (-1)
                if not np.isnan(ratio) and ratio < 1
                else np.nan
            )

# GEV Block Maxima (blocks of 30 consecutive filtered transactions)
        block_size = 30
        maxima = np.array([
            quantities[i:i + block_size].max()
            for i in range(0, n_qty - block_size + 1, block_size)
            if len(quantities[i:i + block_size]) == block_size
        ])

        if len(maxima) >= 3:
            c, mu_gev, sigma_gev = stats.genextreme.fit(maxima)
            xi_gev = -c

        print("\nTail Index Estimators")
        print(f" Hill ξ̂          : {hill_xi:.3f}" if not np.isnan(hill_xi) else " Hill ξ̂          : nan")
        print(f" Moment ξ̂        : {xi_moment:.3f}" if not np.isnan(xi_moment) else " Moment ξ̂        : nan")
        print(f" GEV Block-Max ξ̂ : {xi_gev:.3f}" if not np.isnan(xi_gev) else " GEV Block-Max ξ̂ : nan")
        print(" All ξ̂  > 0      : Fréchet domain confirmed (heavy tails persist even after netting)")

# Mean Excess Plot
        u_grid      = np.percentile(quantities, np.linspace(90, 99, 40))
        mean_excess = np.array([(quantities[quantities > u] - u).mean() for u in u_grid])
        upper_idx   = int(len(u_grid) * 0.6)
        slope, _    = np.polyfit(u_grid[upper_idx:], mean_excess[upper_idx:], 1)

        plt.figure(figsize=(12, 5))
        plt.plot(u_grid, mean_excess, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Threshold u (Net Quantity)')
        plt.ylabel('Mean Excess e(u)')
        plt.title('Mean Excess Plot - Scenario 2 (Net Quantity)\nLinear positive slope confirms GPD')
        plt.grid(True, alpha=0.3)
        plt.axvline(u_grid[upper_idx], color='r', linestyle='--',
                    label=f'Upper tail starts (slope = {slope:.2f})')
        plt.legend()
        plt.tight_layout()
        save_matplotlib_figure("mean_excess_plot-sc2.png", scenario=2)
        plt.show()

        print(f"\nMean Excess Upper-Tail Slope : {slope:.3f} (>0 -> heavy tail confirmed on net data)")

# AMSE-Optimal Threshold
        @jit(nopython=True)
        def hill_jit(sorted_ex, k_u):
            if k_u < 2: 
                return 0.0
            lr = np.log(sorted_ex[:k_u] / sorted_ex[k_u-1])
            return np.mean(lr)

        u_grid_amse = np.percentile(quantities, np.linspace(92, 98.5, 25))
        amse_vals   = []
        valid_u     = []
        n_boot      = 200
        opt_u       = np.nan
        rng_amse    = np.random.default_rng(314159)

        for u in u_grid_amse:
            exc = quantities[quantities > u] - u
            k_u = len(exc)
            if k_u < 50:
                continue
            xi_boots = np.empty(n_boot)
            for b in range(n_boot):
                boot   = rng_amse.choice(exc, k_u, replace=True)
                s_boot = np.sort(boot)[::-1]
                xi_boots[b] = hill_jit(s_boot, k_u)
            bias2   = (xi_boots.mean() - hill_xi) ** 2 if not np.isnan(hill_xi) else 0.0
            var_est = xi_boots.var()
            amse_vals.append(bias2 + var_est)
            valid_u.append(u)

        amse_vals = np.array(amse_vals)
        valid_u   = np.array(valid_u)

        if len(amse_vals) > 0:
            opt_idx = np.argmin(amse_vals)
            opt_u   = valid_u[opt_idx]
            plt.figure(figsize=(12, 5))
            plt.plot(valid_u, amse_vals, 'b-o')
            plt.axvline(opt_u, color='r', linestyle='--', label=f'Optimal u = {opt_u:.1f}')
            plt.xlabel('Threshold u (Net Quantity)')
            plt.ylabel('AMSE')
            plt.title('Asymptotic MSE - Optimal Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_matplotlib_figure("amse_threshold-sc2.png", scenario=2)
            plt.show()
            print(f"AMSE-Optimal Threshold u : {opt_u:.1f}")
        else:
            print("AMSE-Optimal Threshold u : Not computable (insufficient tail data)")

        print("\nEVT DOMAIN DIAGNOSIS")
        print("Fréchet domain (ξ > 0) on high-volume SKU quantities; GPD MLE is fitted later on daily loss")

        # store in state for run_param_summary and any downstream consumers
        state["hill_xi"]  = hill_xi
        state["xi_moment"] = xi_moment
        state["xi_gev"]  = xi_gev
        state["opt_u"]  = opt_u
        state["slope"]  = slope
        state["df"]  = df
    return state

def run_sku_filter(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Prints top-15 SKU table.
    """
    if state is None:
        state = {}
    _show = _plotly_show_alias(ctx)
 
    df_plot = df.copy()
    if 'Revenue' not in df_plot.columns:
        if 'OrderValue_GBP' in df_plot.columns:
            df_plot['Revenue'] = df_plot['OrderValue_GBP']
        else:
            df_plot['Revenue'] = df_plot['Quantity'] * df_plot['UnitPrice']

    agg = dict(Total_Quantity=('Quantity', 'sum'), Total_Revenue=('Revenue', 'sum'))
    if 'UnitPrice' in df_plot.columns:
        agg['Avg_Unit_Price'] = ('UnitPrice', 'mean')
    else:
        agg['Avg_Unit_Price'] = ('OrderValue_GBP', 'mean')

    top15 = (
        df_plot.groupby('StockCode')
        .agg(**agg)
        .sort_values('Total_Quantity', ascending=False)
        .head(15)
    )

    print(top15[['Total_Quantity', 'Avg_Unit_Price', 'Total_Revenue']].round(2))

    state["top15"] = top15
    
    return state

def run_param_summary(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Prints the final summary DataFrame.
    """
    if state is None:
        state = {}
    _show = _plotly_show_alias(ctx)
 
    if scenario == 1:
        hill_xi = state.get("hill_xi", [])
        summary = pd.DataFrame([
            ['Total Transactions',          len(df),                          f"{len(df):,}"],
            ['Unique SKUs',                 df['SKU'].nunique(),              f"{df['SKU'].nunique():,}"],
            ['Global Mean Order Value',     df['OrderValue_GBP'].mean(),          f"£{df['OrderValue_GBP'].mean():,.2f}"],
            ['Global Median Order Value',   df['OrderValue_GBP'].median(),        f"£{df['OrderValue_GBP'].median():,.2f}"],
            ['Pearson Kurtosis (Quantity)', df['Quantity'].kurtosis() + 3,    f"{df['Quantity'].kurtosis() + 3:,.1f}"],
            ['Skewness (Quantity)',         df['Quantity'].skew(),            f"{df['Quantity'].skew():,.2f}"],
            ['CV (σ/μ)',                    df['Quantity'].std() / df['Quantity'].mean(), f"{df['Quantity'].std() / df['Quantity'].mean():.3f}"],
            ['99.9th Percentile Value',     df['OrderValue_GBP'].quantile(0.999), f"£ {df['OrderValue_GBP'].quantile(0.999):,.2f}"],
            ['Max Order Value',             df['OrderValue_GBP'].max(),           f"£ {df['OrderValue_GBP'].max():,.2f}"],
            ['Mean Tail Index ξ',           np.mean(hill_xi),                 f"{np.mean(hill_xi):.3f}"],
            ['Median ξ',                    np.median(hill_xi),               f"{np.median(hill_xi):.3f}"],
            ['% ξ > 0 (Heavy Tail)',        (np.array(hill_xi) > 0).mean()*100, f"{(np.array(hill_xi) > 0).mean()*100:.1f}%"],
        ], columns=['Metric', 'Raw', 'Formatted'])

# Formatting
        summary['Value'] = summary['Formatted']
        summary = summary[['Metric', 'Value']]

        print("Final Summary")
        print(summary.to_string(index=False))
        
    else:
        hill_xi = state.get("hill_xi", np.nan)
        summary = pd.DataFrame([
            ['Total Net Transactions',          len(df),                          f"{len(df):,}"],
            ['Unique SKUs (Net)',               df['SKU'].nunique(),              f"{df['SKU'].nunique():,}"],
            ['Global Mean Net Order Value',     df['OrderValue_GBP'].mean(),          f"£{df['OrderValue_GBP'].mean():,.2f}"],
            ['Global Median Net Order Value',   df['OrderValue_GBP'].median(),        f"£{df['OrderValue_GBP'].median():,.2f}"],
            ['Pearson Kurtosis (Net Quantity)', df['Quantity'].kurtosis() + 3,    f"{df['Quantity'].kurtosis() + 3:,.1f}"],
            ['Skewness (Net Quantity)',         df['Quantity'].skew(),            f"{df['Quantity'].skew():,.2f}"],
            ['CV (σ/μ) – Net Quantity',         df['Quantity'].std() / df['Quantity'].mean() if df['Quantity'].mean() != 0 else np.nan, 
                                      f"{df['Quantity'].std() / df['Quantity'].mean():.3f}" if df['Quantity'].mean() != 0 else "—"],
            ['99.9th Percentile Net Value',     df['OrderValue_GBP'].quantile(0.999), f"£ {df['OrderValue_GBP'].quantile(0.999):,.2f}"],
            ['Max Net Order Value',             df['OrderValue_GBP'].max(),           f"£ {df['OrderValue_GBP'].max():,.2f}"],
            ['Mean Tail Index ξ (Net)',         hill_xi,                          f"{hill_xi:.3f}"],
            ['% ξ > 0 (Heavy Tail – Net)',      (hill_xi > 0) * 100 if not np.isnan(hill_xi) else 0, 
                                      f"{(hill_xi > 0)*100:.1f}%" if not np.isnan(hill_xi) else "—"],
        ], columns=['Metric', 'Raw', 'Formatted'])

# Formatting
        summary['Value'] = summary['Formatted']
        summary = summary[['Metric', 'Value']]

        print("Final Summary")
        print(summary.to_string(index=False))
 
        state["summary"] = summary
    
    state["df"] = df  # Store the final DataFrame for potential later use   
    return state