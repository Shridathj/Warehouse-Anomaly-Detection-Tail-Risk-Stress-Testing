# src/risk/monte_carlo.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import gaussian_kde
from src.data.loader import load_and_clean_uci
import warnings
warnings.filterwarnings("ignore")

def _plotly_show_alias(ctx):
    def _show(fig):
        if ctx is not None:
            ctx.save_plotly(fig)
        else:
            fig.show()
    return _show

def run_monte_carlo(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    10 000-path Monte Carlo annual loss distribution.
    """
    if state is None:
        state = {}
    df = state.get("df", df)
    _show = _plotly_show_alias(ctx)

    if scenario == 1:   
        df['Unfulfilled_Dragon'] = (
            df['Is_Dragon'] & 
            (df['Delay_min'] >= cfg["SLA_BREACH_MIN"]) & 
            df['SLA_Status'].isin(['SLA Breach', 'Fulfillment Failure'])
        )

# Empirical Inputs
        sample_days = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days + 1   
        target_annual = cfg["TARGET_ANNUAL_DRAGONS"]

        n_dragons = df['Is_Dragon'].sum()
        n_unfulfilled = df['Unfulfilled_Dragon'].sum()
        total_dragon_rev = df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum()
        unfulfilled_dragon_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()

        breach_rate_rev = unfulfilled_dragon_rev / total_dragon_rev if total_dragon_rev > 0 else 0.0
        breach_rate_count = n_unfulfilled / n_dragons if n_dragons > 0 else 0.0
        avg_daily_dragon_rev = total_dragon_rev / sample_days

        GROSS_MARGIN = cfg["GROSS_MARGIN"]
        print(f"\nGross margin applied: {GROSS_MARGIN:.0%} (Retail benchmark)")

        # Dragon-revenue days only
        daily_dragon = df[df['Is_Dragon']].groupby(df['InvoiceDate'].dt.date)['OrderValue_GBP'].sum()
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
        print(f"Unfulfilled Dragons (≥4h)   : {n_unfulfilled:,} ({breach_rate_count:.1%})")
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

        print(f"\nHeadline (Scenario 1 Max-Risk): In a bad year (worst 5 %), preventable dragon loss ≈ £ {es95:,.0f}")

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
            'Monte Carlo Annual Loss Distribution — Scenario 1'
            f'~{target_annual} dragons/year (Hawkes+BSTS) | {N_PATHS:,} paths| '
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

        state["annual_loss"] = annual_loss
        state["median"] = median
        state["var95"] = var95
        state["var99"] = var99
        state["es95"] = es95
        state["ul95"] = ul95
        state["annual_el"] = annual_el
        state["gross_rev"] = gross_rev

    else:  
        df['Unfulfilled_Dragon'] = (
            df['Is_Dragon'] &
            (df['Delay_min'] >= cfg["SLA_BREACH_MIN"]) &
            df['SLA_Status'].isin(['SLA Breach', 'Fulfillment Failure'])
        )

# Empirical Inputs
        sample_days = (df['Date'].max() - df['Date'].min()).days + 1   
        target_annual = cfg["TARGET_ANNUAL_DRAGONS"]                                                

        n_dragons = df['Is_Dragon'].sum()
        n_unfulfilled = df['Unfulfilled_Dragon'].sum()
        total_dragon_rev = df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum()
        unfulfilled_dragon_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()

        breach_rate_rev = unfulfilled_dragon_rev / total_dragon_rev if total_dragon_rev > 0 else 0.0
        breach_rate_count = n_unfulfilled / n_dragons if n_dragons > 0 else 0.0
        avg_daily_dragon_rev = total_dragon_rev / sample_days

        GROSS_MARGIN = cfg["GROSS_MARGIN"]
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

        state["annual_loss"] = annual_loss
        state["median"] = median
        state["var95"] = var95
        state["var99"] = var99
        state["es95"] = es95
        state["ul95"] = ul95
        state["annual_el"] = annual_el
        state["gross_rev"] = gross_rev
    
    state["df"] = df    
    return state