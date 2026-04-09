# src/simulation/delays.py
import numpy as np
import pandas as pd
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


def run_mock_delays(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Assign delay tiers (normal / surge / dragon) and compute SLA status.
    """
    if state is None:
        state = {}
    _show = _plotly_show_alias(ctx)

    rng = np.random.default_rng(seed=314159)
    n   = len(df)
        
    if scenario == 1:
        SURGE_PCT  = cfg["SURGE_PCT"]
        DRAGON_PCT  = cfg["DRAGON_PCT"]
        DRAGON_BIAS_EXP = cfg["DRAGON_BIAS_EXP"]
        NORMAL_MEAN_MIN = cfg["NORMAL_MEAN_MIN"]
        SURGE_MEAN_MIN  = cfg["SURGE_MEAN_MIN"]
        DRAGON_MEAN_MIN = cfg["DRAGON_MEAN_MIN"]
        SIGMA_NORMAL = cfg["SIGMA_NORMAL"]
        SIGMA_SURGE  = cfg["SIGMA_SURGE"]
        SIGMA_DRAGON  = cfg["SIGMA_DRAGON"]
        NORMAL_CLIP  = cfg["NORMAL_CLIP"]
        SURGE_CLIP  = cfg["SURGE_CLIP"]
        DRAGON_CLIP  = cfg["DRAGON_CLIP"]
        SLA_BINS   = cfg["SLA_BINS"]
        SLA_LABELS  = cfg["SLA_LABELS"]

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
        sample_days = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days + 1
        total_rev = df['OrderValue_GBP'].sum()
        sla_counts = df['SLA_Status'].value_counts().reindex(SLA_LABELS)
        revenue_at_risk = df[df['SLA_Status'].isin(['SLA Breach','Fulfillment Failure'])]['OrderValue_GBP'].sum()

        print(f"\nDelays")
        print(f"{'='*50}")
        print(f"Surge events : {(df['Delay_Tier']=='surge').sum():,} / {n:,} ({(df['Delay_Tier']=='surge').mean():.1%})")
        print(f"Dragon events : {(df['Delay_Tier']=='dragon').sum():,} / {n:,} ({(df['Delay_Tier']=='dragon').mean():.4%})")
        print(f"Sample period : {sample_days} days ({df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()})")
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

        state["df"]          = df
        state["surge_idx"]   = surge_idx
        state["dragon_idx"]  = dragon_idx
        state["sample_days"] = sample_days

    else:
        SURGE_PCT        = cfg["SURGE_PCT"]
        DRAGON_PCT       = cfg["DRAGON_PCT"]
        DRAGON_BIAS_EXP  = cfg["DRAGON_BIAS_EXP"]
        NORMAL_MEAN_MIN  = cfg["NORMAL_MEAN_MIN"]
        SURGE_MEAN_MIN   = cfg["SURGE_MEAN_MIN"]
        DRAGON_MEAN_MIN  = cfg["DRAGON_MEAN_MIN"]
        SIGMA_NORMAL     = cfg["SIGMA_NORMAL"]
        SIGMA_SURGE      = cfg["SIGMA_SURGE"]
        SIGMA_DRAGON     = cfg["SIGMA_DRAGON"]
        NORMAL_CLIP      = cfg["NORMAL_CLIP"]
        SURGE_CLIP       = cfg["SURGE_CLIP"]
        DRAGON_CLIP      = cfg["DRAGON_CLIP"]
        SLA_BINS         = cfg["SLA_BINS"]
        SLA_LABELS       = cfg["SLA_LABELS"]

        weights = df['OrderValue_GBP'] ** DRAGON_BIAS_EXP
        weights = weights / weights.sum()

        all_idx    = np.arange(n)
        dragon_idx = rng.choice(all_idx, size=int(DRAGON_PCT * n), p=weights, replace=False)
        remaining  = np.setdiff1d(all_idx, dragon_idx)
        surge_idx  = rng.choice(remaining, size=int(SURGE_PCT * n), replace=False)

        tier = np.full(n, 'normal', dtype=object)
        tier[dragon_idx] = 'dragon'
        tier[surge_idx]  = 'surge'
        df['Delay_Tier'] = tier

        delays = np.zeros(n, dtype=float)
        for mask, mean_min, sigma, clip in [
            (df['Delay_Tier'] == 'normal', NORMAL_MEAN_MIN, SIGMA_NORMAL, NORMAL_CLIP),
            (df['Delay_Tier'] == 'surge',  SURGE_MEAN_MIN,  SIGMA_SURGE,  SURGE_CLIP),
            (df['Delay_Tier'] == 'dragon', DRAGON_MEAN_MIN, SIGMA_DRAGON, DRAGON_CLIP),
        ]:
            mu  = np.log(mean_min) - 0.5 * sigma**2
            raw = np.exp(mu + sigma * rng.standard_normal(mask.sum()))
            delays[mask] = np.clip(raw, clip[0], clip[1])

        df['Delay_min']  = delays
        df['SLA_Status'] = pd.cut(df['Delay_min'], bins=SLA_BINS, labels=SLA_LABELS, right=False)

        sample_days = (df['Date'].max() - df['Date'].min()).days + 1
        total_rev   = df['OrderValue_GBP'].sum()
        sla_counts  = df['SLA_Status'].value_counts().reindex(SLA_LABELS)
        revenue_at_risk = df[df['SLA_Status'].isin(['SLA Breach', 'Fulfillment Failure'])]['OrderValue_GBP'].sum()

        print(f"\nDelays - SCENARIO 2 (NETTED)")
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

        state["surge_idx"] = surge_idx
        state["dragon_idx"] = dragon_idx
        state["sample_days"] = sample_days
    
    state["df"] = df
    return state