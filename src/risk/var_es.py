# src/risk/var.py
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

def run_var(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Value-at-Risk: holding cost drag, unfulfilled-dragon risk aggregates.
    """
    if state is None:
        state = {}
    df = state.get("df", df)
    _show = _plotly_show_alias(ctx)

    if scenario == 1:
        import plotly.graph_objects as go
        ANNUAL_HOLDING_RATE = cfg["ANNUAL_HOLDING_RATE"]
        SLA_BREACH_STATUSES = cfg["SLA_LABELS"][2:]

        df['Days_Delayed']     = df['Delay_min'] / 1440.0
        df['Holding_Cost_GBP'] = df['OrderValue_GBP'] * df['Days_Delayed'] * ANNUAL_HOLDING_RATE
        df['Net_Revenue_GBP']  = df['OrderValue_GBP'] - df['Holding_Cost_GBP']

        df['Is_Dragon']      = df['Delay_Tier'] == 'dragon'
        dragon_val_thresh    = df['OrderValue_GBP'].quantile(0.999)
        df['Is_Value_Dragon'] = df['OrderValue_GBP'] >= dragon_val_thresh

        df['Unfulfilled_Dragon'] = (
            df['Is_Dragon']
            & (df['Delay_min'] >= cfg["SLA_BREACH_MIN"])
            & df['SLA_Status'].isin(SLA_BREACH_STATUSES)
        )

        total_gross   = df['OrderValue_GBP'].sum()
        total_holding = df['Holding_Cost_GBP'].sum()
        total_net     = df['Net_Revenue_GBP'].sum()

        breach_surge  = df.loc[df['Delay_Tier'] == 'surge',  'SLA_Status'].isin(SLA_BREACH_STATUSES).mean()
        breach_normal = df.loc[df['Delay_Tier'] == 'normal', 'SLA_Status'].isin(SLA_BREACH_STATUSES).mean()
        breach_uplift = (breach_surge / breach_normal - 1) if breach_normal > 0 else float('inf')

        n_dragons       = df['Is_Dragon'].sum()
        n_unfulfilled   = df['Unfulfilled_Dragon'].sum()
        unfulfilled_rev = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()
        max_dragon_loss = df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].max() if n_unfulfilled > 0 else 0.0
        unfulfill_pct   = n_unfulfilled / n_dragons if n_dragons > 0 else 0.0
        sample_days     = state.get("sample_days") or (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days + 1

        print("\n" + "="*50)
        print("Scenario 1 - Gross Max-Risk")
        print("="*50)
        print(f"{'Metric':<45} {'Value':>20} {'% of Gross':>2}")
        print("-"*50)
        print(f"{'Gross Revenue (firm demand)':<45} £ {total_gross:>2,.2f}")
        print(f"{'Holding Cost (25% APR on delays)':<45} £ {total_holding:>2,.2f} {total_holding/total_gross:>11.3%}")
        print(f"{'Net Revenue (after holding drag)':<45} £ {total_net:>2,.2f} {total_net/total_gross:>11.3%}")
        print("-"*50)
        print(f"{'SLA Breach Rate — Normal ':<45} {breach_normal:>2.2%}")
        print(f"{'SLA Breach Rate — Surge ':<45} {breach_surge:>2.2%} {breach_uplift:>+11.1%} vs normal")
        print("-"*50)
        print(f"{'Sample Period':<45} {sample_days:>2} days")
        print(f"{'Value Dragon Threshold (99.9th %ile)':<45} £ {dragon_val_thresh:>2,.2f}  (Pareto tail)")
        print(f"{'Extreme Delay Dragons (Delay_Tier)':<45} {n_dragons:>2,} ({n_dragons/len(df):.4%} of txns)")
        print(f"{'Unfulfilled Dragons (≥4h SLA breach)':<45} {n_unfulfilled:>2,} ({unfulfill_pct:.1%} of dragons)")
        print(f"{'Revenue at Risk (unfulfilled dragons)':<45} £ {unfulfilled_rev:>2,.2f}")
        print(f"{'Max Single Dragon Loss':<45} £ {max_dragon_loss:>2,.2f}")
        print("="*50)

        df['Realized_Value_GBP'] = df['OrderValue_GBP'].copy()
        df['Days_Delayed']       = df['Delay_min'] / 1440.0
        df['Holding_Cost_GBP']   = df['OrderValue_GBP'] * df['Days_Delayed'] * ANNUAL_HOLDING_RATE
        df['Net_Revenue_GBP']    = df['Realized_Value_GBP'] - df['Holding_Cost_GBP']
        df['Is_Dragon']          = df['Delay_Tier'] == 'dragon'

        unfulfilled_count = df['Unfulfilled_Dragon'].sum()
        total_dragons     = df['Is_Dragon'].sum()
        unfulfill_pct     = 100 * unfulfilled_count / total_dragons if total_dragons > 0 else 0

        print(f"\nSummary")
        print(f"Dragons at risk of unfulfillment : {unfulfilled_count:,} / {total_dragons:,} ({unfulfill_pct:.1f}%)")
        print(f"Expected revenue at risk from Dragons : £{df.loc[df['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum():,.0f}")

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
            title=f"<b>Delay Distribution of Dragons & Unfulfillment Rate | Scenario 1 Max-Risk</b><br>"
                f"{unfulfilled_count:,} / {total_dragons:,} Dragons unfulfilled ({unfulfill_pct:.1f}%) at ≥4h SLA",
        xaxis_title="Delay (minutes)",
        yaxis_title="Count of Dragon Transactions",
        barmode='overlay',
        height=680,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        fig.add_vline(
        x=240,
        line=dict(color="red", dash="dash", width=3),
        annotation_text="4h SLA Breach Threshold (revenue loss starts)",
        annotation_position="top right"
        )
        
        _show(fig)   

        state["df"]                   = df
        state["total_gross"]          = total_gross
        state["total_holding"]        = total_holding
        state["total_net"]            = total_net
        state["n_dragons"]            = n_dragons
        state["n_unfulfilled"]        = n_unfulfilled
        state["unfulfilled_rev"]      = unfulfilled_rev
        state["breach_rate_rev"]      = unfulfilled_rev / df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum()
        state["breach_rate_count"]    = unfulfill_pct / 100
        state["avg_daily_dragon_rev"] = df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum() / sample_days
        state["sample_days"]          = sample_days 

    else: 
        import plotly.graph_objects as go
        ANNUAL_HOLDING_RATE = 0.25          # 25% APR — WERC benchmark for working capital cost
        SLA_BREACH_STATUSES = ['SLA Breach', 'Fulfillment Failure']

        df['Days_Delayed']     = df['Delay_min'] / 1440.0
        df['Holding_Cost_GBP'] = df['OrderValue_GBP'] * df['Days_Delayed'] * ANNUAL_HOLDING_RATE
        df['Net_Revenue_GBP']  = df['OrderValue_GBP'] - df['Holding_Cost_GBP']
        df['Is_Dragon'] = df['Delay_Tier'] == 'dragon'

# Secondary: High-value tail 
        dragon_val_thresh = df['OrderValue_GBP'].quantile(0.999)
        df['Is_Value_Dragon'] = df['OrderValue_GBP'] >= dragon_val_thresh

# Unfulfilled Dragon = primary dragon that also breached SLA
        df['Unfulfilled_Dragon'] = (
            df['Is_Dragon'] & 
            (df['Delay_min'] >= cfg["SLA_BREACH_MIN"]) & 
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
        
        _show(fig)
        
        state["total_gross"]  = total_gross
        state["total_holding"]  = total_holding
        state["total_net"]  = total_net
        state["n_dragons"]  = n_dragons
        state["n_unfulfilled"]   = n_unfulfilled
        state["unfulfilled_rev"]  = unfulfilled_rev
        state["breach_rate_rev"]  = unfulfilled_rev / df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum()
        state["breach_rate_count"]  = unfulfill_pct / 100
        state["avg_daily_dragon_rev"] = df.loc[df['Is_Dragon'], 'OrderValue_GBP'].sum() / sample_days
        state["sample_days"]   = sample_days 
        
    state["df"] = df
    return state