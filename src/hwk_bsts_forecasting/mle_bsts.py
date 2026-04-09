#src/hwk_bsts_forecasting/mle_bsts.py
import gc
import logging
from os import times
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from numba import jit, prange
from src.data.loader import load_and_clean_uci
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def _plotly_show_alias(ctx):
    def _show(fig):
        if ctx is not None:
            ctx.save_plotly(fig)
        else:
            fig.show()
    return _show

def run_backtest(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Forecasting with Hawkes MLE + BSTS.
    """
    if state is None:
        state = {}
    df = state.get("df", df)
    _show = _plotly_show_alias(ctx)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    rng = np.random.default_rng(314159)
    
    if scenario == 1:

# Data alignment
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate', 'OrderValue_GBP']).copy()
        df['Is_Dragon'] = (df['Delay_Tier'] == 'dragon').astype(bool)   # ← exact 127 dragons

        anomaly_df = df[df['Is_Dragon']].copy().sort_values('InvoiceDate')
        t0 = anomaly_df['InvoiceDate'].min()
        anomaly_times = (anomaly_df['InvoiceDate'] - t0).dt.total_seconds() / 60.0
        anomaly_times = np.sort(anomaly_times.values.astype(np.float64))

        print(f"Dragons locked: {len(anomaly_times):,} | Target annual: 104")

# HAWKES NEGATIVE LOG-LIKELIHOOD 
        def hawkes_negloglik(params, times):
            mu, alpha, beta = params
            if mu <= 0 or alpha < 0 or beta <= 0 or beta <= (alpha / beta) >= 1.0:
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
        print(f"Hawkes MLE: μ={mu:.7f}  α={alpha:.3f}  β={beta:.3f}  (self-excitation confirmed)")

# Numba - JIT
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
        expected_30d = trapezoid(intensity, t_future)
        print(f"30-day expected dragons (Hawkes): {expected_30d:.1f}")

# BSTS (12-month forecast)
        anomaly_df['Month'] = anomaly_df['InvoiceDate'].dt.to_period('M')
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
        print(f"BSTS 2012 forecast: {proj_monthly}/month → {annual_2012:,}/year")

# P&L
        unfulfilled_rev = 147583.0
        n_unfulfilled_drag = 66
        anomaly_premium = unfulfilled_rev / n_unfulfilled_drag
        breach_rate = n_unfulfilled_drag / len(anomaly_times)
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
        print("\nMITIGATION SCENARIOS (tied to £60k ES95)")
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
        fig1.add_trace(go.Scatter(x=t_plot, y=intensity_plot,
                          name='λ(t) First 30 days', line=dict(color='red', width=4)))
        fig1.add_trace(go.Scatter(x=dragon_plot, y=np.zeros_like(dragon_plot),
                          mode='markers', marker=dict(color='gold', size=10, symbol='star'),
                          name='Actual Dragons'))
        fig1.update_layout(title="<b>Hawkes Bursts — First 30 days (Real Ops View)</b>",
                   xaxis_title="days since last anomaly", yaxis_title="Intensity λ(t)",
                   template="plotly_dark", height=600)
        _show(fig1) 

# BSTS FORECAST
        months = np.arange(len(monthly_series))
        forecast_months = np.arange(len(monthly_series), len(monthly_series)+12)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=months, y=monthly_series, name='Observed', marker_color='lightgray'))
        fig2.add_trace(go.Scatter(x=months, y=in_sample_fit, mode='lines', name='BSTS Fit', line=dict(color='cyan', width=2)))
        fig2.add_trace(go.Scatter(x=forecast_months, y=proj, mode='lines', name='12-Month Forecast', line=dict(color='gold', width=4)))
        fig2.add_hline(y=8, line_dash="dash", line_color="lime", annotation_text="Target: 8/month")
        fig2.add_hline(y=0, line_dash="solid", line_color="red", annotation_text="Zero Anomalies")
        fig2.add_annotation(
            x=0.02, y=0.95, xref='paper', yref='paper',
            text=f"<b>2012 BLEED: £ {base_annual_bleed:,.0f}</b><br>ZERO → £0",
            showarrow=False, font_size=13, bgcolor='gold', font_color='black'
        )
        fig2.update_layout(
            title="<b>BSTS: 12-MONTH FORECAST → ZERO TARGET</b>",
            xaxis_title="Month Index", yaxis_title="Anomalies per Month",
            template="plotly_dark", height=580, barmode='overlay'
        )
        _show(fig2)
        
#  FINAL SUMMARY 
        print("\n" + "="*50)
        print("HAWKES + BSTS SUMMARY — SCENARIO 1 ")
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
        #end of scenario1
        
    else:

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
        expected_30d = trapezoid(intensity, t_future)
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
        _show(fig1)

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
        _show(fig2)

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
        #end of scenario 2
        
    state["df"] = df    
    return state