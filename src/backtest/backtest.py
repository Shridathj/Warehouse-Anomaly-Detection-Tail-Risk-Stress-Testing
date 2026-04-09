# src/backtest/backtest.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.stats import gaussian_kde
from datetime import timedelta
import seaborn as sns
import warnings
from src.data.loader import load_and_clean_uci
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def _plotly_show_alias(ctx):
    def _show(fig):
        if ctx is not None:
            ctx.save_plotly(fig)
        else:
            fig.show()
    return _show

def run_quantitative_backtest(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Full quantitative risk report and P&L summary with backtesting analysis.
    Expects state to contain (at minimum):
        annual_loss, var95, es95, var95_mc, es95_mc,
        daily_loss_df, rolling_* arrays, violations, n_viol,
        xi, beta, u, N_u, GROSS_MARGIN, date_range_days
    """
    if state is None:
        state = {}
    df = state.get("df", df)
    _show = _plotly_show_alias(ctx)

    if scenario == 1:
        SEED = 314159
        N_MC_PATHS = 10_000
        SURGE_PCT = cfg["SURGE_PCT"]

        DRAGON_PCT = cfg["DRAGON_PCT"]
        DRAGON_BIAS_EXP = cfg["DRAGON_BIAS_EXP"]
        NORMAL_MEAN_MIN = cfg["NORMAL_MEAN_MIN"]
        SURGE_MEAN_MIN  = cfg["SURGE_MEAN_MIN"]
        DRAGON_MEAN_MIN = cfg["DRAGON_MEAN_MIN"]
        SIGMA_NORMAL  = cfg["SIGMA_NORMAL"]
        SIGMA_SURGE  = cfg["SIGMA_SURGE"]
        SIGMA_DRAGON = cfg["SIGMA_DRAGON"]

        SLA_BREACH_MIN  = cfg["SLA_BREACH_MIN"]
        GROSS_MARGIN  = cfg["GROSS_MARGIN"]
        ANNUAL_HOLDING_RATE = cfg["ANNUAL_HOLDING_RATE"]

        MIN_TRAIN_DAYS = cfg["MIN_TRAIN_DAYS"]
        WINDOW_STEP  = cfg["WINDOW_STEP"]
        PURGE_GAP  = cfg["PURGE_GAP"]

# LOAD DATA - Use loader.py which handles all data cleaning, filtering, and validation
        print("SCENARIO 1 BACKTEST")
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').dt.date
        if 'OrderValue_GBP' not in df.columns:
            df['OrderValue_GBP'] = df.get('OrderValue', df['Quantity'] * df['UnitPrice'])

        date_range_days = (df['Date'].max() - df['Date'].min()).days
        print(f"Loaded {len(df):,} positive transactions | "
      f"£{df['OrderValue_GBP'].sum():,.0f} gross revenue | "
      f"{df['Date'].min()} → {df['Date'].max()}")

# DELAY SIMULATION
        rng = np.random.default_rng(seed=SEED)
        n   = len(df)

        weights = df['OrderValue_GBP'] ** DRAGON_BIAS_EXP
        weights = weights / weights.sum()

        all_idx    = np.arange(n)
        dragon_idx = rng.choice(all_idx, size=int(DRAGON_PCT * n), p=weights, replace=False)
        remaining  = np.setdiff1d(all_idx, dragon_idx)
        surge_idx  = rng.choice(remaining, size=int(SURGE_PCT * n), replace=False)

        tier             = np.full(n, 'normal', dtype=object)
        tier[dragon_idx] = 'dragon'
        tier[surge_idx]  = 'surge'
        df['Delay_Tier'] = tier

        delays = np.zeros(n, dtype=float)
        for condition, mean_min, sigma, clip_range in [
            (df['Delay_Tier'] == 'normal', NORMAL_MEAN_MIN, SIGMA_NORMAL, (5,    120)),
            (df['Delay_Tier'] == 'surge',  SURGE_MEAN_MIN,  SIGMA_SURGE,  (30,   480)),
            (df['Delay_Tier'] == 'dragon', DRAGON_MEAN_MIN, SIGMA_DRAGON, (180, 1440)),
        ]:
            mu                = np.log(mean_min) - 0.5 * sigma ** 2
            raw_d             = np.exp(mu + sigma * rng.standard_normal(condition.sum()))
            delays[condition] = np.clip(raw_d, *clip_range)

        df['Delay_min']          = delays
        df['Days_Delayed']       = df['Delay_min'] / 1440.0
        df['Holding_Cost_GBP']   = df['OrderValue_GBP'] * df['Days_Delayed'] * ANNUAL_HOLDING_RATE
        df['Is_Dragon']          = df['Delay_Tier'] == 'dragon'
        df['Unfulfilled_Dragon'] = df['Is_Dragon'] & (df['Delay_min'] >= SLA_BREACH_MIN)

        dragon_breach_mask = df['Unfulfilled_Dragon']
        df['Realised_Loss_GBP'] = 0.0
        df.loc[dragon_breach_mask, 'Realised_Loss_GBP'] = (
            df.loc[dragon_breach_mask, 'OrderValue_GBP'] +
            df.loc[dragon_breach_mask, 'Holding_Cost_GBP'])

        print(f"Simulation done → {df['Is_Dragon'].sum():,} dragons "
            f"({df['Is_Dragon'].mean():.4%}) | "
            f"{df['Unfulfilled_Dragon'].sum():,} SLA breaches")

# PURGED EXPANDING-WINDOW BACKTEST
        dates     = sorted(df['Date'].unique())                           
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
            train_cutoff    = window_end_date - timedelta(days=PURGE_GAP)

            train = df[df['Date'] <= train_cutoff]
            test  = df[(df['Date'] > train_cutoff) & (df['Date'] <= window_end_date)]

            if len(train) < 50_000 or len(test) < 5:
                continue
        
            train_dragons          = train[train['Is_Dragon']]
            train_unfulfilled_rev  = train.loc[train['Unfulfilled_Dragon'], 'OrderValue_GBP'].sum()
            train_total_dragon_rev = train_dragons['OrderValue_GBP'].sum()
            breach_rate            = (train_unfulfilled_rev / train_total_dragon_rev
                              if train_total_dragon_rev > 0 else 0.0)

            daily_dragon = (
                train_dragons
                .groupby(train_dragons['InvoiceDate'].dt.date)['OrderValue_GBP']
                .sum())
            avg_daily = daily_dragon.mean() if len(daily_dragon) > 0 else 1e-6
            cv_train  = min(3.0, (daily_dragon.std() / avg_daily
                          if len(daily_dragon) > 1 else 2.0))

            mu_daily    = np.log(max(avg_daily, 1e-6)) - 0.5 * np.log(1 + cv_train ** 2)
            sigma_daily = np.sqrt(np.log(1 + cv_train ** 2))

    # Annual MC 
            daily_rev_mc   = np.exp(rng.normal(mu_daily, sigma_daily, size=(N_MC_PATHS, 365)))
            annual_loss_mc = daily_rev_mc.sum(axis=1) * breach_rate * GROSS_MARGIN
            var95_mc       = np.percentile(annual_loss_mc, 95)
            es95_mc        = annual_loss_mc[annual_loss_mc >= var95_mc].mean()

    # Window-matched MC 
            days_test      = max((pd.Timestamp(window_end_date)
                          - pd.Timestamp(train_cutoff)).days, 1)
            window_loss_mc = daily_rev_mc[:, :days_test].sum(axis=1) * breach_rate * GROSS_MARGIN
            var95_window   = np.percentile(window_loss_mc, 95)
            es95_window    = window_loss_mc[window_loss_mc >= var95_window].mean()

    # Violation test
            test_loss = test['Realised_Loss_GBP'].sum()
            test_violations.append(1 if test_loss > var95_window else 0)

            rolling_window_ends.append(pd.Timestamp(window_end_date))
            rolling_es95_mc.append(es95_mc)
            rolling_var95_mc.append(var95_mc)
            rolling_es95_window.append(es95_window)
            rolling_var95_window.append(var95_window)

    # Persist for plots
            last_annual_loss_mc = annual_loss_mc.copy()
            last_var95_mc       = var95_mc
            last_es95_mc        = es95_mc

            if i % 90 == 0:
                print(f"  Window end: {window_end_date} | "
                    f"Annual ES95: £{es95_mc:,.0f} | "
                    f"Window VaR95 ({days_test}d): £{var95_window:,.0f} | "
                    f"Realised: £{test_loss:,.0f}")

        rolling_window_ends  = np.array(rolling_window_ends)
        rolling_es95_mc      = np.array(rolling_es95_mc)
        rolling_var95_mc     = np.array(rolling_var95_mc)
        rolling_es95_window  = np.array(rolling_es95_window)
        rolling_var95_window = np.array(rolling_var95_window)
        violations           = np.array(test_violations)

        annual_loss_mc = last_annual_loss_mc
        var95_mc       = last_var95_mc
        es95_mc        = last_es95_mc

# GPD PEAKS-OVER-THRESHOLD + COVERAGE TESTS
        daily_loss_df     = df.groupby('Date')['Realised_Loss_GBP'].sum().reset_index()
        daily_loss_series = daily_loss_df['Realised_Loss_GBP'].values.astype(float)

        GPD_THRESHOLD_PCT = 99.0
        u      = np.percentile(daily_loss_series, GPD_THRESHOLD_PCT)
        exceed = daily_loss_series[daily_loss_series > u] - u
        N_u    = len(exceed)
        n_obs  = len(daily_loss_series)

        def gpd_negloglik(theta, exceedances):
            xi_, beta_ = theta
            if beta_ <= 0 or xi_ >= 1:
                return 1e12
            terms = 1 + xi_ * exceedances / beta_
            if np.any(terms <= 0):
                return 1e12
            return N_u * np.log(beta_) + (1 / xi_ + 1) * np.sum(np.log(terms))

        res      = optimize.minimize(gpd_negloglik, [0.2, np.mean(exceed)], args=(exceed,),
                             method='L-BFGS-B', bounds=[(-0.5, 1), (1e-8, None)])
        xi, beta = res.x
        alpha = 0.05

# GPD VaR + ES
        if xi != 0:
            var95_gpd = max(u, u + (beta / xi) * ((alpha * n_obs / N_u) ** (-xi) - 1))
        else:
            var95_gpd = max(u, u - beta * np.log(alpha * n_obs / N_u))

        if xi != 0:
            es95_gpd = (var95_gpd + beta - xi * u) / (1 - xi)
        else:
            es95_gpd = var95_gpd + beta

        print(f"\nGPD fit: ξ={xi:.4f}  β={beta:.2f}  |  "
            f"VaR95_GPD=£{var95_gpd:,.0f},  ES95_GPD=£{es95_gpd:,.0f}")

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
                if   prev == 0 and curr == 0: n00 += 1
                elif prev == 0 and curr == 1: n01 += 1
                elif prev == 1 and curr == 0: n10 += 1
                else: n11 += 1

            total = n00 + n01 + n10 + n11
            if total == 0:
                return 1.0

            pi  = (n01 + n11) / total
            pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
            pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

            eps = 1e-12
            pi0 = np.clip(pi0, eps, 1 - eps)
            pi1 = np.clip(pi1, eps, 1 - eps)
            pi  = np.clip(pi,  eps, 1 - eps)

            ll_restricted   = (n00 + n10) * np.log(1 - pi)  + (n01 + n11) * np.log(pi)
            ll_unrestricted = (n00 * np.log(1 - pi0) + n01 * np.log(pi0) +
                            n10 * np.log(1 - pi1) + n11 * np.log(pi1))
            lr_ind = -2 * (ll_restricted - ll_unrestricted)
            return 1 - stats.chi2.cdf(lr_ind, 1)

        kupiec_p = kupiec_pval(violations)
        christ_p = christoffersen_pval(violations)
        n_viol   = violations.sum()
        viol_pct = violations.mean() * 100

        print(f"Backtest: {n_viol}/{len(violations)} violations ({viol_pct:.1f}%) | "
            f"Kupiec p={kupiec_p:.4f} | Christoffersen p={christ_p:.4f}")

# PLOTS
        COLORS = dict(
            loss='#C0392B',
            var95='#E67E22',
            es95='#C0392B',
            gpd='#27AE60',
            mc='#2980B9',
            surge='#F39C12',
            dragon='#8E44AD',
            normal='#2ECC71',
            band='#FADBD8',
            window_var='#1A5276',
            window_es='#6C3483',)

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
        x   = pd.to_datetime(dld['Date'])
        y   = dld['Realised_Loss_GBP']

        ax.fill_between(x, 0, y, where=(y >= var95_gpd),
                color=COLORS['band'], alpha=0.8, label='Tail breach zone')
        ax.fill_between(x, 0, y, where=(y < var95_gpd),
                color='#D6EAF8', alpha=0.5)
        ax.plot(x, y, color=COLORS['loss'], lw=1.0, zorder=3)

        ax.axhline(var95_gpd, color=COLORS['var95'], ls='--', lw=2,
           label=f'GPD VaR95 = £{var95_gpd:,.0f}')
        ax.axhline(es95_gpd,  color=COLORS['es95'],  ls='-',  lw=2.5,
           label=f'GPD ES95  = £{es95_gpd:,.0f}')

        breach_mask = y >= var95_gpd
        ax.scatter(x[breach_mask], y[breach_mask],
           color=COLORS['loss'], s=18, zorder=5, alpha=0.9)

        style_ax(ax, 'Daily Dragon Revenue-at-Risk  (GPD tail threshold)',
         ylabel='£ Realised Loss')
        ax.legend(fontsize=10, framealpha=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# Rolling Backtest
        ax = axes[0, 1]
        rwe = rolling_window_ends

        ax.fill_between(rwe, rolling_var95_mc, rolling_es95_mc,
                alpha=0.18, color=COLORS['es95'], label='Annual VaR95–ES95 band')
        ax.plot(rwe, rolling_es95_mc, color=COLORS['es95'],  lw=2.5,
        label='Rolling Annual ES95 (MC)',
        marker='o', markersize=3, zorder=4)
        ax.plot(rwe, rolling_var95_mc, color=COLORS['var95'], lw=1.8, ls='--',
        label='Rolling Annual VaR95 (MC)',
        marker='s', markersize=2.5, zorder=4)
        ax.plot(rwe, rolling_var95_window, color=COLORS['window_var'], lw=1.4, ls=':',
        label='Window-Matched VaR95 (violation test)', marker='^', markersize=2.5, zorder=4)
        ax.plot(rwe, rolling_es95_window, color=COLORS['window_es'], lw=1.4, ls='-.',
        label='Window-Matched ES95', marker='D', markersize=2.0, zorder=4)
        ax.autoscale()
        y_min_data = min(rolling_var95_window.min(), rolling_var95_mc.min())
        viol_dates = rwe[violations == 1]
        if len(viol_dates):
            viol_y_pos = y_min_data * 0.92
            ax.scatter(viol_dates, [viol_y_pos] * len(viol_dates),
                    marker='v', color='black', s=45, zorder=6,
                    label=f'VaR breach ({n_viol}×)')
        stats_txt = (f"Windows tested: {len(violations)}\n"
             f"Violations: {n_viol} ({viol_pct:.1f}%)\n"
             f"Kupiec p = {kupiec_p:.3f}\n"
             f"Christoff. p = {christ_p:.3f}\n"
             f"[Violations use window-matched VaR]")
        ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.85, edgecolor='#ccc'))
        style_ax(ax, 'Purged Expanding-Window Backtest  (zero leakage, scale-corrected)',
                 ylabel='£ Risk Estimate')
        ax.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# MC Annual Loss Distribution
        ax = axes[1, 0]

        counts, bin_edges = np.histogram(annual_loss_mc, bins=80)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        mask_normal = bin_centres < var95_mc
        mask_var    = (bin_centres >= var95_mc) & (bin_centres < es95_mc)
        mask_es     = bin_centres >= es95_mc
        for m, col, lbl in [
            (mask_normal, COLORS['mc'],    'Below VaR95'),
            (mask_var,    COLORS['var95'], 'VaR95 – ES95'),
            (mask_es,     COLORS['es95'],  'Beyond ES95'),
        ]:
            ax.bar(bin_centres[m], counts[m],
                width=bin_edges[1] - bin_edges[0],
                color=col, alpha=0.75, label=lbl)

        kde   = gaussian_kde(annual_loss_mc)
        x_kde = np.linspace(annual_loss_mc.min(), annual_loss_mc.max(), 400)
        ax.plot(x_kde, kde(x_kde) * len(annual_loss_mc) * (bin_edges[1] - bin_edges[0]),
                color='#1A252F', lw=1.8)
        ax.axvline(var95_mc, color=COLORS['var95'], ls='--', lw=2,
                label=f'MC VaR95  = £{var95_mc:,.0f}')
        ax.axvline(es95_mc,  color=COLORS['es95'],  ls='-',  lw=2.5,
           label=f'MC ES95   = £{es95_mc:,.0f}')
        ax.axvline(es95_gpd, color=COLORS['gpd'],   ls=':',  lw=2,
           label=f'GPD ES95  = £{es95_gpd:,.0f}')
        tail_pct = (annual_loss_mc >= var95_mc).mean() * 100
        ax.text(0.97, 0.97, f'{tail_pct:.1f}% of paths\nexceed VaR95',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        style_ax(ax, 'MC Annual Loss Distribution  (last backtest window)',
                xlabel='Annual Preventable Loss (£)', ylabel='Path Count')
        ax.legend(fontsize=9.5, loc='upper left', framealpha=0.85)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:,.0f}'))

# Delay Distribution
        ax = axes[1, 1]

        tier_order   = ['normal', 'surge', 'dragon']
        tier_palette = {
            'normal': COLORS['normal'],
            'surge':  COLORS['surge'],
            'dragon': COLORS['dragon'],}
        tier_labels = {
            'normal': f"Normal\n(n={df['Delay_Tier'].eq('normal').sum():,})",
            'surge':  f"Surge\n(n={df['Delay_Tier'].eq('surge').sum():,})",
            'dragon': f"Dragon\n(n={df['Delay_Tier'].eq('dragon').sum():,})",}

        log_delays = df[['Delay_Tier', 'Delay_min']].copy()
        log_delays['Log_Delay'] = np.log10(log_delays['Delay_min'])
        sns.violinplot(
            data=log_delays, x='Delay_Tier', y='Log_Delay',
            order=tier_order, palette=tier_palette,
            inner='quartile', linewidth=1.4, ax=ax,
            scale='width', cut=0,)

        dragon_log = log_delays[log_delays['Delay_Tier'] == 'dragon']['Log_Delay']
        ax.scatter(np.full(len(dragon_log), 2) + rng.uniform(-0.08, 0.08, len(dragon_log)),
                dragon_log, color='white', edgecolors=COLORS['dragon'],
                s=18, alpha=0.6, zorder=5)

        ax.axhline(np.log10(SLA_BREACH_MIN), color='red', ls='--', lw=2,
           label=f'SLA breach ({SLA_BREACH_MIN} min)')

        yticks_min = [5, 25, 90, 240, 480, 1440]
        ax.set_yticks([np.log10(v) for v in yticks_min])
        ax.set_yticklabels([f'{v:,} min' for v in yticks_min], fontsize=10)
        ax.set_xticklabels([tier_labels[t] for t in tier_order], fontsize=10.5)

        style_ax(ax, 'Fulfilment Delay Distribution by Tier  (log scale)',
                xlabel='Order Tier', ylabel='Delay (minutes, log scale)')
        ax.legend(fontsize=10, framealpha=0.85)

# Final layout
        fig.suptitle(
            'Dragon Revenue-at-Risk — Full Quantitative Risk Report  '
            '[Scale-Corrected Backtest]',
            fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig('dragon_risk_report.png', dpi=160, bbox_inches='tight')
        plt.show()

# P&L SUMMARY
        annual_gross = df['OrderValue_GBP'].sum() * (365 / date_range_days)

        sla_tail     = np.percentile(daily_loss_series, 99) * 252
        sla_moderate = np.percentile(daily_loss_series, 95) * 252

        print("\n  P&L SUMMARY  (Scenario 1 – Positive Quantities Only)\n")
        print(f"{'Metric':<50} {'Value':>14} {'% Gross':>8}")
        print(f"{'Annualised Gross Revenue':<50} £{annual_gross:>12,.0f}")
        print(f"{'GPD ES95  (daily tail, annualised)':<50} £{es95_gpd:>12,.0f} "
            f"{es95_gpd/annual_gross*100:>7.2f}%  ← headline")
        print(f"{'GPD VaR95 (daily tail)':<50} £{var95_gpd:>12,.0f} "
            f"{var95_gpd/annual_gross*100:>7.2f}%")
        print(f"{'MC  ES95  (annual, last backtest window)':<50} £{es95_mc:>12,.0f} "
            f"{es95_mc/annual_gross*100:>7.2f}%")
        print(f"{'MC  VaR95 (annual, last backtest window)':<50} £{var95_mc:>12,.0f} "
            f"{var95_mc/annual_gross*100:>7.2f}%")
        print(f"{'Margin impact (30% GM × MC ES95)':<50} £{es95_mc*GROSS_MARGIN:>12,.0f} "
            f"{es95_mc*GROSS_MARGIN/annual_gross*100:>7.2f}%")
        print(f"{'5-Year projected bleed (no action)':<50} £{es95_mc*5:>12,.0f}")
        print(f"{'Annual bleed — tail scenario (99th pctile day ×252)':<50} £{sla_tail:>12,.0f}")
        print(f"{'Annual bleed — moderate (95th pctile day ×252)':<50} £{sla_moderate:>12,.0f}")
        print(f"{'Backtest horizon':<50} {'Window-matched VaR (FIXED)':>22}")
        print(f"{'Backtest violations':<50} {n_viol:>12} / {len(violations)}")
        print(f"{'Kupiec p-value':<50} {kupiec_p:>14.4f}  "
            f"{'calibrated' if kupiec_p > 0.05 else 'needs review'}")
        print(f"{'Christoffersen p-value':<50} {christ_p:>14.4f}  "
            f"{'independent' if christ_p > 0.05 else 'clustered'}")
        print("-" * 50)
        print(f"{'GPD tail shape (ξ)':<50} {xi:>14.4f}")
        print(f"{'GPD scale (β)':<50} {beta:>14.2f}")
        print(f"{'POT threshold u (99th pctile daily loss)':<50} £{u:>12,.0f}")
        print(f"{'Exceedances above threshold':<50} {N_u:>12,}")
        print("=" * 50)
        print(f"\n  In a bad year, preventable dragon loss ≈ £{es95_mc:,.0f} (MC ES95).")
        print(f"  Tail scenario daily ×252 trading days → £{sla_tail:,.0f} bleed.")
        print(f"  Backtest: {n_viol} violations in {len(violations)} windows "
            f"({viol_pct:.1f}%) — model is "
            f"{'stable' if kupiec_p > 0.05 else 'NEEDS REVIEW'}.")

    else:
        SEED = 314159
        N_MC_PATHS = 10_000
        SURGE_PCT = cfg["SURGE_PCT"]
        DRAGON_PCT = cfg["DRAGON_PCT"]
        DRAGON_BIAS_EXP = cfg["DRAGON_BIAS_EXP"]
        NORMAL_MEAN_MIN = cfg["NORMAL_MEAN_MIN"]
        SURGE_MEAN_MIN = cfg["SURGE_MEAN_MIN"]
        DRAGON_MEAN_MIN = cfg["DRAGON_MEAN_MIN"]
        SIGMA_NORMAL = cfg["SIGMA_NORMAL"]
        SIGMA_SURGE = cfg["SIGMA_SURGE"]
        SIGMA_DRAGON = cfg["SIGMA_DRAGON"]
        SLA_BREACH_MIN = cfg["SLA_BREACH_MIN"]
        GROSS_MARGIN = cfg["GROSS_MARGIN"]
        ANNUAL_HOLDING_RATE = cfg["ANNUAL_HOLDING_RATE"]
        MIN_TRAIN_DAYS = cfg["MIN_TRAIN_DAYS"]
        WINDOW_STEP = cfg["WINDOW_STEP"]
        PURGE_GAP = cfg["PURGE_GAP"]

# LOAD DATA - Use loader.py which handles all data cleaning, filtering, and netting
        print("SCENARIO 2 BACKTEST")
        
        # Ensure required columns exist
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').dt.date
        if 'OrderValue_GBP' not in df.columns:
            df['OrderValue_GBP'] = df.get('OrderValue', df['Quantity'] * df['UnitPrice'])

        n = len(df)
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

        state["xi"]        = xi
        state["beta"]      = beta
        state["var95_gpd"] = var95_gpd
        state["es95_gpd"]  = es95_gpd
        state["kupiec_p"]  = kupiec_p
        state["christ_p"]  = christ_p
        state["n_viol"]    = n_viol
        state["viol_pct"]  = viol_pct
        
    state["df"] = df
    return state