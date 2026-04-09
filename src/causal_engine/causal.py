# src/causal/causal_engine.py
import gc
import logging
from tracemalloc import start
from matplotlib.pylab import sample
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import bootstrap
from typing import List
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

def run_causal_engine(
    df,
    scenario: int,
    cfg: dict,
    ctx=None,
    state: dict = None,
) -> dict:
    """
    Regression Discontinuity + Quantile Regression + GPD causal analysis.
    The CausalEngine class definition differs between scenarios
    """
    if state is None:
        state = {}
    df = state.get("df", df)
    _show = _plotly_show_alias(ctx)
    surge_idx = state.get("surge_idx", np.array([], dtype=int))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    rng = np.random.default_rng(314159)
        
    if scenario == 1:
        CALIPER = cfg["PSM_CALIPER"]  # Caliper for PSM matching
        PS_SUBSAMPLE = cfg["PSM_SUBSAMPLE"]
        QR_SUBSAMPLE = cfg["QR_SUBSAMPLE"]
        BOOTSTRAP_REPS = cfg["BOOTSTRAP_REPS"]
        QUANTILES = cfg["QUANTILES"]


        class CausalEngine:
            def __init__(self, df: pd.DataFrame, surge_idx: pd.Index, dragon_flag_col: str = "Is_Dragon"):
                if not isinstance(df.index, pd.RangeIndex):
                    df = df.reset_index(drop=True)
                self.df = df.copy()
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
                df['Country'] = df['Country'].fillna('Unknown').astype('category').cat.add_categories('Unknown')
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
                df['Hour'] = df['InvoiceDate'].dt.hour.fillna(-1).astype(np.int8)
                df['date'] = df['InvoiceDate'].dt.date
                df['Log_Quantity'] = np.log1p(df['Quantity'].clip(lower=1)).astype(np.float32)
                df['Net_Revenue_GBP'] = df['Net_Revenue_GBP'].astype(np.float32)
                df['OrderValue'] = df['OrderValue_GBP'].astype(np.float32)
                df['Will_Cancel'] = 0
                self.df = df
                logging.info(f"Features engineered — {df['dragon'].sum():,} validated dragons")

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
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Country']),
                        ('num', StandardScaler(), ['Log_Quantity', 'Hour'])],
                    remainder='drop',
                    sparse_threshold=0)
                model = Pipeline([
                    ('prep', preprocessor),
                    ('clf', LogisticRegression(
                        penalty='l2', solver='saga', C=1.0, max_iter=1000,
                        n_jobs=-1, random_state=314159, warm_start=True, tol=2.614e-4))])
                model.fit(X, y)
                self.ps_model = model
                ps_full = model.predict_proba(df[['Country', 'Log_Quantity', 'Hour']])[:, 1]
                self.df['ps'] = np.clip(ps_full, 1e-6, 1 - 1e-6).astype(np.float32)
                logging.info(f"Propensity fitted (n={len(sample):,}), PS range: [{self.df['ps'].min():.4f}, {self.df['ps'].max():.4f}]")

    # PSM
            def psm_caliper_exact_country(self):
                df = self.df
                treated = df[df['treatment'] == 1].copy()
                control = df[df['treatment'] == 0].copy()
                if len(treated) == 0:
                    self.ate_psm = np.nan
                    return
                def safe_qcut(s: pd.Series, q: int = 50) -> pd.Series:
                    try:
                        return pd.qcut(s, q=q, duplicates='drop')
                    except ValueError:
                        return pd.cut(s, bins=min(20, len(s.unique())), duplicates='drop')
                treated['ps_bin'] = safe_qcut(treated['ps']).astype(str)
                control['ps_bin'] = safe_qcut(control['ps']).astype(str)
                matched_pairs: List[pd.DataFrame] = []
                for country in treated['Country'].cat.categories:
                    t_sub = treated[treated['Country'] == country]
                    c_sub = control[control['Country'] == country]
                    if len(t_sub) == 0 or len(c_sub) == 0:
                        continue
                    merged = pd.merge(t_sub[['ps', 'Net_Revenue_GBP', 'ps_bin']], c_sub[['ps', 'Net_Revenue_GBP', 'ps_bin']],
                                    on='ps_bin', suffixes=('_t', '_c'))
                    merged = merged[np.abs(merged['ps_t'] - merged['ps_c']) <= CALIPER]
                    if len(merged) == 0:
                        continue
                    merged['dist'] = np.abs(merged['ps_t'] - merged['ps_c'])
                    merged = merged.sort_values(['ps_t', 'dist']).drop_duplicates('ps_t', keep='first')
                    matched_pairs.append(merged[['Net_Revenue_GBP_t', 'Net_Revenue_GBP_c']])
                if matched_pairs:
                    matched_df = pd.concat(matched_pairs, ignore_index=True)
                    ate = matched_df['Net_Revenue_GBP_t'].mean() - matched_df['Net_Revenue_GBP_c'].mean()
                else:
                    ate = np.nan
                    logging.warning("PSM: No caliper matches found.")
                self.ate_psm = float(ate)
                logging.info(f"PSM ATE = £{ate:,.0f}" if not np.isnan(ate) else "PSM: No matches")
                gc.collect()

    # QUANTILE REGRESSION
            def quantile_regression(self, quantiles: List[float] = QUANTILES):
                df = self.df
                for q in quantiles:
                    sample = df.sample(n=min(QR_SUBSAMPLE, len(df)), random_state=314159)
                    if len(sample) < 200:
                        logging.warning(f"QR {q:.1%}: Sample too small ({len(sample)}). Skipping.")
                        self.ate_qr[q] = np.nan
                        continue
                    formula = 'Net_Revenue_GBP ~ treatment + C(Country) + Log_Quantity + C(Hour)'
                    mod = QuantReg.from_formula(formula, sample)
                    try:
                        res = mod.fit(q=q, kernel='epa', max_iter=5000, p_tol=1e-8)
                        self.qr_models[q] = res
                        self.ate_qr[q] = float(res.params.get('treatment', np.nan))
                        self.p_qr[q] = float(res.pvalues.get('treatment', np.nan))
                        logging.info(f"QR {q:.1%} → ATE = £{self.ate_qr[q]:,.0f}, p = {self.p_qr[q]:.2e}")
                    except Exception as e:
                        logging.error(f"QR {q:.1%} failed: {e}")
                        self.ate_qr[q] = np.nan
                        self.p_qr[q] = np.nan
            gc.collect()

    # DRAGON METRICS
            def dragon_metrics(self):
                df = self.df
                dragons = df[df['dragon']]
                non = df[~df['dragon']]
                if len(dragons) == 0:
                    self._reset_dragon_metrics()
                    return
        # Time Convergence
                days = (df['date'].max() - df['date'].min()).days + 1
                months_covered = days / 30.4375
                total_dragons = len(dragons)
                monthly_dragons = total_dragons / months_covered
                premium = dragons['OrderValue'].mean() - non['OrderValue'].mean()
                qte = {}
                for q in QUANTILES:
                    if len(dragons) >= 50 and len(non) >= 50:
                        q_d = np.quantile(dragons['OrderValue'], q)
                        q_n = np.quantile(non['OrderValue'], q)
                        qte[q] = float(q_d - q_n)
                    else:
                        qte[q] = np.nan
                self.dragon_premium = float(premium)
                self.dragon_cancel_rate = 0.0
                self.dragon_cancel_uplift = 0.0
                self.monthly_dragons = int(np.round(monthly_dragons))
                self.months_covered = float(months_covered)
                self.dragon_qte = qte
                logging.info(
                    f"Dragons: {total_dragons:,} over {months_covered:.1f} months → {monthly_dragons:.1f}/month | "
                    f"Mean Premium: £{premium:,.0f} | 95th QTE: £{qte[0.95]:,.0f} | 99.9th QTE: £{qte[0.999]:,.0f}")

            def _reset_dragon_metrics(self):
                self.dragon_premium = 0.0
                self.dragon_cancel_rate = 0.0
                self.dragon_cancel_uplift = 0.0
                self.monthly_dragons = 0
                self.months_covered = 12.0
                self.dragon_qte = {q: 0.0 for q in QUANTILES}

            def annual_impact(self, gross_margin: float = 0.30):
                unfulfilled_mask   = self.df['Unfulfilled_Dragon']        # uses unified definition
                unfulfilled_rev    = self.df.loc[unfulfilled_mask, 'OrderValue_GBP'].sum()
                days               = (self.df['date'].max() - self.df['date'].min()).days + 1
                annualization      = 365 / days

                self.annual_revenue_risk      = float(unfulfilled_rev * annualization)
                self.annual_cancellation_loss = 0.0                       # Scenario 1 explicit zero
                self.total_annual_impact      = float(self.annual_revenue_risk * gross_margin)
                self.annual_dragons           = int(self.df['dragon'].sum() * annualization)

                logging.info(
                    f"Annualized: {self.annual_dragons:,} dragons → "
                    f"£{self.annual_revenue_risk:,.0f} gross risk → "
                    f"£{self.total_annual_impact:,.0f} profit impact ({gross_margin:.0%} margin)")
        
            def run_all(self):
                self.fit_propensity(); gc.collect()
                self.psm_caliper_exact_country(); gc.collect()
                self.quantile_regression(); gc.collect()
                self.dragon_metrics(); gc.collect()
                self.annual_impact(); gc.collect()
                self.plot_convergence()

            def plot_convergence(self):
                valid = []
                items = [
                    ('PSM Caliper', self.ate_psm),
                    ('Dragon Premium (mean)', self.dragon_premium)
                ] + [(f'QR {q:.1%}', self.ate_qr[q]) for q in QUANTILES] + \
                    [(f'Dragon QTE {q:.1%}', self.dragon_qte[q]) for q in QUANTILES]
        
                for label, value in items:
                    if not (pd.isna(value) or np.isinf(value) or value == 0):
                        valid.append((label, float(value)))
        
                if not valid:
                    print("No valid ATEs to plot.")
                    return
                methods, ates = zip(*valid)
                fig = go.Figure()
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FF6B6B', '#4ECDC4']
                fig.add_trace(go.Bar(
                    x=methods, y=ates,
                    text=[f"£{v:,.0f}" for v in ates],
                    textposition='outside',
                    marker_color=colors[:len(methods)]))
                fig.update_layout(
                    title="<b>Causal Lift + Dragon QTE Convergence (95th & 99.9th)</b>",
                    yaxis_title="Effect on Net Revenue (£)",
                    template="plotly_dark",
                    height=620,
                    yaxis=dict(range=[0, max(ates) * 1.35]))
                fig.add_annotation(
                    x=0.5, y=0.92, xref='paper', yref='paper',
                    text=f"<b>£{self.total_annual_impact:,.0f} ANNUAL IMPACT</b><br>{self.annual_dragons:,} dragons/year",
                    showarrow=False, font_size=16, bgcolor='gold', font_color='black')
                _show(fig)

            def summary_dashboard(self):
                print("\n" + "="*50)
                print(f"CAUSAL IMPACT DASHBOARD — SURGE + DRAGON (95th & 99.9th %ile)")
                print("="*50)
                print(f"{'Method':<32} {'ATE (Net Revenue)':>22} {'p-value':>15}")
                print("-"*50)
                for q in QUANTILES:
                    ate = self.ate_qr[q]
                    p = self.p_qr[q]
                    label = f"Quantile Reg ({q:.1%})"
                    if not pd.isna(ate):
                        ate_str = f"{ate:>+,.2f}" if abs(ate) < 1.0 else f"{ate:>,.0f}"
                        print(f"{label:<32} £{ate:>2,.0f} {p:>12.2e}")
                    else:
                        print(f"{label:<32} {'—':>20} {'—':>12}")
                if not pd.isna(self.ate_psm):
                    print(f"{'PSM (caliper 0.005)':<32} £{self.ate_psm:>2,.0f} {'—':>12}")
                else:
                    print(f"{'PSM (caliper 0.005)':<32} {'—':>2} {'—':>12}")
                print(f"{'Dragon Premium (mean)':<32} £{self.dragon_premium:>2,.0f} {'—':>12}")
                for q in QUANTILES:
                    qte = self.dragon_qte[q]
                    label = f"Dragon QTE ({q:.1%})"
                    if not np.isnan(qte):
                        print(f" {label:<32} £ {qte:>2,.0f} {'—':>12}")
                    else:
                        print(f" {label:<32} {'—':>2} {'—':>12}")
                print("-"*50)
                print(f"{'Time Coverage':<32} {self.months_covered:>2.1f} months")
                print(f"{'Dragon Cancel Rate':<32} {self.dragon_cancel_rate:>2.2%} (+{self.dragon_cancel_uplift:>+.1%} vs non-dragon)")
                print(f"{'Monthly Dragons':<32} {self.monthly_dragons:>2,}")
                print(f"{'Annual Dragons':<32} {self.annual_dragons:>2,}")
                print(f"{'Annual Revenue Risk':<32} £ {self.annual_revenue_risk:>2,.0f}")
                print(f"{'Annual Cancellation Loss':<32} £ {self.annual_cancellation_loss:>2,.0f}")
                print(f"{'TOTAL ANNUAL IMPACT':<32} £ {self.total_annual_impact:>2,.0f}")
                print("="*50)  # Fixed: missing closing parenthesis

# EXECUTION
        engine = CausalEngine(df=df, surge_idx=surge_idx, dragon_flag_col='Is_Dragon')
        engine.run_all()
        engine.summary_dashboard()

        state["engine"]   = engine        # fitted CausalEngine instance
        state["ate_psm"]  = engine.ate_psm
        state["ate_qr"]   = engine.ate_qr
        state["p_qr"]     = engine.p_qr
        state["dragon_premium"] = engine.dragon_premium
        state["annual_dragons"] = engine.annual_dragons
        state["total_annual_impact"] = engine.total_annual_impact

    else:

        CALIPER = cfg["PSM_CALIPER"]  # Caliper for PSM matching
        PS_SUBSAMPLE = cfg["PSM_SUBSAMPLE"]
        QR_SUBSAMPLE = cfg["QR_SUBSAMPLE"]
        BOOTSTRAP_REPS = cfg["BOOTSTRAP_REPS"]
        QUANTILES = cfg["QUANTILES"]
        MAX_COUNTRIES = cfg["MAX_COUNTRIES"]


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
                _show(fig)

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
        
        state["engine"]   = engine        # fitted CausalEngine instance
        state["ate_psm"]  = engine.ate_psm
        state["ate_qr"]   = engine.ate_qr
        state["p_qr"]     = engine.p_qr
        
    state["df"] = df    
    return state