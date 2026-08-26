## Limitations & Trade-offs 
 
Public portfolio project on 374 days of UCI Online Retail (2010–11). Accepted at the outset: public data, one year of tickets, undergraduate / portfolio compute, and the pipeline as implemented. Wiring the stack does not close the science — there is still a lot to learn. Sterling figures are directional outputs under stated assumptions, not operational forecasts. Identities: monograph §13 and §15.

Structural

Delays and dragon labels are assigned, not observed. The file has no pick-complete time, SLA clock, or dragon flag. Delay is a clipped log-normal informed by public WERC/CSCMP summaries (average-retail, not a WMS extract). Dragons are a thin value-biased sample, not a detector. Nothing about SLA, unfulfilment, holding, Monte Carlo, Hawkes times, or backtest loss is warehouse telemetry.

Vintage is 2010–11. One year gives 34 / 33 backtest windows; coverage tests have low power. Five-year arithmetic assumes a constant dragon rate, margin and SLA. Serial dependence is left in (no declustering); Hill / GEV / POT standard errors are optimistic.

Identification

Surge ATE on net revenue is a negative control (assigned; must be near zero; empirically £7 / −£21). Dragon QTE on value is the object of interest and is partly mechanical (V^γ sampling). PSM: 50k-row logit, caliper 0.005, no balance diagnostics; Scenario 1 may reuse controls. QR: one 180k subsample. Will_Cancel = 0.

Different functionals, not refinements

Do not add the sterling numbers. Do not read Kalman bleed as a clustered Monte Carlo. Hawkes (μ,α,β) do not enter B = g N U / n_D.

┌───┬───────────────────────────────────────────┬────────────────────────────────────┐
│   │ What it is                                │ S1 / S2                            │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ E │ g(365/Δ)U                                 │ £43,209 / £9,123                   │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ F │ LDA ES95 (calendar-day mean, i.i.d. year) │ £60,134 / £11,716                  │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ G │ Kalman bleed (clipped count N)            │ £41,835 / £10,197                  │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ H │ Last-window ES95 (active-day mean)        │ £248,220 / £75,001                 │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ I │ GPD mean above the 99th of daily loss     │ £35,966 / £7,102 (daily)           │
├───┼───────────────────────────────────────────┼────────────────────────────────────┤
│ J │ p₉₉(ℓₜ)× 252                              │ stress exhibit, not an expectation │
└───┴───────────────────────────────────────────┴────────────────────────────────────┘

The E vs G gap is N versus 365/Δ⋅ n_D, not evidence that clustering is small.

As implemented

• Monte Carlo. i.i.d. log-normal; calendar-day mean; active-day CV capped at 3. Hawkes is not injected.
• Hawkes. α at the 0.99 bound; branching ratio ≈ 0.0024; half-life ≈ 0.10 s. Printed 30-day count 1.5 is a coarse-grid artefact (true increment ∼ 0.08).
• Kalman. Fixed Q,R (not priors); monthly rate clipped to {6,…,10}× 12. Scenario 1 bleed hard-codes £147,583 / 66. Mitigation rows are ρ-arithmetic, not treatment effects.
• EVT / GPD. Quantity EVT is a Fréchet diagnosis; no GPD on quantity. The only GPD is daily realised loss, Nᵤ=4, VaR clamped to the 99th percentile - a far-tail daily mean excess, not annual 95% VaR. GPD and Hawkes are full-sample, not purged.
• Backtest. Replays the delay DGP. Test loss is V+H; window VaR is margin-adjusted — 0/34 is not a calibration theorem. Kupiec as coded returns p=1 at zero violations (correct k=0 LR is p≈ 0.062). The 30% GM × MC ES95 row double-counts g.

Still a lot to learn

This is a first wiring, not a solved warehouse problem. Open: the observed definition of dragon; the right Hawkes clock (the minute-scale fit collapsed); putting clustering into the annual loss; EVT under dependence; a GPD with more than four excesses; a real treatment (capacity -> cycle time); count projection without a clip; which of E–J a desk should use once delays are no longer assigned.

Compute

The shortcuts above are also what fits in a single-machine, single-seed pass of two scenarios.

• One seed (314159) — the table is a point, not a sampling distribution.
• Backtest re-draws 10,000× 365 paths per window, which is why Hawkes is not thinned into the year and GPD/Hawkes are not re-fit inside each purge.
• Hawkes 15k-point grid is too coarse for β≈ 410; Kalman Q,R are not estimated; the dashboard reads cached stages.pkl.

A production pass would need a seed ensemble, purged refits, and a Hawkes-modulated year. That multiply is not in this repo.

What this does not show

The stack can be derived and wired, with every headline a named functional. It does not show a validated operational model, a real SLA effect, or a policy to hold 99th-percentile service on dragons. Those need observed delays, more data, and more compute. There is still a lot to learn.
