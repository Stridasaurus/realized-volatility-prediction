# Speaker Notes — "Can a Small LSTM Beat HAR?"

10-minute slot, ML audience (MTH 5320). They know deep learning cold; they do **not**
know quant-finance vocabulary (HAR, QLIKE, range-based estimators, DM test) — define
each the first time it appears, briefly, then move on. Don't over-explain LSTM/Optuna —
that part of the audience is your home turf.

Target pacing below sums to ~9:15, leaving a buffer before Q&A. Times are cumulative
elapsed, not per-slide duration.

---

## Slide 1 — Title (0:00 → 0:30)

- One line: "Small LSTM vs. a 40-year-old three-line regression, on volatility."
- Tee up the twist early so it lands as a structure, not a surprise: the honest answer
  turned out to be **two experiments**, not one — a null, then a reversal. Say that
  now so the audience knows to expect a null result and doesn't read it as you failing.

## Slide 2 — The Setup (0:30 → 1:30)

- **HAR** (Corsi, 2009): regress volatility on three lagged terms — yesterday's value,
  the trailing 5-day average, the trailing 22-day average. That's it. Three OLS
  coefficients. It is the standard benchmark in the realized-vol forecasting
  literature and is notoriously hard to beat with anything fancier.
- The catch: HAR's reputation was built on **high-frequency realized variance**
  (intraday tick data, 5-min sums of squared returns). This project only has daily
  OHLC bars — so the targets had to be **range-based** variance proxies instead
  (define range-based in one clause: "uses the day's high/low/open/close instead of
  intraday ticks to estimate variance more efficiently than a single squared close-to-
  close return"). Different estimator family, unclear a priori whether HAR's edge
  transfers.
- This reframing is why the project exists: is HAR's dominance a property of the model
  or a property of the high-frequency data it was validated on?

## Slide 3 — Method, one slide (1:30 → 2:45)

- This is the slide that earns credibility with an ML audience — spend real time here.
- Frozen, checksummed data snapshot — no silent redefinition of the dataset mid-project.
- Chronological split + **monthly walk-forward retraining** in the test region (78
  folds) — not one static train/test split. Explain briefly: every model, including
  HAR, is refit monthly on a rolling window, so nobody has a "stale" advantage.
- Hyperparameters tuned **once**, before the model ever saw test data, then frozen —
  this is the pre-registration equivalent for a course project. Emphasize: the same
  frozen hyperparameters get reused unchanged across every later experiment (h=5, OC
  target, RV+aux) — that's a deliberate handicap on the network, not laziness, because
  it keeps every later comparison controlled to a single varying factor.
- 10-seed ensembles + Diebold–Mariano test — flag that this is a paired significance
  test for forecast accuracy (like a paired t-test on loss differentials, but built for
  autocorrelated forecast errors). Don't over-explain here; it's in the technical
  appendix if asked.

## Slide 4 — Finding 1: HAR earns its reputation here (2:45 → 3:45)

- This slide exists to establish that HAR is actually a strong baseline **on this
  data**, not assumed. Ran the full classical field: persistence, EWMA, AR(1), GARCH.
- HAR wins outright against everything except GARCH on the total-variance target,
  where the two are statistically tied. Say plainly: "HAR is the right benchmark — it
  isn't a strawman."
- QLIKE is the loss/metric on the y-axis — flag it by name now, define in one clause
  ("a loss for variance forecasts that penalizes under-forecasting risk more than
  over-forecasting it") and point to backup for the formula if asked.

## Slide 5 — Finding 2: Architecture alone finds nothing (3:45 → 4:45)

- Give this null result its full weight — don't rush past it apologetically. It's a
  real finding, not a failed experiment.
- Same information HAR has (just the target's own lagged history), fed to a tuned
  LSTM instead of an OLS regression. Result: statistically indistinguishable from HAR,
  both horizons (p = 0.585, 0.667).
- The takeaway for an ML audience: **more flexible function class ≠ automatic edge**
  when the input information is exactly the same. The LSTM successfully re-derives
  something HAR-shaped from raw lags — it just doesn't beat it.

## Slide 6 — Finding 3: Auxiliary features reverse the null, decisively (4:45 → 6:00)

- The turn. Same frozen architecture, same frozen hyperparameters — the only change is
  adding a small, pre-registered set of auxiliary features (SPY return/volume, VIX
  level and change).
- 17–19% QLIKE improvement over HAR, on **every** target/horizon block tested, p-values
  from 5×10⁻⁸ to 2×10⁻⁴, 100% seed agreement (all 10 random seeds agree on direction).
- Say the sentence that makes this land: "Nothing about the network changed. What
  changed is what it was allowed to look at."

## Slide 7 — Where it comes from: features, not architecture (6:00 → 7:00)

- Direct head-to-head: RV+aux network beats the RV-only network by itself
  (p = 7.2×10⁻⁹) — so the edge isn't just "LSTM > HAR," it's specifically attributable
  to the auxiliary information, not the architecture change.
- Robustness sweep in one breath: present at both horizons, both targets, and in every
  regime cut (COVID, calm 2021, 2022 bear, 2023–26 recovery) — cumulative advantage
  never gives anything back, including in the calm periods.

## Slide 8 — Caveat: the network is data-hungry (7:00 → 7:45)

- Practical, slightly humbling result: at ~2 years of training history HAR wins
  decisively; at ~5 years HAR still leads; only at the full ~12.6-year window does the
  network reach parity (this is the RV-only comparison, used as the cleanest read on
  architecture vs. data volume in isolation).
- One-line moral: if you don't have a long history, don't reach for a sequence model
  here — HAR is the better choice at short samples.

## Slide 9 — Limitations (7:45 → 8:30)

- Say these fast and confidently — they read as rigor, not weakness, to an ML/stats
  audience.
- Multiplicity: many DM tests reported, but headline p-values are orders of magnitude
  below any reasonable multiple-comparison correction.
- Statistical, not economic: no transaction-cost or vol-targeting backtest — this is a
  forecasting-accuracy claim, not a trading-strategy claim.
- Floor, not ceiling: this establishes a lower bound on extractable signal beyond HAR,
  not the max achievable.

## Slide 10 — Conclusion (8:30 → 9:15)

- Land the floor claim as written on the slide, slowly: there is substantively
  large predictive information beyond HAR — at least 17% of QLIKE — and it lives
  in the auxiliary market-state features, not the deep architecture.
- Close with the two-line summary: RV-only was a genuine null; RV+aux was decisive
  everywhere it was tested. The honest project reports both.
- Stop talking. Let Q&A start.

## Slide 11 — Backup (only if pulled into Q&A)

- Full test-span forecast plot. Use if asked "does it actually track realized vol or
  just win on average" — this shows the LSTM visually hugging realized vol tighter
  than HAR, especially around volatility spikes.

---

# Technical Appendix (for Q&A — not spoken from directly)

## QLIKE — the loss function

For a variance prediction $\hat\sigma^2$ and realized (proxy) variance $\sigma^2$:

$$\text{QLIKE} = \frac{\sigma^2}{\hat\sigma^2} - \ln\!\left(\frac{\sigma^2}{\hat\sigma^2}\right) - 1$$

- Always ≥ 0, zero only at a perfect forecast. Asymmetric: **under**-forecasting
  variance ($\hat\sigma^2 \ll \sigma^2$) is penalized far more heavily than
  over-forecasting — the ratio term blows up as $\hat\sigma^2 \to 0$, while it only
  grows logarithmically as $\hat\sigma^2 \to \infty$. That's the right shape for risk
  forecasting: understating risk is the costly mistake.
- Robust to noisy volatility proxies: Patton (2011) shows QLIKE preserves the true
  ranking of forecasters even when the target you're scoring against is itself a noisy
  proxy for "true" latent variance — as long as the proxy is conditionally unbiased.
  That's *why* QLIKE (not MSE) is the primary metric here, and why the "conditionally
  unbiased" property of Rogers–Satchell/Garman–Klass (below) matters: it's the
  precondition that licenses using QLIKE for ranking at all.
- **This project trains directly on QLIKE**, not just evaluates on it (`src/train.py`,
  `qlike_loss_torch`). The network outputs a log-variance prediction `pred_log`; the
  loss exponentiates it, clamps to a pre-specified positive floor, forms the ratio
  against the level-space target, and applies the QLIKE formula — so gradients flow
  through the `exp()` back to the raw network output. There's an MSE-of-log fallback
  wired in for numerical divergence, but across all 10 seeds in every experiment it
  never triggered — 100% of training runs used QLIKE end to end.
- **Why the floor?** QLIKE has a $\sigma^2/\hat\sigma^2$ term — if the network
  predicts something at or near zero variance, the loss (and its gradient) explodes.
  The floor is a small pre-specified positive constant, fixed at data-freeze time
  (before any test contact) from the calibration data, not tuned to the results. Floor
  bind-rate (how often the floor was actually active) is logged every run as a
  diagnostic; it stayed at 0% for the network and HAR at test time — only the naive
  persistence baseline tripped it.

## Range-based volatility estimators — why not just squared daily returns

Squared close-to-close returns are an unbiased but extremely *noisy* proxy for daily
variance (huge day-to-day sampling error since it's built from a single draw). Range
estimators use the day's full OHLC bar to reduce that noise, in the same spirit as
using tick data — but from data you already have.

- **Rogers–Satchell** (1991): uses open, high, low, close; a drift-independent
  estimator of the variance of the trading session. Combined here with the squared
  overnight (close-to-open) return to build the **total variance (TV)** target —
  Rogers–Satchell alone only captures the session, so the overnight leg is added back
  in to make the target span the full calendar day.
- **Garman–Klass** (1980): also OHLC-based, more statistically efficient than
  close-to-close under its assumptions, used here for the **open-to-close (OC)**
  target — i.e., deliberately scoped to just the trading session, no overnight leg.
- Both are only "conditionally unbiased" for the specific object they're built to
  estimate (TV → total daily variance; OC → session variance) — that's why every claim
  in the report is carefully scoped to one or the other and never conflated. This
  conditional-unbiasedness is also the technical precondition for QLIKE-based ranking
  to be valid (see Patton, 2011, above).
- Calibration: before trusting either proxy, both were checked against squared-return
  benchmarks on the training span and had to fall inside a pre-set tolerance band
  ([0.8, 1.25]) — this passed for both targets.

## HAR — full specification

$$\log \sigma^2_{t+1} = \beta_0 + \beta_D \log\sigma^2_t + \beta_W \overline{\log\sigma^2}_{t-4:t} + \beta_M \overline{\log\sigma^2}_{t-21:t} + \varepsilon_t$$

- Three OLS terms: daily (lag-1), weekly (5-day trailing mean), monthly (22-day
  trailing mean) log-variance. Fit here in **log space** with a lognormal
  back-transform for point forecasts (Jensen's-inequality correction), not Corsi's
  original level-space form — a deliberate, pre-registered choice, not an oversight.
- Why it's hard to beat: volatility is long-memory / slowly mean-reverting, and these
  three terms are a remarkably good cheap approximation to a fractionally-integrated
  process without needing to estimate a fractional-differencing parameter directly.
  It's basically a parsimonious, interpretable proxy for long-memory dynamics.
- Refit monthly on the rolling window in the test region, exactly like every other
  model — no static-fit advantage or disadvantage.

## GARCH(1,1) — full specification

Mean equation: returns $r_t = \mu + \epsilon_t$, $\epsilon_t = \sigma_t z_t$,
$z_t \sim \text{iid}(0,1)$.

Variance equation:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- Fit by (quasi-)maximum likelihood on **close-to-close returns** (the `arch` package
  here), so its native forecast object is *total* return variance — properly scored
  against the TV target; scoring it against OC is flagged in the report as an object
  mismatch (GARCH doesn't know about the open/close split at all).
- It's the one model statistically indistinguishable from HAR on TV (p = 0.64 at h=1,
  0.37 at h=5) — worth saying out loud if asked "is HAR really the best classical
  model": no, GARCH ties it on the target GARCH is actually built for.

## EWMA (RiskMetrics)

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2$$

Decay parameter $\lambda$ fit on the training span (not fixed at the classic 0.94)
following the J.P. Morgan/Reuters (1996) RiskMetrics convention. Included as a cheap,
zero-parameter-estimation-cost benchmark; HAR beats it significantly everywhere.

## Diebold–Mariano test + HLN correction + HAC variance

Given two competing forecasts, form the loss differential $d_t = L(\text{model}_t) -
L(\text{bench}_t)$ (here $L$ = QLIKE). Under the null of equal predictive accuracy,

$$DM = \frac{\bar d}{\sqrt{\widehat{\text{Var}}(\bar d)}}$$

- The variance of $\bar d$ is estimated with a **HAC (Newey–West/Bartlett) long-run
  variance** estimator, not the naive sample variance — because forecast errors are
  serially correlated, especially at h=5 (a 5-day-average target means adjacent daily
  loss differentials mechanically overlap). Truncation lag is set ≥ h−1 as a hard
  floor.
- **Harvey–Leybourne–Newbold (1997) correction**: a finite-sample scale correction to
  the DM statistic, since the raw DM stat is asymptotic and can over-reject in
  moderate samples. The corrected stat is referred to a Student-t(n−1) distribution
  rather than standard normal.
- Sign convention here: negative mean loss differential = the model (LSTM) beats the
  benchmark (HAR).
- Decision rule (pre-registered, "S6" in project notes): claim an edge only if
  **DM p < 0.05 on the 10-seed ensemble-mean prediction** *and* the same-signed loss
  differential holds in **≥ 80% of individual seeds** — protects against a single
  lucky/unlucky seed driving the ensemble mean.

## Giacomini–White framework (why monthly rolling refit, not one static fit)

Giacomini & White (2006) formalize predictive-ability testing when the forecasting
model itself is periodically **re-estimated** on a rolling window, rather than fit
once and frozen. That's exactly this project's setup (monthly refit, 78 folds, fixed
3,167-day rolling window) — the rolling-refit protocol is chosen specifically so the
DM comparisons stay valid under that framework rather than requiring a single static
model fit, which would be unrealistic for volatility forecasting in practice anyway.

## LSTM architecture and exact training setup

- `SmallLSTM`: single-layer `nn.LSTM` (hidden size 100) → take the **last timestep's**
  hidden state → dropout (0.096) → linear head to a single scalar. That scalar is the
  predicted **log**-variance. Deliberately shallow/small — the stated design belief is
  that at this data scale (~12.6 years of daily data, one asset) the risk is
  overfitting, not underfitting, so capacity was capped rather than searched upward.
- Inputs: a sliding window of length 28 (`seq_len`, tuned) trading days of feature
  vectors. RV-only feature set = lagged log-target (lags 0–21) plus 5-day/22-day
  log-aggregates — deliberately the *same information HAR sees*, just handed to a
  flexible model instead of a linear one. RV+aux adds SPY return, |return|, overnight
  return, volume ratio (Tier 2) and log VIX, VIX change, VIX-implied daily variance
  (Tier 3) — all close-of-day-available, never revised.
- Training: full determinism (fixed seeds, `torch.use_deterministic_algorithms(True)`),
  CPU-only specifically to avoid CUDA nondeterminism. Early stopping on validation
  QLIKE. 10 seeds per fold; the **seed-ensemble mean** prediction sequence (not any
  single seed) is the object DM tests are run against — reduces variance from
  initialization/training-order noise and is the pre-specified object of inference.
- Scaling: a train-only-fit scaler (fit exclusively on training indices, applied
  everywhere) — never fit on validation or test data, to avoid leakage through
  normalization statistics.

## Optuna tuning

50-trial Optuna search (TPE sampler, the library default), single search on TV h=1
using train/validation only, minimizing validation QLIKE. Searched: hidden size,
number of layers, dropout, learning rate, weight decay, batch size, sequence length.
Result frozen (best validation QLIKE 0.4027) and reused unchanged for h=5 and for the
OC target — deliberately, to keep every downstream comparison controlled to a single
varying factor (see Method slide notes above).

## VIX-derived features

VIX is CBOE's model-free, options-implied 30-day expected variance measure (computed
from a strip of SPX option prices, not from any single option's Black–Scholes
implied vol) per the Cboe methodology whitepaper. Three derived features used here:
log VIX level, day-over-day VIX change, and a VIX-implied daily variance conversion
(de-annualizing the VIX index into a comparable daily-variance unit). It's forward-
looking, options-market information — categorically different from anything in the
RV-only feature set, which is why it's a plausible source of the auxiliary edge.

## Leakage controls

- Every feature at day *t* uses only information available at the close of day *t*,
  verified by an explicit **future-perturbation test**: perturb the data after day
  *t* and assert the day-*t* feature value is unchanged (in the 116-test `src/`
  suite).
- Scalers fit on training indices only; series never shuffled (sequence models need
  temporal order preserved); an embargo gap between splits sized to the largest model
  lookback (66 trading days) so no lookback window straddles a split boundary.
