# Volatility Forecasting — Manifesto

> **Layer 1 (root).** This is the single source of truth. The *why*, the boundaries, the
> invariants, the module map, and the vocabulary live here. Everything below — the per-module
> `SPEC.md`s, their `design.md`s, the `CLAUDE.md`s — inherits from this and must not contradict it.
> Implementation detail that will drift does **not** belong here; it belongs one layer down.

---

## 0. Spine — read this first

**One sentence:** *HAR's benchmark-beating reputation is established for high-frequency realized
variance; test whether it transfers to a daily range-based target, whether a deep model can beat
it there, and validate the answer without self-deception.*

The capability on display is honest measurement under adversarial self-scrutiny. A single model
family can only ever establish a **lower bound** on extractable signal: beat HAR by 2% and you've
shown the signal-beyond-HAR is *at least* 2%; tie HAR and you've learned *this model* found nothing
— not that a better one couldn't. The true **ceiling** (the target proxy's own noise / the
forecastability floor) is a property of the target, not of any one model; computing it is a
deliberate Stage-3 analysis (§5), not something the base pipeline delivers. **So the base project
claims a floor, never a ceiling.** A rigorously validated *tie* with HAR is a real result; a
suspicious *win* is worth less than an honest tie.

**The single design rule everything inherits from — Shared library, thin notebooks.**
Every piece of logic that must be *identical* across experiments lives in `src/` as importable
Python. Notebooks configure one experiment, call the library, and save artifacts. This makes
"consistent metrics across experiments" true *by construction*: fix a bug once, every experiment
inherits the fix. If two experiments could ever disagree on RV construction, the splitter, the
metrics, the baselines, the model harness, or the artifact format, the comparison is void.

---

## 1. Purpose & the problem

Volatility is highly forecastable, and the **HAR** model — three OLS terms on daily, weekly, and
monthly lagged volatility — is famously hard to beat. But that reputation was earned on a specific
target: **high-frequency realized variance** (the sum of squared *intraday* returns), the object
Corsi's HAR was built on. This project is constrained to **daily OHLC data**, so its target is a
**daily range-based variance** (Garman–Klass) — a distinct, older estimator family, and a *proxy*
for latent daily volatility rather than the canonical high-frequency RV. Whether HAR's dominance
*transfers* to this range-based target is an open empirical question, not a settled fact — and
assuming it is the trap this framing avoids.

The research question, stated honestly:

> HAR dominates for high-frequency RV. (1) Does that dominance *transfer* to a daily range-based
> target? (2) Can a deep sequence model beat HAR there? (3) *Where does any edge live*?

The point is not to win a leaderboard. It is to (a) build a leak-free, reproducible forecasting
pipeline, (b) measure whatever signal exists beyond a strong benchmark, and (c) report wherever the number
lands. The economic-significance question — does a statistical edge convert into money — is
explicitly a **separate follow-on project**, specced but not built here (see §4, §8).

---

## 2. Core users & concrete use cases

This document serves three readers, in priority order:

1. **The grader / evaluating reader.** Deliverable is a GitHub repo **plus** a written report
   **plus** a presentation. They reward correct, leak-free, rigorously-validated work over a
   flashy result. Provenance, honesty, and reproducibility *are* the grade.
2. **Future-self / the portfolio reader.** After submission this becomes a portfolio piece and
   the seed of the economic-significance follow-on. Stage boundaries (§5) exist so the grade-ready
   cut is a clean stopping point that later work extends without rewrites.
3. **The AI coding agent (Claude Code)** that builds each module from the `SPEC.md` this manifesto
   implies. Everything here is written so an agent can act on it without guessing.

**Concrete use cases** (these are the *experiments*, each a thin notebook over the library):
score the classical baselines, tune one net leak-safely, compare the net to HAR on identical
inputs, add leak-safe auxiliary features, study window decay, and produce the final comparison.

---

## 3. Project-level success criteria — measurable & executable

Definitions of done that map to a runnable check, not a vibe. Each module's `SPEC.md` owns its
full test list; these are the project-level gates.

| # | Criterion | How it's checked (executable) |
|---|-----------|-------------------------------|
| S1 | **The bar exists.** HAR and GARCH(1,1) are scored on the frozen splits. | `99_evaluation` loads their `metrics.json`; HAR QLIKE is present and finite. |
| S2 | **The pipeline is reproducible from nothing.** | Clean clone + frozen data snapshot + pinned `requirements.txt` runs end-to-end with no manual steps; snapshot SHA-256 matches the committed hash. |
| S3 | **No lookahead, provably.** | The feature leakage test (§7) passes: perturbing data after day *t* leaves every feature value at day *t* unchanged. Scaler fit-indices ⊆ train indices. |
| S4 | **Results are pure replay.** | `99_evaluation` rebuilds the headline table from saved artifacts, re-running **no** training. |
| S5 | **Significance is reported, not asserted.** | Diebold–Mariano test vs HAR on the QLIKE loss differential emits a p-value per model. |
| S6 | **Edges survive seed noise.** | Headline numbers are mean ± std over ≥10 seeds; any claimed edge exceeds the seed spread. |
| S7 | **The edge decomposition is pre-committed and reported.** | The three "where the edge lives" cuts (regime, horizon, source) are computed and reported *regardless* of whether the net wins, ties, or loses. |

**Done-check for this manifesto:** a fresh reader can list the modules they'd build and the rules
every module must obey, without guessing.

---

## 4. Scope boundaries & non-goals

Stated as hard boundaries so specs don't quietly expand them.

- **ONE asset (SPY) in v1.** Multi-asset (QQQ, a single name, a second period) is robustness
  *stretch*, meaningful only after a single-asset result exists.
- **A daily range-based variance (Garman–Klass, open-to-close) is the v1 target** — named honestly
  as a *proxy*, not canonical RV, and covering the trading session only (overnight excluded). **An
  overnight-inclusive daily target and high-frequency (5-min) realized variance — the standard RV,
  the target HAR's reputation is built on — are Stage-3 stretches** (the latter is the prerequisite
  for realized semivariance; the former needs no new data).
- **NO economic-significance backtest in v1.** Vol-targeted position sizing, Sharpe/drawdown
  comparison, transaction costs — deferred to its own follow-on project (spec recorded, build later).
- **NO cadence or window-length *sweep*.** Retrain cadence, rolling length, and architecture are
  **frozen controls**, not experiments. Letting them become experiments is the grid that kills
  the timeline.
- **NO architecture search** beyond tuning sizes *within* the frozen LSTM family.
- **NOT a live or production system.** This is an offline research pipeline over a frozen data
  snapshot; nothing pulls live data at experiment time.

---

## 5. Stages — grade-ready, then portfolio

Build in this order. **Never start a later stage until the previous one is locked and saved.**
The way this project dies is falling in love with a fancier model and never finishing the boring,
decisive validation.

- **Stage 0 — Harness (must work before any net).** Data snapshot + RV construction + frozen
  splits + metrics + HAR + GARCH, scored on the frozen splits. *Everything after is upside.*
- **Stage 1 — Tuned net + controlled comparison.** Leak-safe Optuna search → one frozen HP set;
  RV-only net vs HAR on identical inputs.
- **Stage 2 — Real results (this is the grade-ready cut).** RV+aux experiment, decay curve,
  final evaluation notebook. **Lock and save before any stretch.**
- **Stage 3 — Portfolio extensions.** An **overnight-inclusive daily target** (per-day Rogers–Satchell
  + overnight², or windowed Yang–Zhang) so the target approximates *total* daily variance instead of
  open-to-close only — this repairs the QLIKE-unbiasedness caveat and is computable from the **same
  daily OHLC snapshot (no new data)**; keeping GK alongside it turns the swap into a result of its own
  (does the HAR-vs-net story change once overnight variance is in the target?). Then a genuine
  **forecastability-floor analysis** — estimate the target proxy's own measurement noise / irreducible
  loss, so a *ceiling* on predictability can be stated honestly (this is what lets a tie with HAR mean
  something instead of nothing); high-frequency 5-min RV (the canonical target HAR's reputation rests
  on, and the one item here that needs *new* intraday data) → realized semivariance; multi-asset
  robustness; data refresh + second-source cross-check. A PK+GK+RS **blend** stays an *optional*
  robustness footnote only — all three are open-to-close, so it addresses estimator-choice
  arbitrariness, **not** the overnight gap. Then the economic-significance follow-on as a separate project.

Stage 2 is the deliverable a grader sees. Stage 3 is what makes it a portfolio piece.

---

## 6. Capability / module map  *(load-bearing)*

The **modules** are the durable `src/` engine — the things an agent builds and tests. The
**notebooks are not modules**; they are thin experiments that configure and call the engine, and
map to the use cases in §2. Carved so each module is one focused agent session. Dependency
direction points *downward* (a module may use those listed under "Depends on," nothing else).

> Open carve question (see §8): whether `models`+`train` are one module or two, and whether
> `baselines` is one module or split HAR/GARCH. Default: one each.

### `data` — `src/data.py`
- **Responsibility:** turn the frozen price/VIX snapshot into the canonical RV series.
- **Owns:** snapshot loading; RV construction (Garman–Klass first); the documented adjustment
  basis; the trading-day calendar that indexes everything downstream.
- **Doesn't own:** features, splitting, any model.
- **Depends on:** the committed snapshot only. (No network at experiment time.)
- **Done-check:** snapshot checksum matches committed hash; RV is non-negative with no NaN after
  warmup; GK matches a hand-computed value on a small OHLC fixture.

### `features` — `src/features.py`
- **Responsibility:** build the leak-safe feature matrix (RV history + tiered aux features).
- **Owns:** the feature-level leak contract; the Tier 1/2/3 aux menu; scaler/transform objects
  (fit on train only, applied forward).
- **Doesn't own:** the *temporal* split (that's `splits`); the RV target (that's `data`).
- **Depends on:** `data`.
- **Done-check:** the future-perturbation leakage test (§7) passes; every scaler's fit-indices are
  a subset of the training indices; all feature columns finite.

### `splits` — `src/splits.py`
- **Responsibility:** **the** canonical walk-forward splitter — single source of truth.
- **Owns:** the frozen split definition (train/val/test boundaries), the embargo gap, the
  monthly walk-forward retrain folds inside the test region.
- **Doesn't own:** anything model- or feature-specific; it operates on a date index alone.
- **Depends on:** nothing.
- **Done-check:** zero overlap between any train fold and its test fold; embargo ≥ configured gap;
  every fold chronological (test strictly after train); deterministic; union of test folds covers
  the test region with no gaps.

### `metrics` — `src/metrics.py`
- **Responsibility:** scoring and significance.
- **Owns:** `qlike()`, `rmse()`, the Diebold–Mariano loss-differential test; the
  variance-space discipline.
- **Doesn't own:** any data or model.
- **Depends on:** nothing.
- **Done-check:** `qlike`/`rmse` match a hand-computed fixture; `qlike` floors/rejects
  non-positive predictions; a guard catches volatility-vs-variance unit mismatch.

### `baselines` — `src/baselines.py`
- **Responsibility:** the classical reference models.
- **Owns:** HAR (the real bar to beat) and GARCH(1,1) (context baseline only — fit on returns,
  not a serious competitor; see Glossary).
- **Doesn't own:** the net.
- **Depends on:** `data`, `splits`, `metrics`.
- **Done-check:** HAR produces one RV-space forecast per test date with sane coefficients; GARCH's
  variance forecast is positive and in **variance space** before QLIKE touches it.

### `model + harness` — `src/models.py` + `src/train.py`
- **Responsibility:** the frozen small-LSTM family and a seeded, leak-safe train/eval loop.
- **Owns:** the LSTM definition (family frozen; Optuna tunes sizes within it); seeding;
  early stopping; log-RV modeling with exponentiation back to level space; QLIKE-with-floor
  training loss with an MSE-of-log-RV fallback; multi-seed runs.
- **Doesn't own:** which features enter (that's the experiment config) or how results are stored.
- **Depends on:** `features`, `splits`, `metrics`.
- **Done-check:** same seed → identical predictions; log-RV round-trips within tolerance; early
  stopping triggers on a synthetic overfit; loss falls back to MSE-of-log-RV cleanly if QLIKE
  training misbehaves.

### `io / artifacts` — `src/io.py` + `results/` schema
- **Responsibility:** the save/load contract for every run.
- **Owns:** the `config.json` / `preds.parquet` / `metrics.json` / checkpoint format and the
  `results/<experiment>/<run_id>/` layout.
- **Doesn't own:** what goes *into* the metrics (that's `metrics`).
- **Depends on:** nothing (schema); used by all.
- **Done-check:** save→load round-trips to identical objects; the results schema validates
  (all required keys present); `metrics.json`/`preds.parquet` are committed (they *are* the result),
  best weights go to Google Drive.

---

## 7. Cross-cutting invariants & principles  *(load-bearing)*

Inherited by **every** module and experiment. Stated as imperatives. The **bold** ones are the
ones whose violation silently invalidates the project.

**Temporal integrity (leakage control)**
- **NEVER shuffle the time series. All splits are chronological.**
- **ALWAYS fit scalers/transforms on training data only, then apply forward.**
- **EVERY feature value at day *t* uses only information known at the close of day *t*, and never
  revised afterward.** Test: perturb the data *after* day *t*; the feature at *t* must not change.
- ALWAYS keep the embargo gap between splits so multi-day lookback windows never straddle a boundary.
- **The test region is touched EXACTLY ONCE.** It never influences any hyperparameter, feature, or
  window choice.

**Consistency (shared library)**
- Every piece of logic identical across experiments lives in `src/`; **NEVER copy-paste it into a
  notebook.**
- There is **ONE** walk-forward splitter (`src/splits.py`) and **ONE** QLIKE/RMSE definition
  (`src/metrics.py`). Every experiment imports them.
- **Frozen controls — architecture family, tuned HPs, window length, retrain cadence — live in
  `configs/default.yaml` and do NOT change between RV-only and RV+aux.** The only variable across
  those two experiments is the input feature set; that is what makes "where the edge lives"
  interpretable. Any aux gain under frozen HPs is reported as a *conservative lower bound*.

**Data**
- **Data is a frozen, committed, checksummed snapshot. NO module pulls live data at experiment time.**
- Use **ONE documented adjustment basis, consistently**: adjusted prices for return features; the
  GK range target is invariant to proportional back-adjustment, so it is unaffected. **NEVER mix an
  adjusted close with raw OHLC inside one range estimator.** This adjustment-invariance holds **only**
  for open-to-close estimators (GK/PK/RS); the Stage-3 overnight-inclusive target uses the close-to-open
  return, which crosses the dividend-adjustment boundary, so it is **not** adjustment-invariant and
  **requires the adjusted basis** — otherwise ex-dividend days register spurious overnight variance.

**Numerical / scoring**
- **ALWAYS score QLIKE in variance space.** RV is a *variance*; never silently mix volatility and
  variance. (QLIKE's proxy-robustness assumes a *conditionally unbiased* proxy; the GK target omits
  the overnight return and is biased — see §8 — so report RMSE alongside and never lean on QLIKE alone.)
- ALWAYS model **log-RV** internally (it's near-lognormal), then exponentiate to level space for scoring.
- ALWAYS apply a small positive floor to predictions before QLIKE so the log/ratio can't blow up.

**Stochasticity**
- ALWAYS seed Python, NumPy, and the framework; document residual CUDA non-determinism.
- **NEVER report a single-seed result as an edge.** Report mean ± std over multiple seeds
  (~5 developing, ≥10 headline).

**Reproducibility & honesty**
- Pin `requirements.txt`; the pipeline must run end-to-end from a clean clone + the frozen snapshot.
- Commit the small artifacts (`config`/`preds`/`metrics`); they are the deliverable.
- **Pre-register the "where the edge lives" cuts before looking at test results, and report
  wherever the number lands.** This is the structural defense against manufacturing a victory.

---

## 8. Key decisions (with rationale) & open questions

**Decided**
- **Target named honestly: a daily range-based variance proxy (GK), not canonical RV.** We're
  constrained to daily OHLC, so we use a range estimator; HAR's benchmark-beating reputation is for
  *high-frequency* RV, so we **test** whether that dominance transfers rather than assuming it. This
  reframes the contribution from "beat HAR" to "does HAR's dominance transfer, and can a deep model
  beat it here" — a sharper, more defensible question, and the one a quant reader will probe first.
- **v1 target = Garman–Klass (open-to-close); the committed Stage-3 upgrade is an overnight-inclusive
  target (RS + overnight², or windowed Yang–Zhang), not a blend.** GK keeps the grade cut simple and
  is a one-function definition in `data.py`. The substantive upgrade is *overnight inclusion* — it
  makes the target ≈ total daily variance and repairs the QLIKE-unbiasedness caveat — and it runs off
  the same daily OHLC snapshot (no new data). A PK+GK+RS blend only averages away estimator-choice
  arbitrariness (all three are open-to-close), so it is demoted to an optional robustness footnote.
- **Shared `src/` library + thin notebooks.** Only structure that guarantees identical metrics
  across experiments.
- **Data: Stooq (SPY OHLCV) + CBOE (VIX OHLC), snapshot-and-freeze, adjusted basis for returns.**
  Free live scrapers silently revise/re-adjust history, which breaks reproducibility at the root;
  freezing a checksummed snapshot makes reproducibility executable and turns source choice into a
  one-time quality decision. Stooq is the cleaner programmatic daily source; CBOE is the official
  VIX source. FRED (`VIXCLS`, yields, dollar) is the portfolio-stage source for cross-asset features.
- **Split: fixed three-way chronological** — train 2005→2017 (includes 2008), val 2018→2019,
  test 2020→present (COVID, 2022 bear, calm) — with an embargo gap and monthly walk-forward retrain
  inside the test region. Nested walk-forward is the future-work upgrade. Tune on train/val only.
- **Hyperparameters frozen across experiments** (controlled experiment, not a leaderboard chase).
- **Multi-seed**, mean ± std (~5 dev, ≥10 headline).
- **Architecture: a small LSTM, frozen family** (tiny dataset, 3-parameter opponent → overfitting,
  not novelty, is the enemy). GRU/TCN noted as defensible equivalents, never swept.
- **Tracking: lightweight structured artifacts** (`config`/`preds`/`metrics` per run); W&B is the
  upgrade path.
- **Targets:** h=1 (next-day) spine + h=5 (next-week, = *average* daily RV over the window),
  forecast **directly** (not iterated). Two horizons so the "edge by horizon" cut means something.
- **Scoring:** QLIKE primary (penalizes under-forecasting risk), RMSE secondary. Caveat carried
  openly: QLIKE's robustness to proxy noise (Patton 2011) assumes a *conditionally unbiased* proxy,
  and the GK target omits the overnight return, so it is biased low for total daily variance. Treat
  QLIKE as valid for ranking on the open-to-close range object GK actually estimates, report RMSE
  alongside, and state the bias rather than claiming full proxy-robustness.
- **GARCH(1,1) is context, not competitor** — fit on returns; RV models are known to beat it.

**Open (resolve before/at coding; these are leaf details, not root truths)**
- Exact split boundaries in **trading days** vs calendar dates.
- Rolling-window length for the decay study (default ~2 years).
- LSTM input sequence length (candidate for the Optuna search).
- Module carve: `models`+`train` as one module or two; `baselines` as one or split HAR/GARCH.

---

## 9. Glossary — name things once

Use these exact terms from manifesto to code.

- **Range variance (the v1 target; "RV" in code and experiment names):** daily variance estimated
  from OHLC via **Garman–Klass** — a *biased proxy* for latent daily volatility (it omits the
  overnight return). A **variance**, not its square root. Distinct from the canonical object below;
  "RV" survives as a code shorthand only because it's defined here.
- **Realized variance (canonical RV):** the standard *high-frequency* target — the sum of squared
  *intraday* returns. The object Corsi's HAR was validated on and the source of "HAR is hard to
  beat." **Not** the v1 target; it's the Stage-3 upgrade.
- **Garman–Klass / Parkinson / Rogers–Satchell:** daily range-based *variance* estimators from OHLC
  — roughly 5–7× more efficient than close-to-close squared returns, but a different family from
  high-frequency RV. GK estimates *open-to-close* variance (biased for assets with overnight gaps).
  Within-day ratio estimators → invariant to proportional price adjustment.
- **Overnight-inclusive target (Stage 3):** a daily target that adds the close-to-open (overnight)
  variance to an open-to-close estimator — per-day *RS + overnight²*, or windowed **Yang–Zhang**.
  Approximates *total* daily variance; requires the adjusted price basis; not adjustment-invariant.
- **HAR:** Heterogeneous AutoRegressive model; OLS on daily/weekly/monthly lagged volatility.
  **The bar** — its dominance is established for high-frequency RV and is what we *test*, not assume,
  on the range target.
- **GARCH(1,1):** conditional-variance-of-returns model; **context baseline only.**
- **Forecastability floor / ceiling:** the irreducible loss set by the target proxy's own
  measurement noise — no model can predict the noise. A property of the *target*, not any model;
  estimating it (Stage 3) is what lets a tie with HAR be interpreted.
- **Walk-forward:** chronological evaluation that retrains forward through time; never shuffled.
- **Embargo:** a gap between splits so lookback windows don't straddle the boundary.
- **Leak-safe / leakage:** a feature is leak-safe if its value is fully known at the close of day
  *t* and never revised. Leakage = any information from after *t* reaching the day-*t* forecast.
- **Frozen control:** a setting pinned in `configs/default.yaml`, identical across experiments.
- **Snapshot:** the frozen, committed, checksummed copy of the raw data the pipeline reads.
- **Adjustment basis:** the single documented choice of adjusted vs raw prices, applied consistently.
- **log-RV:** the modeled quantity (the range-variance target is near-lognormal); exponentiated back
  for scoring.
- **Variance space:** the units (variance, not volatility) in which QLIKE is always computed.
- **QLIKE:** quasi-likelihood loss; primary metric; penalizes under-forecasting risk. Robust to
  proxy noise *only when the proxy is conditionally unbiased* (Patton 2011) — a caveat here, since
  the GK target is biased by the overnight omission; report RMSE alongside.
- **DM (Diebold–Mariano):** significance test on a loss differential. **MCS (Model Confidence Set):**
  the multi-model upgrade.
- **"Where the edge lives":** the pre-registered decomposition — by regime, by horizon, by source
  (aux features vs architecture).
- **Realized semivariance:** downside/upside RV split; needs intraday data (portfolio stage).
- **Vol-targeting:** inverse-vol position sizing; the mechanism of the deferred economic-significance
  follow-on.

---

## 10. Global tech stack

Named tools only; **versions are pinned in `requirements.txt`, matched to Colab's current
environment at setup** (do not hardcode drifting versions here).

- **Compute:** Google Colab (school account, free tier). Daily RV is a few thousand rows; the LSTM
  trains fast on CPU — GPU is needed, if at all, only to speed up the Optuna search.
- **Modeling:** PyTorch (LSTM); Optuna (hyperparameter search).
- **Classical:** `statsmodels` (HAR / OLS), `arch` (GARCH).
- **Data:** Stooq (SPY OHLCV), CBOE CSV (VIX); FRED (portfolio-stage cross-asset). `pandas`,
  `numpy`, `pyarrow` (parquet).
- **Persistence:** git for code + small artifacts (`config`/`preds`/`metrics`); Google Drive
  (mounted in Colab) for best-model weights. Skip git-LFS.
- **Reproducibility:** pinned `requirements.txt`; seeded runs; committed data snapshot + hash.

---

## 11. What this manifesto implies (next artifacts in the cascade)

This manifesto implies roughly **seven `SPEC.md`s**, one per engine module in §6. Write them in
dependency order; the two that everything else inherits correctness from come first:

1. **`splits/SPEC.md`** and **`metrics/SPEC.md`** — the canonical splitter and the scoring/DM
   definitions, with their full executable test lists (overlap/embargo/determinism; QLIKE fixture +
   variance-space + non-positive-floor). Everything downstream trusts these.
2. **`data/SPEC.md`** — snapshot contract (Stooq + CBOE, hash, adjustment basis), GK construction,
   trading-day index.
3. **`features/SPEC.md`** — the leak contract + Tier 1/2/3 menu, with the future-perturbation
   leakage test specified concretely.
4. **`baselines/SPEC.md`**, then **`model+harness/SPEC.md`**, then **`io/SPEC.md`**.

Each `SPEC.md` then gets a `design.md` in the named stack, and each `src/` folder gets a lean
`CLAUDE.md` pointing at its spec and restating only that folder's hard rules.
