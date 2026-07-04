# Volatility Forecasting — Manifesto

> **Layer 1 (root).** This is the single source of truth. The *why*, the boundaries, the
> invariants, the module map, and the vocabulary live here. Everything below — the per-module
> `SPEC.md`s, their `design.md`s, the `CLAUDE.md`s — inherits from this and must not contradict it.
> Implementation detail that will drift does **not** belong here; it belongs one layer down.

---

## 0. Spine — read this first

**One sentence:** *Field a pre-specified set of classical volatility models plus one small LSTM on
daily range-based SPY volatility targets, and measure — with inference that is valid for those
targets — whether HAR is still the bar to beat and whether the net adds signal beyond it.*

The capability on display is honest measurement under adversarial self-scrutiny. A single model
family can only ever establish a **lower bound** on extractable signal: beat HAR by 2% and you've
shown the signal-beyond-HAR is *at least* 2%; tie HAR and you've learned *this model* found nothing
— not that a better one couldn't. The true **ceiling** (the target proxy's own noise / the
forecastability floor) is a property of the target, not of any one model; computing it is a
deliberate Stage-3 analysis (§5), not something the base pipeline delivers. **So the base project
claims a floor, never a ceiling.** A rigorously validated *tie* with HAR is a real result; a
suspicious *win* is worth less than an honest tie.

**Claims discipline — what this project can and cannot assert.**
- **CAN:** the relative accuracy of the fielded models *on these targets*, with per-comparison
  significance; a floor on signal-beyond-HAR; whether HAR is the strongest *classical* model in the
  pre-specified field on this data.
- **CANNOT:** that "HAR's dominance transfers" in general — dominance is a claim about HAR versus
  the whole competitor field of the high-frequency literature, which this project does not
  replicate. If HAR beats the classical field here by margins resembling the high-frequency
  literature, that is reported as *descriptive evidence consistent with* transfer, never as a test
  of it. Also cannot: any ceiling claim (Stage 3), and any claim about **total** daily volatility
  from the open-to-close target alone (§7, scoring).

**The single design rule everything inherits from — Shared library, thin notebooks.**
Every piece of logic that must be *identical* across experiments lives in `src/` as importable
Python. Notebooks configure one experiment, call the library, and save artifacts. This makes
"consistent metrics across experiments" true *by construction*: fix a bug once, every experiment
inherits the fix. If two experiments could ever disagree on target construction, the splitter, the
metrics, the baselines, the model harness, or the artifact format, the comparison is void.

---

## 1. Purpose & the problem

Volatility is highly forecastable, and the **HAR** model — three OLS terms on daily, weekly, and
monthly lagged volatility — is famously hard to beat. But that reputation was earned on a specific
target: **high-frequency realized variance** (the sum of squared *intraday* returns), the object
Corsi's HAR was built on. This project is constrained to **daily OHLC data**, so its targets are
**daily range-based variance proxies** — a distinct, older estimator family, and *proxies* for
latent daily volatility rather than the canonical high-frequency RV.

An earlier draft asked "does HAR's dominance *transfer* to this target?" — but that question is
**untestable by this design**: dominance is a claim about HAR versus the full competitor field of
the high-frequency literature (GARCH variants, EWMA, ARFIMA, MIDAS, …), and no single project
re-fields all of it. What *is* testable, and what this project tests:

> On daily SPY range-based volatility targets: (1) **Is HAR the strongest classical model** against
> a pre-specified field — persistence, EWMA, AR(1) on log-RV, GARCH(1,1) — i.e., is it still the
> bar to beat here? (2) **Can a small LSTM add signal beyond HAR?** (3) **Where does any edge
> live?**

Question (1) gives "HAR is hard to beat" empirical content on this target instead of importing it
as folklore; question (2) is the deep-learning contribution; question (3) is the pre-registered
decomposition. The point is not to win a leaderboard. It is to (a) build a leak-free, reproducible
forecasting pipeline, (b) measure whatever signal exists beyond a strong benchmark, and (c) report
wherever the number lands. The economic-significance question — does a statistical edge convert
into money — is explicitly a **separate follow-on project**, specced but not built here (§4, §8).

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
score the classical field, tune one net leak-safely, compare the net to HAR on identical inputs,
add leak-safe auxiliary features, study window decay, and produce the final two-target comparison.

---

## 3. Project-level success criteria — measurable & executable

Definitions of done that map to a runnable check, not a vibe. Each module's `SPEC.md` owns its
full test list; these are the project-level gates.

| # | Criterion | How it's checked (executable) |
|---|-----------|-------------------------------|
| S1 | **The bar exists — and is contested.** The full classical field (persistence, EWMA, AR(1) on log-RV, HAR, GARCH(1,1)) is scored on the frozen splits, both targets. | `99_evaluation` loads their `metrics.json`; every classical model's QLIKE is present and finite on both targets. |
| S2 | **The pipeline is reproducible from nothing.** | Clean clone + frozen data snapshot + pinned `requirements.txt` runs end-to-end with no manual steps; snapshot SHA-256 matches the committed hash; the snapshot manifest pins the data end date. |
| S3 | **No lookahead, provably.** | The feature leakage test (§7) passes: perturbing data after day *t* leaves every feature value at day *t* unchanged. Scaler fit-indices ⊆ train indices. |
| S4 | **Results are pure replay.** | `99_evaluation` rebuilds the headline table from saved artifacts, re-running **no** training. |
| S5 | **Significance is valid, not just reported.** | Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction and HAC (Bartlett) variance with lag ≥ h−1, run on the QLIKE loss differential vs HAR, per model, per horizon, per target, using the seed-ensemble mean predictions; all p-values reported together (no cherry-picking). |
| S6 | **Edges survive seed noise.** | Headline numbers are mean ± std over ≥10 seeds. A claimed edge requires: DM p < 0.05 on the ensemble-mean predictions **and** a same-sign mean loss differential in ≥ 80% of individual seeds. |
| S7 | **The edge decomposition is pre-committed and reported.** | The pre-registered "where the edge lives" cuts (regime, horizon, source, target) are computed and reported *regardless* of whether the net wins, ties, or loses. |
| S8 | **The proxies are checked, not assumed.** | Calibration diagnostics pass: mean(GK)/mean(r²_oc) and mean(TV proxy)/mean(r²_cc) on the train span fall in a pre-set tolerance band; the QLIKE prediction floor binds on < 0.1% of test predictions (bind-rate logged per run). |

**Done-check for this manifesto:** a fresh reader can list the modules they'd build and the rules
every module must obey, without guessing.

---

## 4. Scope boundaries & non-goals

Stated as hard boundaries so specs don't quietly expand them.

- **ONE asset (SPY) in v1.** Multi-asset (QQQ, a single name, a second period) is robustness
  *stretch*, meaningful only after a single-asset result exists.
- **TWO v1 targets, both from the same daily OHLC snapshot; no new data.**
  - **Primary — total daily variance (TV):** per-day Rogers–Satchell + squared overnight return.
    Approximately conditionally unbiased for *total* daily variance, which is what makes QLIKE
    ranking and DM inference valid for it (§7).
  - **Secondary — open-to-close variance (OC):** per-day Garman–Klass. Approximately conditionally
    unbiased for *open-to-close* variance only; all claims from it are scoped to that object.
  - **High-frequency (5-min) realized variance — the canonical RV, the target HAR's reputation is
    built on — stays a Stage-3 stretch** (it is the one item needing *new* intraday data, and the
    prerequisite for realized semivariance).
- **NO economic-significance backtest in v1.** Vol-targeted position sizing, Sharpe/drawdown
  comparison, transaction costs — deferred to its own follow-on project (spec recorded, build later).
- **NO cadence or window-length *sweep*.** Retrain cadence, rolling length, and architecture are
  **frozen controls**, not experiments. Letting them become experiments is the grid that kills
  the timeline.
- **NO architecture search** beyond tuning sizes *within* the frozen LSTM family.
- **Tuning happens ONCE, on the primary (TV) target.** The frozen HP set is reused unchanged on
  OC; scoring both targets is cheap, re-tuning per target is scope creep.
- **NOT a live or production system.** This is an offline research pipeline over a frozen data
  snapshot; nothing pulls live data at experiment time.

---

## 5. Stages — grade-ready, then portfolio

Build in this order. **Never start a later stage until the previous one is locked and saved.**
The way this project dies is falling in love with a fancier model and never finishing the boring,
decisive validation.

- **Stage 0 — Harness (must work before any net).** Data snapshot (end date pinned in the
  manifest) + both target constructions (TV, OC) + calibration diagnostics (S8) + frozen splits +
  metrics incl. the DM-HLN test + the full classical field (persistence, EWMA, AR(1) on log-RV,
  HAR, GARCH), scored on the frozen splits, both targets. This alone answers research question (1).
  *Everything after is upside.*
- **Stage 1 — Tuned net + controlled comparison.** Leak-safe Optuna search on the TV target → one
  frozen HP set; RV-only net vs HAR on identical inputs.
- **Stage 2 — Real results (this is the grade-ready cut).** RV+aux experiment, decay curve, final
  two-target evaluation notebook (frozen HPs re-scored on OC). **Lock and save before any stretch.**
- **Stage 3 — Portfolio extensions.** A genuine **forecastability-floor analysis** — estimate each
  target proxy's own measurement noise / irreducible loss, so a *ceiling* on predictability can be
  stated honestly (this is what lets a tie with HAR mean something instead of nothing).
  High-frequency 5-min RV (the canonical target, the one item here needing *new* intraday data) →
  realized semivariance. Multi-asset robustness; data refresh + second-source cross-check. A
  PK+GK+RS **blend** stays an *optional* robustness footnote only — all three are open-to-close, so
  it addresses estimator-choice arbitrariness, **not** the overnight gap (which TV already closes).
  Model Confidence Set as the multiplicity-aware upgrade to pairwise DM. Then the
  economic-significance follow-on as a separate project.

Stage 2 is the deliverable a grader sees. Stage 3 is what makes it a portfolio piece.

---

## 6. Capability / module map  *(load-bearing)*

The **modules** are the durable `src/` engine — the things an agent builds and tests. The
**notebooks are not modules**; they are thin experiments that configure and call the engine, and
map to the use cases in §2. Carved so each module is one focused agent session. Dependency
direction points *downward* (a module may use those listed under "Depends on," nothing else).

> Open carve question (see §8): whether `models`+`train` are one module or two, and whether
> `baselines` is one module or split classical-field/GARCH. Default: one each.

### `data` — `src/data.py`
- **Responsibility:** turn the frozen price/VIX snapshot into the canonical target series.
- **Owns:** snapshot loading; both target constructions (TV = per-day RS + overnight²; OC = GK);
  the calibration diagnostics (S8); the documented adjustment basis; the trading-day calendar that
  indexes everything downstream.
- **Doesn't own:** features, splitting, any model.
- **Depends on:** the committed snapshot only. (No network at experiment time.)
- **Done-check:** snapshot checksum matches committed hash and the manifest's pinned end date;
  both targets non-negative with no NaN after warmup; GK and RS+overnight² each match a
  hand-computed value on a small OHLC fixture; OHLC internal consistency holds on every row
  (low ≤ min(open, close) ≤ max(open, close) ≤ high) on one basis; calibration ratios (S8) inside
  the tolerance band.

### `features` — `src/features.py`
- **Responsibility:** build the leak-safe feature matrix (target history + tiered aux features).
- **Owns:** the feature-level leak contract; the Tier 1/2/3 aux menu — enumerated and
  pre-registered in `specs/features/SPEC.md` §2 (Tier 1: lagged log-target 0–21 + 5d/22d log
  aggregates; Tier 2: SPY return, |return|, overnight return, volume ratio; Tier 3: log VIX, VIX
  change, implied daily variance; ratified 2026-07-03, frozen before Stage-2 test contact);
  scaler/transform objects (fit on train only, applied forward).
- **Doesn't own:** the *temporal* split (that's `splits`); the targets (that's `data`).
- **Depends on:** `data`.
- **Done-check:** the future-perturbation leakage test (§7) passes; every scaler's fit-indices are
  a subset of the training indices; all feature columns finite.

### `splits` — `src/splits.py`
- **Responsibility:** **the** canonical walk-forward splitter — single source of truth.
- **Owns:** the frozen split definition (train/val/test boundaries), the embargo gap, the
  monthly walk-forward retrain folds inside the test region, the fixed-length rolling retrain
  window (a frozen control — see §8 for why rolling, not expanding).
- **Doesn't own:** anything model- or feature-specific; it operates on a date index alone.
- **Depends on:** nothing.
- **Done-check:** zero overlap between any train fold and its test fold; embargo ≥ configured gap;
  every fold chronological (test strictly after train); every retrain window exactly the configured
  fixed length; deterministic; union of test folds covers the test region with no gaps.

### `metrics` — `src/metrics.py`
- **Responsibility:** scoring and significance.
- **Owns:** `qlike()` (with the pre-specified prediction floor and bind-rate logging), `rmse()`,
  the Diebold–Mariano test — HLN small-sample correction, HAC (Bartlett) variance with
  lag ≥ h−1 (larger if automatic Newey–West selection says so), two-sided, run on a supplied
  prediction sequence; the variance-space discipline.
- **Doesn't own:** any data or model; *which* prediction sequence DM receives (the harness supplies
  the seed-ensemble mean per §7).
- **Depends on:** nothing.
- **Done-check:** `qlike`/`rmse` match a hand-computed fixture; `qlike` floors non-positive
  predictions and logs the bind-rate; DM matches a reference implementation on a fixture and its
  HAC lag responds to h; a guard catches volatility-vs-variance unit mismatch.

### `baselines` — `src/baselines.py`
- **Responsibility:** the classical field — the models that give "HAR is hard to beat" content.
- **Owns:** **persistence** (today's target value; for h=5, today's trailing 5-day mean), **EWMA**
  of the lagged target (decay fit on train; RiskMetrics λ=0.94 as the fixed fallback), **AR(1) on
  log-target**, **HAR** (the presumptive bar), and **GARCH(1,1)** (fit on close-to-close returns —
  its forecast object is *total* return variance, so it is scored against TV; if shown against OC
  the object mismatch is stated in the table note).
- **Doesn't own:** the net.
- **Depends on:** `data`, `splits`, `metrics`.
- **Done-check:** every classical model produces one variance-space forecast per test date per
  target it's scored on; HAR coefficients sane; GARCH's variance forecast positive and in
  **variance space** before QLIKE touches it; persistence and EWMA match hand-computed fixtures.

### `model + harness` — `src/models.py` + `src/train.py`
- **Responsibility:** the frozen small-LSTM family and a seeded, leak-safe train/eval loop.
- **Owns:** the LSTM definition (family frozen; Optuna tunes sizes within it); seeding;
  early stopping; log-target modeling with exponentiation back to level space; QLIKE-with-floor
  training loss with an MSE-of-log-target fallback; multi-seed runs and the **seed-ensemble mean
  prediction sequence** (the object DM tests, per §7).
- **Doesn't own:** which features enter (that's the experiment config) or how results are stored.
- **Depends on:** `features`, `splits`, `metrics`.
- **Done-check:** same seed → identical predictions; log-target round-trips within tolerance; early
  stopping triggers on a synthetic overfit; loss falls back to MSE-of-log cleanly if QLIKE
  training misbehaves; ensemble-mean predictions equal the mean of per-seed predictions on a fixture.

### `io / artifacts` — `src/io.py` + `results/` schema
- **Responsibility:** the save/load contract for every run.
- **Owns:** the `config.json` / `preds.parquet` / `metrics.json` / checkpoint format and the
  `results/<experiment>/<run_id>/` layout; per-seed predictions **and** the ensemble mean are both
  saved.
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
- **The test region is touched EXACTLY ONCE per pre-registered experiment**, in one final scoring
  pass. Multi-seed runs and the pre-registered cuts are *inside* that one protocol; the test region
  never influences any hyperparameter, feature, or window choice.

**Consistency (shared library)**
- Every piece of logic identical across experiments lives in `src/`; **NEVER copy-paste it into a
  notebook.**
- There is **ONE** walk-forward splitter (`src/splits.py`) and **ONE** QLIKE/RMSE/DM definition
  (`src/metrics.py`). Every experiment imports them.
- **Frozen controls — architecture family, tuned HPs, window length, retrain cadence — live in
  `configs/default.yaml` and do NOT change between RV-only, RV+aux, or across targets.** The only
  variable across the two feature experiments is the input feature set; the only variable across
  targets is the target. That is what makes "where the edge lives" interpretable. Any aux gain
  under frozen HPs is reported as a *conservative lower bound*.

**Data**
- **Data is a frozen, committed, checksummed snapshot with a pinned end date in its manifest. NO
  module pulls live data at experiment time.** "Test = 2020 → snapshot end," never "→ present."
- Use **ONE documented adjustment basis, consistently**: adjusted prices for return features.
  Open-to-close range estimators (GK/PK/RS) are invariant to proportional back-adjustment, so the
  OC target is unaffected either way. **NEVER mix an adjusted close with raw OHLC inside one range
  estimator.** The TV target's overnight return crosses the dividend-adjustment boundary, so it is
  **not** adjustment-invariant and **requires the adjusted basis** — otherwise ex-dividend days
  register spurious overnight variance. This is a v1 requirement, not a stretch concern.

**Numerical / scoring**
- **ALWAYS score QLIKE in variance space.** Every target is a *variance*; never silently mix
  volatility and variance.
- **QLIKE inference is only as valid as the proxy is conditionally unbiased (Patton 2011) — so
  every claim is scoped to the object its proxy is unbiased for.** TV (RS + overnight²) is
  approximately conditionally unbiased for *total* daily variance → QLIKE ranking and DM inference
  on TV speak to total daily volatility. OC (GK) is approximately conditionally unbiased for
  *open-to-close* variance only → OC claims are about the trading session, never the whole day.
  The S8 calibration diagnostics are the empirical check on both. Report RMSE alongside QLIKE.
- ALWAYS model **log-target** internally (near-lognormal), then exponentiate to level space for scoring.
- ALWAYS apply the **pre-specified** positive floor to predictions before QLIKE (fixed at data
  freeze, e.g. the 1st percentile of the train-span target; recorded in `configs/default.yaml`),
  and log the bind-rate every run. A floor that binds materially (S8 threshold) invalidates the
  comparison and must be flagged, not absorbed.
- For **h=5** the daily forecasts of a 5-day average target **overlap**, so their loss
  differentials are serially correlated by construction: DM variance MUST be HAC with lag ≥ h−1.
  This is why DM lives in `metrics` and is never hand-rolled in a notebook.

**Stochasticity**
- ALWAYS seed Python, NumPy, and the framework; document residual CUDA non-determinism.
- **NEVER report a single-seed result as an edge.** Report mean ± std over multiple seeds
  (~5 developing, ≥10 headline).
- **DM tests the seed-ensemble mean prediction sequence** (pre-specified here, not chosen after
  seeing results); per-seed loss differentials are reported as the robustness check (S6's
  same-sign-in-≥80%-of-seeds requirement).

**Reproducibility & honesty**
- Pin `requirements.txt`; the pipeline must run end-to-end from a clean clone + the frozen snapshot.
- Commit the small artifacts (`config`/`preds`/`metrics`); they are the deliverable.
- **Pre-register the "where the edge lives" cuts before looking at test results, and report
  wherever the number lands.** This is the structural defense against manufacturing a victory.
- **ALL DM p-values are reported together, per model, per horizon, per target — never a curated
  subset.** With multiple pairwise comparisons, individual p-values overstate the family-wise
  evidence; say so in the report, and note MCS (Stage 3) as the multiplicity-aware upgrade.
- The DM loss-differential stationarity assumption is strained by a regime-heavy test span
  (COVID, 2022): carry this caveat openly; the pre-registered regime cut (S7) is the mitigation,
  not a rug.

---

## 8. Key decisions (with rationale) & open questions

**Decided**
- **The research question is scoped to what the design can test.** "Does HAR's dominance transfer?"
  was retired as untestable (dominance = HAR vs the high-frequency literature's whole field).
  Replaced by: is HAR the strongest classical model here, against a pre-specified field, and does a
  small LSTM add signal beyond it? Transfer talk survives only as clearly-labeled interpretation.
- **A classical field, not a lone benchmark: persistence, EWMA, AR(1) on log-target, HAR,
  GARCH(1,1).** All are closed-form or OLS — near-zero cost — and they are what makes "HAR is hard
  to beat" a finding on this target instead of imported folklore. GARCH stays context (fit on
  close-to-close returns; its object is total variance, so it is scored against TV). AR(1) and
  HAR are both fit in log space (see glossary **HAR**; ratified 2026-07-03).
- **Two v1 targets, both from the same daily OHLC snapshot.** Primary **TV** = per-day
  Rogers–Satchell + squared overnight return: approximately conditionally unbiased for total daily
  variance, which is precisely what Patton-validity of QLIKE ranking and the DM test requires —
  this repair is why TV was promoted from Stage 3 into v1 (it needs no new data). Secondary
  **OC** = Garman–Klass: the cleanest one-function range estimator, kept because it connects to the
  range-estimator literature and the TV-vs-OC contrast ("does the story change once overnight
  variance is in the target?") is itself a pre-registered cut. A PK+GK+RS blend only averages away
  estimator-choice arbitrariness (all open-to-close), so it is demoted to an optional Stage-3
  robustness footnote.
- **Inference: Diebold–Mariano with the HLN small-sample correction, HAC (Bartlett) lag ≥ h−1,
  two-sided, on the seed-ensemble mean predictions.** The overlapping h=5 target forces the HAC
  lag; HLN because the monthly-retrain test span has limited effective length; ensemble-mean as the
  tested sequence is pre-specified to close the garden-of-forking-paths on seeds. MCS is the
  Stage-3 multiplicity upgrade.
- **Monthly retrain uses a fixed-length rolling window (length = the initial train span), not an
  expanding one.** A finite-memory estimation scheme is what keeps comparisons of re-estimated
  forecasts within the Giacomini–White framework's conditions, so out-of-sample tests remain
  well-posed as models are re-fit through the test region. Rolling length is a frozen control.
- **Shared `src/` library + thin notebooks.** Only structure that guarantees identical metrics
  across experiments.
- **Data: Stooq (SPY OHLCV) + CBOE (VIX OHLC), snapshot-and-freeze, adjusted basis for returns and
  the overnight leg.** Free live scrapers silently revise/re-adjust history, which breaks
  reproducibility at the root; freezing a checksummed snapshot with a **pinned end date** makes
  reproducibility executable and turns source choice into a one-time quality decision. The
  snapshot's OHLC must be verified internally consistent on one basis (data done-check) — a mixed
  basis silently corrupts both range estimators and the overnight return. Stooq is the cleaner
  daily source but **login-gates CSV export** (discovered at freeze, 2026-07): the snapshot's SPY
  leg is a **manual export from a logged-in session**, canonicalized and checksummed by
  `scripts/freeze_snapshot.py` with a provenance note in the manifest (amendment ratified
  2026-07-03; the freeze-and-checksum contract is unchanged). CBOE is the official VIX source and
  remains programmatic. FRED (`VIXCLS`, yields, dollar) is the portfolio-stage source for
  cross-asset features.
- **Split: fixed three-way chronological** — train 2005→2017 (includes 2008), val 2018→2019,
  test 2020→snapshot end (COVID, 2022 bear, calm) — with an embargo gap and monthly walk-forward
  retrain inside the test region. Nested walk-forward is the future-work upgrade. Tune on
  train/val only.
- **Hyperparameters frozen across experiments and targets; tuned once, on TV** (controlled
  experiment, not a leaderboard chase).
- **Multi-seed**, mean ± std (~5 dev, ≥10 headline); edge claims per S6.
- **Architecture: a small LSTM, frozen family** (tiny dataset, 3-parameter opponent → overfitting,
  not novelty, is the enemy). GRU/TCN noted as defensible equivalents, never swept.
- **Tracking: lightweight structured artifacts** (`config`/`preds`/`metrics` per run); W&B is the
  upgrade path.
- **Horizons:** h=1 (next-day) spine + h=5 (next-week, = *average* daily target over the window),
  forecast **directly** (not iterated). Two horizons so the "edge by horizon" cut means something;
  h=5's overlap is handled in DM (§7), never ignored.
- **Scoring:** QLIKE primary (penalizes under-forecasting risk), RMSE secondary — with QLIKE
  validity *scoped per target* (§7) rather than caveated after the fact.

**Open (resolve before/at coding; these are leaf details, not root truths)**
- Exact split boundaries in **trading days** vs calendar dates; exact snapshot end date (pinned at
  freeze).
- EWMA decay: fit on train vs fixed RiskMetrics λ=0.94 (default: fit, with 0.94 as fallback).
- Rolling-window length for the decay study (default ~2 years).
- LSTM input sequence length (candidate for the Optuna search).
- S8 tolerance band for the calibration ratios (default: [0.8, 1.25]).
- Module carve: `models`+`train` as one module or two; `baselines` as one or split.

---

## 9. Glossary — name things once

Use these exact terms from manifesto to code.

- **TV (total-variance target; primary; `rv_tv` in code):** per-day **Rogers–Satchell +
  squared overnight (close-to-open) return**. Approximately conditionally unbiased for *total*
  daily variance — the property QLIKE/DM validity rests on. Requires the adjusted price basis;
  **not** adjustment-invariant. A **variance**, not its square root.
- **OC (open-to-close target; secondary; `rv_oc` in code):** per-day **Garman–Klass**.
  Approximately conditionally unbiased for *open-to-close* (trading-session) variance only; claims
  from it are scoped to that object. Adjustment-invariant. A **variance**.
- **"RV":** generic code shorthand for either target series; survives only because both targets
  are defined here. Never means canonical high-frequency RV in this repo.
- **Realized variance (canonical RV):** the standard *high-frequency* target — the sum of squared
  *intraday* returns. The object Corsi's HAR was validated on and the source of "HAR is hard to
  beat." **Not** a v1 target; it's the Stage-3 upgrade.
- **Garman–Klass / Parkinson / Rogers–Satchell:** daily range-based *variance* estimators from OHLC
  — roughly 5–7× more efficient than close-to-close squared returns, but a different family from
  high-frequency RV. All three estimate *open-to-close* variance. Within-day ratio estimators →
  invariant to proportional price adjustment.
- **Classical field:** the pre-specified benchmark set — persistence, EWMA, AR(1) on log-target,
  HAR, GARCH(1,1). What gives "hard to beat" empirical content here.
- **Persistence:** forecast = today's target value (h=5: today's trailing 5-day mean). The naive
  bar under the bar.
- **EWMA:** exponentially weighted moving average of the lagged target; decay fit on train
  (RiskMetrics λ=0.94 fallback).
- **HAR:** Heterogeneous AutoRegressive model; OLS on daily/weekly/monthly lagged volatility,
  **fit on the log-target (log-HAR)** with the lognormal half-variance back-transform, per the §7
  log-target invariant (pre-registered in `specs/baselines/SPEC.md`; ratified 2026-07-03).
  Corsi's original level-space HAR is *not* fielded — it may appear only as a clearly-labeled
  descriptive robustness footnote in the report, never as a competitor in the pre-specified
  field. **The presumptive bar** — whether it actually tops the classical field here is research
  question (1), tested, not assumed.
- **GARCH(1,1):** conditional-variance-of-returns model; **context baseline only.** Fit on
  close-to-close returns; forecasts *total* return variance, so scored against TV.
- **Forecastability floor / ceiling:** the irreducible loss set by the target proxy's own
  measurement noise — no model can predict the noise. A property of the *target*, not any model;
  estimating it (Stage 3) is what lets a tie with HAR be interpreted.
- **Conditional unbiasedness (of a proxy):** E[proxy | information at t] equals the true
  conditional variance. The Patton (2011) condition under which QLIKE rankings against the proxy
  match rankings against the truth — the license for every QLIKE/DM claim in this project, checked
  empirically via the S8 calibration diagnostics.
- **Walk-forward:** chronological evaluation that retrains forward through time; never shuffled.
  Retraining uses a **fixed-length rolling window** (frozen control).
- **Embargo:** a gap between splits so lookback windows don't straddle the boundary.
- **Leak-safe / leakage:** a feature is leak-safe if its value is fully known at the close of day
  *t* and never revised. Leakage = any information from after *t* reaching the day-*t* forecast.
- **Frozen control:** a setting pinned in `configs/default.yaml`, identical across experiments.
- **Snapshot:** the frozen, committed, checksummed copy of the raw data the pipeline reads, with a
  pinned end date in its manifest.
- **Adjustment basis:** the single documented choice of adjusted vs raw prices, applied
  consistently; the TV overnight leg requires adjusted.
- **log-target:** the modeled quantity (both targets are near-lognormal); exponentiated back for
  scoring.
- **Variance space:** the units (variance, not volatility) in which QLIKE is always computed.
- **QLIKE:** quasi-likelihood loss; primary metric; penalizes under-forecasting risk. Ranking-robust
  to proxy noise *only when the proxy is conditionally unbiased* (Patton 2011) — which is why each
  target's claims are scoped to the object its proxy is unbiased for.
- **DM / DM-HLN (Diebold–Mariano, Harvey–Leybourne–Newbold corrected):** significance test on a
  loss differential; run with HAC (Bartlett) variance, lag ≥ h−1, on the seed-ensemble mean
  predictions. **Giacomini–White:** the conditional-predictive-ability framework whose
  finite-memory condition motivates the rolling retrain window. **MCS (Model Confidence Set):**
  the multi-model, multiplicity-aware upgrade (Stage 3).
- **Seed-ensemble mean:** the per-date mean of predictions across seeds; the pre-specified
  sequence DM tests.
- **"Where the edge lives":** the pre-registered decomposition — by regime, by horizon, by source
  (aux features vs architecture), by target (TV vs OC).
- **Realized semivariance:** downside/upside RV split; needs intraday data (portfolio stage).
- **Vol-targeting:** inverse-vol position sizing; the mechanism of the deferred economic-significance
  follow-on.

---

## 10. Global tech stack

Named tools only; **versions are pinned in `requirements.txt`, matched to Colab's current
environment at setup** (do not hardcode drifting versions here).

- **Compute:** Google Colab (school account, free tier). Daily targets are a few thousand rows;
  the LSTM trains fast on CPU — GPU is needed, if at all, only to speed up the Optuna search.
- **Modeling:** PyTorch (LSTM); Optuna (hyperparameter search).
- **Classical:** `statsmodels` (HAR / AR(1) / OLS), `arch` (GARCH); persistence and EWMA are
  `numpy`/`pandas` one-liners in `baselines`.
- **Data:** Stooq (SPY OHLCV), CBOE CSV (VIX); FRED (portfolio-stage cross-asset). `pandas`,
  `numpy`, `pyarrow` (parquet).
- **Persistence:** git for code + small artifacts (`config`/`preds`/`metrics`); Google Drive
  (mounted in Colab) for best-model weights. Skip git-LFS.
- **Reproducibility:** pinned `requirements.txt`; seeded runs; committed data snapshot + hash +
  pinned end date.

---

## 11. What this manifesto implies (next artifacts in the cascade)

This manifesto implies roughly **seven `SPEC.md`s**, one per engine module in §6. Write them in
dependency order; the two that everything else inherits correctness from come first:

1. **`splits/SPEC.md`** and **`metrics/SPEC.md`** — the canonical splitter (incl. the fixed-length
   rolling retrain window) and the scoring/DM definitions, with their full executable test lists
   (overlap/embargo/determinism/window-length; QLIKE fixture + variance-space + floor + bind-rate;
   DM-HLN fixture + HAC-lag behavior). Everything downstream trusts these.
2. **`data/SPEC.md`** — snapshot contract (Stooq + CBOE, hash, pinned end date, adjustment basis,
   OHLC-consistency checks), both target constructions (TV, OC), calibration diagnostics,
   trading-day index.
3. **`features/SPEC.md`** — the leak contract + Tier 1/2/3 menu, with the future-perturbation
   leakage test specified concretely.
4. **`baselines/SPEC.md`** (the full classical field), then **`model+harness/SPEC.md`** (incl. the
   seed-ensemble mean contract), then **`io/SPEC.md`**.

Each `SPEC.md` then gets a `design.md` in the named stack, and each `src/` folder gets a lean
`CLAUDE.md` pointing at its spec and restating only that folder's hard rules.
