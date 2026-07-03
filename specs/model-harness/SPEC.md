# SPEC — `model + harness` (`src/models.py` + `src/train.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `model + harness` module
> of manifesto §6 as **one module, two files** (§8 open carve resolved to defaults).

## 1. Purpose

The frozen small-LSTM family and a seeded, leak-safe train/eval loop (manifesto §6). This is the
deep-learning contribution (research question (2)): can a small LSTM add signal beyond HAR?
Owns the LSTM definition, seeding, early stopping, log-target modeling, the QLIKE-with-floor
training loss with MSE-of-log fallback, multi-seed runs, and the **seed-ensemble mean prediction
sequence** — the object DM tests (§7).

## 2. Scope

**`src/models.py`** — the architecture family (frozen; Optuna tunes sizes *within* it, §4):
LSTM (1–2 layers, hidden 8–128, dropout 0–0.5) over a lookback window of feature vectors →
linear head → scalar prediction of the horizon-h **log-target**. GRU/TCN are noted as defensible
equivalents and **never** swept (§8).

**`src/train.py`** — the harness:
1. Sequence windowing: turn the flat `FeatureSet` into (lookback × n_features) tensors per
   origin date; sequence length is a tuned HP (frozen after tuning).
2. Seeded training loop: seeds Python/NumPy/torch; early stopping on validation loss with
   patience; gradient clipping; Adam.
3. Loss: QLIKE-with-floor on `exp(pred_log)` vs level target (primary); **fallback** to MSE on
   log-target, auto-triggered per run if QLIKE training produces non-finite loss or diverges
   (divergence rule: val loss > 10× its initial value or NaN for 3 consecutive epochs); which
   loss ran is recorded in the run config.
4. Walk-forward protocol: per `splits.retrain_folds`, re-train with **frozen HPs** on each fold's
   rolling window, predict the fold's test origins (monthly retrain, §8).
5. Multi-seed: run S seeds (~5 dev, ≥10 headline, §7); emit per-seed prediction frames and the
   **seed-ensemble mean** frame (per-origin mean of per-seed level-space predictions).
6. Optuna search (Stage 1): tune on train/val of the canonical split ONLY, TV target only (§4);
   objective = val QLIKE; search space = hidden size, layers, dropout, lr, weight decay, batch
   size, sequence length. Emits the frozen HP set into config.

**Non-goals**
- Which features enter is the experiment config (§6); no feature logic here.
- No result persistence (calls `io`); no metric definitions (calls `metrics`).
- No architecture search beyond the frozen family; no cadence/window sweep (§4).

## 3. Inherited invariants

- ALWAYS seed Python, NumPy, and torch; document residual non-determinism (CPU target → expect
  exact determinism with seeded torch; assert it). (§7)
- **NEVER report a single-seed result as an edge**; the harness always runs the seed list. (§7)
- **DM tests the seed-ensemble mean prediction sequence** — the harness must produce it as a
  first-class artifact, pre-specified, not derived ad hoc. (§7)
- ALWAYS model log-target internally, exponentiate to level space for scoring. (§7)
- Scalers fit on train only (enforced with `features.TrainOnlyScaler`; fit-indices ⊆ fit window).
- Tuning happens ONCE, on TV, on train/val only; the test region never influences any HP. (§4, §7)
- Frozen controls (HPs, window length, cadence, architecture family) live in
  `configs/default.yaml` and do not change between RV-only, RV+aux, or across targets. (§7)

## 4. Interfaces / contracts

```python
class SmallLSTM(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float): ...
    def forward(self, x: Tensor) -> Tensor   # (B, L, F) -> (B,) log-target prediction

@dataclass(frozen=True)
class TrainResult:
    preds_per_seed: pd.DataFrame   # index origin dates; one column per seed (level/variance space)
    preds_ensemble: pd.Series      # per-origin mean over seeds (the DM object)
    loss_used: str                 # "qlike" | "mse_log" per seed, recorded
    early_stop_epochs: dict[int, int]

def run_walk_forward(features: FeatureSet, folds: list[RetrainFold], hp: dict,
                     seeds: list[int], cfg: dict) -> TrainResult
def tune(features: FeatureSet, split: CanonicalSplit, cfg: dict,
         n_trials: int) -> dict            # frozen HP set (Stage 1 only)
```

Contracts:
- Prediction rows are indexed by forecast-origin date, aligned to the same convention as
  `features`/`baselines` — directly comparable to HAR's frame on identical inputs (§1 use case).
- `preds_ensemble[t] == preds_per_seed.loc[t].mean()` exactly (done-check).
- Sequence windows must respect the embargo by construction: a window ending at origin t uses
  features at dates ≤ t only; windows never straddle into the fold's test region during fitting.
- Val split for early stopping *inside a fold*: the trailing 15% of the fold's fit window
  (chronological, never shuffled) — pre-specified here.

## 5. Dependencies

`features`, `splits`, `metrics` (manifesto §6). `io` consumes its outputs at the experiment
layer.

## 6. Tech stack (this module)

PyTorch (CPU wheel locally; Colab for the Optuna search if needed, §10), Optuna. Pinned in
`requirements.txt`, matched to Colab's environment at setup (§10).

## 7. Requirements & behavior

R1. Same seed + same inputs → bitwise-identical predictions (CPU). torch seeded via
    `torch.manual_seed`; dataloader shuffling seeded; `torch.use_deterministic_algorithms(True)`.
R2. Log-target round-trip: model predicts log; exponentiation to level space happens in exactly
    one place; round-trip error on a fixture < 1e-10.
R3. Early stopping: patience (default 10) on the inner-val loss; must trigger on a synthetic
    overfit fixture; best-epoch weights restored.
R4. QLIKE training loss uses the same floor constant as `metrics` (from config) — never a
    separate floor; implemented in torch, verified equal to `metrics.qlike` on a fixture.
R5. Fallback activation is automatic, logged, and per-seed; a headline run mixing losses across
    seeds is flagged in the artifact (comparability caveat surfaces in the report).
R6. Multi-seed runs share everything except the seed; the seed list is config, ≥10 for headline.
R7. `tune` must be structurally unable to see test data: it receives only train/val rows (the
    harness slices before Optuna is invoked); assert max date seen < test_start − embargo.
R8. Every run emits the config actually used (resolved HPs, seeds, loss, window) for `io`.

## 8. Edge cases & error handling

E1. Fold fit window shorter than sequence length + inner-val + minimum batches → hard error
    (config bug; frozen controls guarantee this never happens on the real calendar).
E2. Non-finite loss at first epoch (bad init) → re-init once from the next seed offset; recurring
    → fallback loss path (R5).
E3. A fold where early stopping never triggers → cap at max_epochs (config, default 200), log.
E4. NaN in features → refuse to train (features R1 guarantees; assert defensively).
E5. Seed list with duplicates → ValueError.
E6. GPU present: ignored by default (determinism first); enabling CUDA requires an explicit
    config flag and documents non-determinism (§7).

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_model_harness.py` (manifesto §6 done-check):

S1. Same seed → identical predictions (two full runs on a small fixture, exact equality).
S2. Log-target round-trips within tolerance.
S3. Early stopping triggers on a synthetic overfit fixture (tiny data, big net) before
    max_epochs.
S4. Forced-non-finite QLIKE path falls back to MSE-of-log cleanly and records it.
S5. Ensemble-mean predictions equal the mean of per-seed predictions on a fixture (exact).
S6. Torch QLIKE == `metrics.qlike` on a shared fixture (1e-8).
S7. `tune` smoke test: 2 trials on a fixture; asserts no test-region date ever enters (R7 probe).

## 10. Open questions

- Inner-val fraction (15%) and patience (10) are pre-specified defaults not in the manifesto —
  they are *frozen controls* once Stage 1 tuning starts; confirm before tuning.
- Optuna trial budget (default 50) and sampler (TPE, seeded) — decide at Stage 1 kickoff; cost
  lives on Colab free tier (§10).
- Whether h=1 and h=5 get separate tuned HP sets or share one (manifesto says tuned once on TV;
  silent on horizon). Default adopted: one search jointly minimizing mean val QLIKE across both
  horizons via two heads? **No** — simpler pre-registration: tune on h=1 only, reuse for h=5
  (consistent with "tuned once"; the h=5 net differs only in its label). Flag for confirmation
  before Stage 1.
