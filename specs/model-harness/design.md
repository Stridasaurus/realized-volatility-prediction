# design — `model + harness` (`src/models.py` + `src/train.py`)

## 1. Overview

Implements `specs/model-harness/SPEC.md`: the frozen small-LSTM family, the seeded leak-safe
train/eval loop, walk-forward retraining, multi-seed ensembling, and the (Stage-1) Optuna
search. Greenfield; sits on `features`, `splits`, `metrics`.

## 2. Approach / architecture

`models.py` holds only `SmallLSTM` (nn.LSTM → dropout → Linear head on last hidden state).
`train.py` holds: `WindowDataset` (builds (L,F) windows ending at each origin, entirely from
rows ≤ origin), `_train_one(seed, fold, hp)` (inner chronological 85/15 fit/val split, Adam,
grad-clip 1.0, early stopping patience 10, best-weights restore), `run_walk_forward` (folds ×
seeds loops, assembles per-seed frames + ensemble mean), `tune` (Optuna TPE seeded, train/val
of canonical split only), and `qlike_loss_torch` (floor via `torch.clamp`, mirrors
`metrics.qlike` exactly).

Determinism: `torch.manual_seed`, `numpy`/`random` seeds, `torch.use_deterministic_algorithms(True)`,
num_workers=0, CPU default (SPEC E6).

Loss fallback state machine per seed: start QLIKE; on divergence rule (SPEC §2.3) restart that
seed's fold training with MSE-of-log; record `loss_used`.

## 3. File-by-file plan

- `src/models.py` — `SmallLSTM`.
- `src/train.py` — `WindowDataset`, `TrainResult`, `run_walk_forward`, `tune`,
  `qlike_loss_torch`, privates `_train_one`, `_seed_everything`, `_inner_split`.
- `tests/test_model_harness.py` — S1–S7 on tiny fixtures (short synthetic FeatureSet, 2 folds,
  hidden=8, epochs≤30 so the suite stays fast).

## 4. Data models / schemas

`TrainResult` per SPEC §4. `WindowDataset[i] -> (x: float32 (L,F), y_log: float32, origin: ts)`.
HP dict keys: `hidden, layers, dropout, lr, weight_decay, batch_size, seq_len` (the frozen set
serialized into `configs/default.yaml` after Stage-1 tuning).

## 5. Key interfaces & signatures

Per SPEC §4 (`run_walk_forward`, `tune`, `SmallLSTM`). `qlike_loss_torch(pred_log, y_level,
floor)` exponentiates inside — single exponentiation site (SPEC R2).

## 6. Implementation sequence

1. `SmallLSTM` + shape test; `_seed_everything`.
2. `WindowDataset` with ≤-origin windowing + a probe test (perturb later rows → windows at
   earlier origins unchanged).
3. `qlike_loss_torch` == `metrics.qlike` fixture (S6).
4. `_train_one` (inner split, early stop, fallback machine) → S3, S4.
5. `run_walk_forward` (multi-seed, ensemble mean) → S1, S5.
6. `tune` (Optuna study, seeded sampler, R7 max-date assert) → S7 smoke.

## 7. Integration points

`features.FeatureSet` in; `splits.RetrainFold` protocol shared with baselines (identical
origins → directly comparable frames, the §1 controlled comparison); floor + HP + seed list via
`io.load_config`; outputs persisted by `io.save_run` at the experiment layer. Colab: only
`tune` is worth GPU; guarded by config flag per SPEC E6.

## 8. Test plan

SPEC §9 S1–S7; all CPU, tiny nets, seeded. S1 is exact equality of two full runs. S3 uses
50 rows/hidden=64 to force overfit. S4 forces divergence via lr=1e3 monkeypatch of the rule
threshold. Suite budget: < 2 min.

## 9. Risks & open questions

- `torch.use_deterministic_algorithms(True)` can raise on some ops — LSTM/Linear on CPU are
  safe; keep the flag and let any violation fail tests early.
- Optuna trial budget / h-horizon tuning policy: SPEC Open Qs, decided at Stage-1 kickoff (not
  needed for Stage 0).
- Windowing near fold boundaries: windows may span into pre-fit history (allowed — info ≤ t);
  the test in step 2 pins the semantics.
