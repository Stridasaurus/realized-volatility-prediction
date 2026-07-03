# design — `io / artifacts` (`src/io.py`)

## 1. Overview

Implements `specs/io/SPEC.md`: results schema, save/load/validate, config loading with
`embargo: auto` resolution, checkpoint pointer policy. Greenfield.

## 2. Approach / architecture

Plain functions + one `RunArtifacts` dataclass. Atomic writes via tmp-dir rename (SPEC E6).
JSON with `sort_keys=True, indent=2`. Parquet via pyarrow. Config: `configs/default.yaml`
loaded with `yaml.safe_load`; a `_resolve` pass computes `embargo_days` from
`max(har_monthly_lag, lstm.seq_len_max, features.max_aux_window)` and injects floors/paths;
resolved dict is what every caller sees and what `save_run` snapshots.

Import discipline: module is always referenced as `src.io`; a repo-wide guard test greps for
`sys.path.append`-style additions of `src` (SPEC §4).

## 3. File-by-file plan

- `src/io.py` — `RunArtifacts`, `load_config`, `save_run`, `load_run`, `validate_run`,
  privates `_resolve_config`, `_atomic_write_dir`, `_run_id`, `_versions` (package version
  capture), `_git_rev` (best-effort `git rev-parse`, subprocess with fallback).
- `configs/default.yaml` — created here (single home for frozen controls): splits boundaries,
  `embargo_days: auto`, window/cadence, lookbacks (har_monthly_lag: 22, lstm seq_len_max,
  max_aux_window: 22), floors (written by freeze script), seeds, metrics floor keys, model HP
  defaults + search space bounds.
- `tests/test_io.py` — S1–S6.
- `results/.gitkeep` — layout root; run dirs are committed selectively (config/preds/metrics).

## 4. Data models / schemas

Per SPEC §4: run dir layout, `preds.parquet` long format (`origin_date, target, horizon, model,
seed, y_pred, y_true`; ensemble as `seed == -1`), `metrics.json` blocks incl. complete `dm`
comparisons and optional `cuts` key.

## 5. Key interfaces & signatures

Per SPEC §4 (`load_config`, `save_run`, `load_run`, `validate_run`).

## 6. Implementation sequence

1. `configs/default.yaml` authored with all frozen controls (floors placeholder until freeze).
2. `load_config` + `_resolve_config` (auto-embargo; missing-component error) → S4.
3. `_atomic_write_dir`, `_run_id`, `save_run` (overwrite refusal) → S5.
4. `load_run`, `validate_run` (schema checks, ensemble-mean consistency 1e-12) → S1–S3.
5. Shadowing guard test (S6) over `src/` + `notebooks/`.

## 7. Integration points

Every module reads frozen controls through `load_config` (splits boundaries → experiment layer;
floor → metrics args; HPs/seeds → train). `data`'s freeze script writes floor values into the
yaml this module owns. `99_evaluation` uses `load_run` + `validate_run` exclusively (S4 gate:
pure replay). Checkpoints: `train` hands weight blobs + metadata; io writes pointer file, blob
to Drive path or local per config.

## 8. Test plan

SPEC §9 S1–S6: round-trip equality (config/preds/metrics), required-key deletion sweep,
corrupted-ensemble detection, auto-embargo fixture, overwrite + tmp-dir fixtures, shadow-guard
grep. Plus: JSON determinism (two saves → identical bytes).

## 9. Risks & open questions

- stdlib-`io` shadowing is the known sharp edge — mitigated by import discipline + guard test;
  renaming would need a manifesto edit (recorded in SPEC Open Qs, workflow audit).
- `_git_rev` on Colab (no git) → records `"unknown"`; acceptable, provenance carried by
  snapshot hash + versions.
- Committing per-run parquet keeps the repo honest (S4) but grows it; daily-scale preds are
  ~100 KB/run — fine without LFS per manifesto §10.
