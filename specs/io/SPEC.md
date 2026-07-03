# SPEC — `io / artifacts` (`src/io.py` + `results/` schema)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `io / artifacts` module
> of manifesto §6.

## 1. Purpose

The save/load contract for every run (manifesto §6 `io / artifacts`). Owns the
`config.json` / `preds.parquet` / `metrics.json` / checkpoint format and the
`results/<experiment>/<run_id>/` layout. This is what makes S4 ("results are pure replay") and
S2 (reproducibility) executable: `99_evaluation` rebuilds the headline table from these files,
re-running no training.

## 2. Scope

1. The `results/` directory schema and a `save_run` / `load_run` API.
2. Schema validation (required keys, types) on save **and** load.
3. Config handling: load `configs/default.yaml`, resolve derived values (notably
   `embargo_days = max over all configured lookbacks` — the decision recorded at kickoff), and
   snapshot the fully-resolved config into each run's `config.json`.
4. Checkpoint policy: best-model weights go to Google Drive (path recorded in `config.json`);
   small artifacts (`config`/`preds`/`metrics`) are committed to git (§10 — they *are* the
   result; skip git-LFS).

**Non-goals**
- Does not compute anything in `metrics.json` (that's `metrics` at the experiment layer).
- Does not decide experiment names or run semantics; it persists what it is given.
- No database, no W&B (named upgrade path, §8), no cloud APIs — Drive is a mounted filesystem
  path in Colab, a plain local path otherwise.

## 3. Inherited invariants

- Commit the small artifacts; they are the deliverable. (§7, §10)
- Per-seed predictions **and** the ensemble mean are both saved. (§6)
- Pin everything a replay needs: the resolved config includes snapshot hash, split boundaries,
  embargo, window length, HPs, seeds, loss used, floor value, library versions. (§3 S2/S4)
- Frozen controls live in `configs/default.yaml`; runs record the *resolved* copy, never mutate
  the source. (§7)

## 4. Interfaces / contracts

Layout: `results/<experiment>/<run_id>/` with `run_id = <utc-timestamp>_<8-char-config-hash>`.

```
results/<experiment>/<run_id>/
  config.json        # fully-resolved config + provenance (snapshot sha, versions, git rev)
  preds.parquet      # long format: origin_date, target, horizon, model, seed, y_pred, y_true
                     #   seed = -1 encodes the seed-ensemble mean row block
  metrics.json       # per (model, target, horizon): qlike, rmse, bind_rate, n;
                     #   dm: per comparison vs HAR: stat, p_value, hac_lag, mean_loss_diff,
                     #   sign_agreement_frac (S6); all comparisons present (§7: never a subset)
  checkpoint.txt     # optional pointer: Drive path + sha256 of best weights per fold/seed
```

```python
def load_config(path: Path = CONFIG_PATH) -> dict          # resolves embargo, validates
def save_run(experiment: str, config: dict, preds: pd.DataFrame,
             metrics: dict, results_dir: Path = RESULTS_DIR) -> Path  # returns run dir
def load_run(run_dir: Path) -> RunArtifacts                # dataclass of the three objects
def validate_run(run_dir: Path) -> list[str]               # [] if schema-valid
```

Contracts:
- `save_run` → `load_run` round-trips to identical objects (DataFrame equality incl. dtypes).
- `preds.parquet` written via `pyarrow`; datetimes tz-naive; floats float64.
- **Import discipline:** `src/io.py` shadows the stdlib `io` module by name. It must only ever
  be imported as `from src import io as io_mod` (or `import src.io`), and `src/` itself must
  NEVER be added to `sys.path` (which would shadow stdlib `io` for pandas/pyarrow and break the
  interpreter). A test enforces that no notebook or module does `sys.path.append(...'src')`.

## 5. Dependencies

Nothing among project modules (manifesto §6: schema; used by all). `pandas`, `pyarrow`,
`pyyaml`, `json`, `hashlib`.

## 6. Tech stack (this module)

`pyyaml` for `configs/default.yaml`; `pyarrow` parquet; `json` with sorted keys + indent for
diff-able committed artifacts (§10). Pinned in `requirements.txt`.

## 7. Requirements & behavior

R1. `save_run` must refuse to overwrite an existing `run_id` directory (append-only results).
R2. Schema validation must check: all required keys present and typed; every (model, target,
    horizon) block in `metrics.json` complete; `preds` contains a `seed == -1` ensemble block
    whose values equal the per-seed mean within 1e-12 for every model with >1 seed.
R3. `config.json` must embed: snapshot manifest hash, git revision (best-effort), resolved
    embargo and window length, floor values, seed list, package versions of
    torch/numpy/pandas/statsmodels/arch/optuna.
R4. `load_config` must resolve `embargo_days: auto` → max(HAR monthly lag 22, LSTM max sequence
    length from the search-space config, longest aux window) and fail if any component is
    missing; explicit integer values pass through.
R5. All JSON written deterministically (sorted keys) so git diffs are meaningful.
R6. `validate_run` is what `99_evaluation` calls before trusting any run; it never mutates.

## 8. Edge cases & error handling

E1. Run dir exists → error naming it (R1); caller picks a new run_id (timestamp makes collision
    ~impossible; hash collision with different config = error, not overwrite).
E2. `preds.parquet` missing the ensemble block for a multi-seed run → validation error (the DM
    object must exist as saved data, §7).
E3. Config with unknown keys → warn (forward compatibility) — but *missing* required keys →
    error.
E4. Drive path unavailable (local run) → checkpoint pointer written with `location: local` and a
    local path; never a silent skip.
E5. Non-UTC or tz-aware timestamps in preds → normalized to tz-naive dates on save, asserted on
    load.
E6. Partial write (crash mid-save) → `save_run` writes to `<run_dir>.tmp` then renames; a
    leftover `.tmp` dir is reported by `validate_run`.

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_io.py` (manifesto §6 done-check):

S1. save→load round-trips to identical objects (config dict ==, DataFrame equal incl. dtypes,
    metrics dict ==).
S2. The results schema validates on a good fixture; each required-key deletion is caught by
    `validate_run`.
S3. Ensemble block consistency check catches a corrupted ensemble row.
S4. `embargo: auto` resolution equals the hand-computed max on a fixture config; missing
    component errors.
S5. Overwrite refusal (E1) and tmp-rename atomicity (E6 fixture: pre-created `.tmp`).
S6. The `sys.path`/stdlib-shadowing guard test passes over the repo (greps notebooks and src for
    the forbidden pattern).

## 10. Open questions

- `run_id` format above is this spec's default (manifesto silent); adopted as stated.
- Whether `metrics.json` should also embed the S7 pre-registered cut results (regime/horizon/
  source/target) or those live in a separate `cuts.json` — default: same `metrics.json` under a
  `cuts` key, so one file is the complete scored record. Flag if separate files preferred.
- The manifesto names the file `src/io.py`; the stdlib shadowing risk (see §4 Contracts) is
  handled by import discipline rather than renaming, to honor the manifesto's naming. If this
  proves fragile in notebooks, renaming to `src/artifacts.py` needs a manifesto edit first —
  recorded for the workflow audit.
