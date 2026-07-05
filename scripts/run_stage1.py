"""Stage 1: leak-safe Optuna tuning on TV, then RV-only net vs HAR on identical inputs.

Thin experiment layer (manifesto: shared library, thin notebooks/runners) — all logic
lives in src/. Two independent phases, run separately:

  python scripts/run_stage1.py tune [--n-trials 50] [--dry-run]
      Optuna search on train/val of the canonical TV split, horizon h=1 only (the
      frozen HP set is reused for h=5 unchanged — pre-registered, see
      specs/model-harness/SPEC.md s10 and MANIFESTO.md s8). Writes the winning HPs
      into configs/default.yaml's model.hp block (the frozen control every later
      experiment reads) and saves a tuning artifact under results/stage1_tune/.
      --dry-run prints the result without touching configs/default.yaml.

  python scripts/run_stage1.py compare [--seeds dev|headline] [--replay]
      RV-only LSTM (tiers=("t1",), i.e. target history only — identical inputs to
      HAR) vs HAR, walk-forward over the frozen retrain folds, both TV horizons.
      Saves one run under results/stage1_net_vs_har/ via the io contract, then
      replays the headline table from the saved artifacts alone (S4).

Both phases are held out of the tuned HP set until a human commits the config change
that `tune` writes — `compare` always reads whatever is currently in
configs/default.yaml, never an in-memory value from a `tune` call in the same process.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repo root (NOT src/ — stdlib-shadowing guard)

from src import io as io_mod  # noqa: E402
from src import train as train_mod  # noqa: E402
from src.baselines import forecast_classical  # noqa: E402
from src.data import build_targets, load_snapshot  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import dm_test, qlike, qlike_series, rmse  # noqa: E402
from src.splits import SplitConfig, canonical_split, retrain_folds  # noqa: E402

TUNE_EXPERIMENT = "stage1_tune"
COMPARE_EXPERIMENT = "stage1_net_vs_har"
BIND_RATE_THRESHOLD = 0.001  # S8: a floor binding on >0.1% of predictions is flagged
TARGET = "rv_tv"  # tuning + the net-vs-HAR comparison are both TV-only (manifesto s4)
HP_KEYS = ("hidden", "layers", "dropout", "lr", "weight_decay", "batch_size", "seq_len")


def _split_cfg(cfg: dict) -> SplitConfig:
    return SplitConfig(
        train_start=cfg["splits"]["train_start"],
        val_start=cfg["splits"]["val_start"],
        test_start=cfg["splits"]["test_start"],
        test_end=cfg["splits"]["test_end"],
        embargo_days=cfg["splits"]["embargo_days"],
    )


def _write_frozen_hp(config_path: Path, hp: dict) -> None:
    """Patch configs/default.yaml's model.hp block in place, preserving comments.

    Targeted line rewrite rather than a full YAML round-trip: pyyaml/ruamel would
    either drop the file's comments or require a new dependency, and the block's
    shape (model: -> hp: -> one "key: value" line per HP_KEYS entry) is frozen by
    src/train.py's own search space, so a line-level patch is safe here.
    """
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out, in_model, in_hp = [], False, False
    seen = set()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^model:\s*$", line):
            in_model = True
            out.append(line)
            continue
        if in_model and re.match(r"^\s{2}hp:(\s|$|#)", line):
            in_hp = True
            out.append(line)
            continue
        if in_hp:
            m = re.match(r"^(\s{4})(\w+):", line)
            if m and m.group(2) in HP_KEYS:
                key = m.group(2)
                # yaml.safe_dump (not repr) so e.g. 1e-05 round-trips as a float,
                # not a bare string (PyYAML's float resolver requires a ".").
                scalar = yaml.safe_dump(hp[key]).split("\n", 1)[0]
                out.append(f"{m.group(1)}{key}: {scalar}\n")
                seen.add(key)
                continue
            if not stripped or not line.startswith("    "):
                in_hp = False
                in_model = False
        out.append(line)
    missing = set(HP_KEYS) - seen
    if missing:
        raise ValueError(
            f"configs/default.yaml model.hp block is missing key(s) {missing}; "
            "refusing to write a partially-patched config"
        )
    config_path.write_text("".join(out), encoding="utf-8")


def run_tune(n_trials: int, dry_run: bool) -> dict:
    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    targets = build_targets(snap)
    split_cfg = _split_cfg(cfg)
    split = canonical_split(targets.index, split_cfg)

    features = build_features(targets, snap, target=TARGET, horizon=1, tiers=("t1",))
    print(f"tuning on {TARGET} h=1, {n_trials} trials, train/val only ...", flush=True)
    best_hp = train_mod.tune(features, split, cfg, n_trials=n_trials)

    artifact = {
        "target": TARGET,
        "horizon_tuned": 1,
        "horizon_reused": 5,
        "n_trials": n_trials,
        "best_hp": best_hp,
        "note": (
            "frozen once on h=1/TV and reused unchanged for h=5, per "
            "specs/model-harness/SPEC.md s10 (pre-registered before Stage 1)."
        ),
    }
    out_dir = REPO_ROOT / "results" / TUNE_EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}_hp.json"
    if not dry_run:
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        print(f"saved tuning artifact: {out_path}")

    print("\nbest HP set:")
    for k in HP_KEYS:
        print(f"  {k}: {best_hp[k]}")

    if dry_run:
        print("\n--dry-run: configs/default.yaml NOT modified.")
    else:
        _write_frozen_hp(io_mod.CONFIG_PATH, best_hp)
        print(
            f"\nwrote frozen HP set into {io_mod.CONFIG_PATH} (model.hp). "
            "Review the diff and commit it before running `compare` — that commit "
            "IS the pre-registration of the frozen control (manifesto s7)."
        )
    return best_hp


def run_compare(seeds: list[int], folds_limit: int | None = None) -> Path:
    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    targets = build_targets(snap)
    split_cfg = _split_cfg(cfg)
    folds = retrain_folds(targets.index, split_cfg, cfg["splits"]["window_len"])
    if folds_limit is not None:
        folds = folds[:folds_limit]

    metrics: dict = {"models": {}, "dm": {}}
    pred_frames = []

    for horizon in (1, 5):
        print(f"--- TV h={horizon} ---", flush=True)
        floor = cfg["floors"][TARGET]

        print("  scoring har ...", flush=True)
        har = forecast_classical(
            "har", targets, None, folds, target=TARGET, horizon=horizon, cfg=cfg
        )
        q = qlike(har["y_pred"].to_numpy(), har["y_true"].to_numpy(), floor)
        metrics["models"].setdefault("har", {}).setdefault(TARGET, {})[str(horizon)] = {
            "qlike": q.value,
            "rmse": rmse(har["y_pred"].to_numpy(), har["y_true"].to_numpy()),
            "bind_rate": q.bind_rate,
            "n": q.n,
        }
        pred_frames.append(
            har.reset_index()
            .rename(columns={"index": "origin_date"})
            .assign(target=TARGET, horizon=horizon, model="har", seed=-1)[
                ["origin_date", "target", "horizon", "model", "seed", "y_pred", "y_true"]
            ]
        )

        print("  training lstm_rvonly (RV-only: tiers=('t1',)) ...", flush=True)
        features = build_features(
            targets, snap, target=TARGET, horizon=horizon, tiers=("t1",)
        )
        result = train_mod.run_walk_forward(features, folds, cfg["model"]["hp"], seeds, cfg)
        ensemble = result.preds_ensemble
        y_true_net = features.y.reindex(ensemble.index)

        q_net = qlike(ensemble.to_numpy(), y_true_net.to_numpy(), floor)
        leaf = {
            "qlike": q_net.value,
            "rmse": rmse(ensemble.to_numpy(), y_true_net.to_numpy()),
            "bind_rate": q_net.bind_rate,
            "n": q_net.n,
            "loss_used": result.loss_used,
        }
        if q_net.bind_rate > BIND_RATE_THRESHOLD:
            leaf["bind_rate_flag"] = True
            warnings.warn(
                f"lstm_rvonly/{TARGET}/h{horizon}: QLIKE floor bind rate "
                f"{q_net.bind_rate:.2%} exceeds the S8 threshold",
                UserWarning,
            )
        metrics["models"].setdefault("lstm_rvonly", {}).setdefault(TARGET, {})[
            str(horizon)
        ] = leaf

        for seed in seeds:
            s = result.preds_per_seed[seed]
            pred_frames.append(
                pd.DataFrame(
                    {
                        "origin_date": s.index,
                        "target": TARGET,
                        "horizon": horizon,
                        "model": "lstm_rvonly",
                        "seed": seed,
                        "y_pred": s.to_numpy(),
                        "y_true": features.y.reindex(s.index).to_numpy(),
                    }
                )
            )
        pred_frames.append(
            pd.DataFrame(
                {
                    "origin_date": ensemble.index,
                    "target": TARGET,
                    "horizon": horizon,
                    "model": "lstm_rvonly",
                    "seed": -1,
                    "y_pred": ensemble.to_numpy(),
                    "y_true": y_true_net.to_numpy(),
                }
            )
        )

        # ---- DM: net (seed-ensemble mean) vs HAR, aligned on common origins ----
        common = ensemble.index.intersection(har.index)
        net_losses = qlike_series(
            ensemble.loc[common].to_numpy(), y_true_net.loc[common].to_numpy(), floor
        )
        har_losses = qlike_series(
            har.loc[common, "y_pred"].to_numpy(), har.loc[common, "y_true"].to_numpy(), floor
        )
        res = dm_test(net_losses, har_losses, h=horizon)

        ensemble_sign = np.sign(res.mean_loss_diff)
        same_sign = 0
        for seed in seeds:
            seed_pred = result.preds_per_seed[seed].reindex(common)
            seed_losses = qlike_series(
                seed_pred.to_numpy(), y_true_net.loc[common].to_numpy(), floor
            )
            diff = float(np.mean(seed_losses - har_losses))
            if np.sign(diff) == ensemble_sign:
                same_sign += 1

        metrics["dm"][f"lstm_rvonly_vs_har_{TARGET}_h{horizon}"] = {
            "stat": res.stat,
            "p_value": res.p_value,
            "hac_lag": res.hac_lag,
            "mean_loss_diff": res.mean_loss_diff,
            "sign_agreement_frac": same_sign / len(seeds),
            "n": res.n,
        }

    preds = pd.concat(pred_frames).reset_index(drop=True)
    run_cfg = dict(cfg)
    run_cfg["experiment"] = COMPARE_EXPERIMENT
    run_cfg["target"] = TARGET
    run_cfg["seeds_used"] = seeds
    run_cfg["n_folds"] = len(folds)
    run_cfg["snapshot_files"] = snap.manifest["files"]
    run_dir = io_mod.save_run(COMPARE_EXPERIMENT, run_cfg, preds, metrics)
    print(f"\nsaved run: {run_dir}")
    return run_dir


def replay(run_dir: Path | None = None) -> None:
    """Rebuild the headline table from saved artifacts alone — no scoring, no data."""
    if run_dir is None:
        runs = sorted((io_mod.RESULTS_DIR / COMPARE_EXPERIMENT).iterdir())
        if not runs:
            sys.exit(f"no saved runs under {io_mod.RESULTS_DIR / COMPARE_EXPERIMENT}")
        run_dir = runs[-1]
    problems = io_mod.validate_run(run_dir)
    if problems:
        sys.exit(f"run failed validation: {problems}")
    arts = io_mod.load_run(run_dir)

    print(f"\n=== Stage 1 headline (pure replay of {run_dir.name}) ===")
    for horizon in ("1", "5"):
        print(f"\n{TARGET} h={horizon}   (QLIKE lower is better)")
        rows = []
        for model, per_target in arts.metrics["models"].items():
            leaf = per_target[TARGET][horizon]
            dm = arts.metrics["dm"].get(f"{model}_vs_har_{TARGET}_h{horizon}", {})
            rows.append(
                {
                    "model": model,
                    "qlike": leaf["qlike"],
                    "rmse": leaf["rmse"],
                    "bind%": 100 * leaf["bind_rate"],
                    "n": leaf["n"],
                    "DM stat": dm.get("stat"),
                    "p vs HAR": dm.get("p_value"),
                    "seed agree%": (
                        100 * dm["sign_agreement_frac"]
                        if dm.get("sign_agreement_frac") is not None
                        else None
                    ),
                }
            )
        table = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}", na_rep="—"))
    print(
        "\nEdge claim requires DM p<0.05 on the ensemble-mean AND same-sign loss "
        "differential in >=80% of seeds (manifesto S6)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tune = sub.add_parser("tune", help="Optuna search on TV/h=1, train/val only")
    p_tune.add_argument("--n-trials", type=int, default=50)
    p_tune.add_argument("--dry-run", action="store_true")

    p_cmp = sub.add_parser("compare", help="RV-only net vs HAR walk-forward comparison")
    p_cmp.add_argument("--seeds", choices=["dev", "headline"], default="headline")
    p_cmp.add_argument("--replay", action="store_true", help="replay latest run only")

    args = ap.parse_args()
    if args.cmd == "tune":
        run_tune(args.n_trials, args.dry_run)
    else:
        if args.replay:
            replay()
        else:
            cfg = io_mod.load_config()
            seeds = cfg["seeds"][args.seeds]
            replay(run_compare(seeds))
