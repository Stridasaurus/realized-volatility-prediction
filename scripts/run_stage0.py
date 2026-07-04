"""Stage 0: score the classical field on the frozen splits, both targets, both horizons.

Thin experiment layer (manifesto: shared library, thin notebooks/runners) — all logic
lives in src/. Saves one run under results/stage0_classical/ via the io contract, then
replays the headline table from the saved artifacts alone (S4: results are pure replay).

Usage:
  python scripts/run_stage0.py            # run the field, save artifacts, replay
  python scripts/run_stage0.py --replay   # replay the latest saved run only
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repo root (NOT src/ — stdlib-shadowing guard)

from src import io as io_mod  # noqa: E402
from src.baselines import CLASSICAL_MODELS, forecast_classical  # noqa: E402
from src.data import build_targets, load_snapshot  # noqa: E402
from src.metrics import dm_test, qlike, qlike_series, rmse  # noqa: E402
from src.splits import SplitConfig, canonical_split, retrain_folds  # noqa: E402

EXPERIMENT = "stage0_classical"
BIND_RATE_THRESHOLD = 0.001  # S8: a floor binding on >0.1% of predictions is flagged


def run() -> Path:
    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    targets = build_targets(snap)

    split_cfg = SplitConfig(
        train_start=cfg["splits"]["train_start"],
        val_start=cfg["splits"]["val_start"],
        test_start=cfg["splits"]["test_start"],
        test_end=cfg["splits"]["test_end"],
        embargo_days=cfg["splits"]["embargo_days"],
    )
    split = canonical_split(targets.index, split_cfg)
    folds = retrain_folds(targets.index, split_cfg, cfg["splits"]["window_len"])
    returns_cc = np.log(snap.spy["close"] / snap.spy["close"].shift(1)).reindex(
        targets.index
    )

    frames: dict[tuple[str, str, int], pd.DataFrame] = {}
    for model in CLASSICAL_MODELS:
        for target in ("rv_tv", "rv_oc"):
            for horizon in (1, 5):
                print(f"scoring {model} on {target} h={horizon} ...", flush=True)
                frames[(model, target, horizon)] = forecast_classical(
                    model,
                    targets,
                    returns_cc if model == "garch" else None,
                    folds,
                    target=target,
                    horizon=horizon,
                    cfg=cfg,
                )

    # ---- metrics: QLIKE/RMSE per block; DM vs HAR on aligned origins --------------
    metrics: dict = {"models": {}, "dm": {}}
    for (model, target, horizon), f in frames.items():
        floor = cfg["floors"][target]
        q = qlike(f["y_pred"].to_numpy(), f["y_true"].to_numpy(), floor)
        leaf = {
            "qlike": q.value,
            "rmse": rmse(f["y_pred"].to_numpy(), f["y_true"].to_numpy()),
            "bind_rate": q.bind_rate,
            "n": q.n,
        }
        if q.bind_rate > BIND_RATE_THRESHOLD:
            leaf["bind_rate_flag"] = True
            warnings.warn(
                f"{model}/{target}/h{horizon}: QLIKE floor bind rate "
                f"{q.bind_rate:.2%} exceeds the S8 threshold",
                UserWarning,
            )
        if bool(f.attrs.get("object_mismatch")):
            leaf["object_mismatch"] = True  # GARCH forecasts total variance, not OC
        metrics["models"].setdefault(model, {}).setdefault(target, {})[str(horizon)] = (
            leaf
        )

    for target in ("rv_tv", "rv_oc"):
        floor = cfg["floors"][target]
        for horizon in (1, 5):
            har = frames[("har", target, horizon)]
            for model in CLASSICAL_MODELS:
                if model == "har":
                    continue
                f = frames[(model, target, horizon)]
                common = f.index.intersection(har.index)
                res = dm_test(
                    qlike_series(
                        f.loc[common, "y_pred"].to_numpy(),
                        f.loc[common, "y_true"].to_numpy(),
                        floor,
                    ),
                    qlike_series(
                        har.loc[common, "y_pred"].to_numpy(),
                        har.loc[common, "y_true"].to_numpy(),
                        floor,
                    ),
                    h=horizon,
                )
                metrics["dm"][f"{model}_vs_har_{target}_h{horizon}"] = {
                    "stat": res.stat,
                    "p_value": res.p_value,
                    "hac_lag": res.hac_lag,
                    "mean_loss_diff": res.mean_loss_diff,
                    # classical models are deterministic — no seed ensemble (S6 is a
                    # net-only criterion); recorded as None, never fabricated
                    "sign_agreement_frac": None,
                    "n": res.n,
                }

    # ---- artifacts ----------------------------------------------------------------
    preds = pd.concat(
        f.reset_index()
        .rename(columns={"index": "origin_date"})
        .assign(seed=-1)[
            ["origin_date", "target", "horizon", "model", "seed", "y_pred", "y_true"]
        ]
        for f in frames.values()
    ).reset_index(drop=True)

    run_cfg = dict(cfg)
    run_cfg["experiment"] = EXPERIMENT
    run_cfg["models"] = list(CLASSICAL_MODELS)
    run_cfg["window_len_resolved"] = len(split.train_idx)
    run_cfg["n_folds"] = len(folds)
    run_cfg["snapshot_files"] = snap.manifest["files"]
    run_dir = io_mod.save_run(EXPERIMENT, run_cfg, preds, metrics)
    print(f"\nsaved run: {run_dir}")
    return run_dir


def replay(run_dir: Path | None = None) -> None:
    """Rebuild the headline table from saved artifacts alone — no scoring, no data."""
    if run_dir is None:
        runs = sorted((io_mod.RESULTS_DIR / EXPERIMENT).iterdir())
        if not runs:
            sys.exit(f"no saved runs under {io_mod.RESULTS_DIR / EXPERIMENT}")
        run_dir = runs[-1]
    problems = io_mod.validate_run(run_dir)
    if problems:
        sys.exit(f"run failed validation: {problems}")
    arts = io_mod.load_run(run_dir)

    print(f"\n=== Stage 0 headline (pure replay of {run_dir.name}) ===")
    for target in ("rv_tv", "rv_oc"):
        for horizon in ("1", "5"):
            print(f"\n{target} h={horizon}   (QLIKE lower is better)")
            rows = []
            for model, per_target in arts.metrics["models"].items():
                leaf = per_target[target][horizon]
                dm = arts.metrics["dm"].get(f"{model}_vs_har_{target}_h{horizon}", {})
                rows.append(
                    {
                        "model": model,
                        "qlike": leaf["qlike"],
                        "rmse": leaf["rmse"],
                        "bind%": 100 * leaf["bind_rate"],
                        "n": leaf["n"],
                        "DM stat": dm.get("stat"),
                        "p vs HAR": dm.get("p_value"),
                        "note": "obj-mismatch" if leaf.get("object_mismatch") else "",
                    }
                )
            table = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
            print(
                table.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4g}",
                    na_rep="—",
                )
            )
    print(
        "\nAll DM p-values reported together, per model/horizon/target — never a "
        "curated subset (manifesto s7). GARCH rows on rv_oc carry the object-"
        "mismatch note: its forecast object is total variance."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="replay latest run only")
    args = ap.parse_args()
    if args.replay:
        replay()
    else:
        replay(run())
