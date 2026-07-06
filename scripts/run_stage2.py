"""Stage 2: RV+aux net vs HAR, window-length decay curve, final OC re-score.

Thin experiment layer (manifesto: shared library, thin notebooks/runners) — all logic
lives in src/. Three independent phases, run separately (scope/cost decisions made
2026-07-05 with Strider before he stepped away — see the session wrap-up for the
full rationale):

  python scripts/run_stage2.py aux-compare [--seeds dev|headline] [--replay] [--smoke]
      Trains the RV+aux LSTM (tiers=("t1","t2","t3")) vs HAR on rv_tv, both horizons,
      using the frozen HP set from Stage 1 (configs/default.yaml unchanged — only the
      input feature set varies, per the manifesto's frozen-controls invariant, s7).
      Reuses HAR and RV-only LSTM predictions from the most recent
      results/stage1_net_vs_har/ run instead of re-training them — identical config +
      deterministic seeds means re-running would reproduce the same numbers at ~5h of
      wasted compute (model-harness done-check: same seed -> identical predictions).
      Reports DM: aux vs HAR, and aux vs RV-only (the "source" cut in the manifesto's
      where-the-edge-lives decomposition, S7). Default seeds=headline (an S6 edge claim
      needs >=10 seeds).

  python scripts/run_stage2.py decay [--seeds dev|headline] [--smoke]
      Window-length sensitivity study (manifesto s2/s5/s8 "decay curve" -- window
      length is deliberately the swept variable here, unlike everywhere else in this
      project where it is a frozen control). Scoped down for cost: RV-only features
      (tiers=("t1",)), TV target, h=1 only, dev seeds by default (this is a diagnostic,
      not an S6 edge claim, so 5 seeds is legitimate). Sweeps WINDOW_LENGTHS_TD and
      scores lstm_rvonly + HAR at each length.

  python scripts/run_stage2.py final-oc [--seeds dev|headline] [--replay] [--smoke]
      Re-scores the frozen-HP RV+aux LSTM vs HAR on rv_oc (the secondary target), both
      horizons -- the "final two-target comparison" (manifesto s2/s5). RV+aux only
      (not RV-only), per the 2026-07-05 scope decision. Default seeds=headline.

Each phase saves one run via the io contract, then replays its headline table from the
saved artifacts alone (S4: results are pure replay).
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
from src import train as train_mod  # noqa: E402
from src.baselines import forecast_classical  # noqa: E402
from src.data import build_targets, load_snapshot  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import dm_test, qlike, qlike_series, rmse  # noqa: E402
from src.splits import SplitConfig, retrain_folds  # noqa: E402

AUX_EXPERIMENT = "stage2_aux_compare"
DECAY_EXPERIMENT = "stage2_decay"
FINAL_OC_EXPERIMENT = "stage2_final_oc"
BIND_RATE_THRESHOLD = 0.001  # S8: a floor binding on >0.1% of predictions is flagged
TV = "rv_tv"
OC = "rv_oc"
AUX_TIERS = ("t1", "t2", "t3")
RVONLY_TIERS = ("t1",)
# ~2y / ~5y / current frozen default (None -> len(train_idx), resolved at run time)
WINDOW_LENGTHS_TD: list[int | None] = [504, 1260, None]


def _split_cfg(cfg: dict) -> SplitConfig:
    return SplitConfig(
        train_start=cfg["splits"]["train_start"],
        val_start=cfg["splits"]["val_start"],
        test_start=cfg["splits"]["test_start"],
        test_end=cfg["splits"]["test_end"],
        embargo_days=cfg["splits"]["embargo_days"],
    )


def _latest_run(experiment: str) -> Path:
    runs = sorted((io_mod.RESULTS_DIR / experiment).iterdir())
    if not runs:
        sys.exit(f"no saved runs under {io_mod.RESULTS_DIR / experiment}")
    return runs[-1]


def _dm_row(net_losses, har_losses, horizon, seeds, per_seed_preds, y_true, floor):
    """DM(net vs har) + per-seed sign-agreement, matching run_stage1.py's convention."""
    res = dm_test(net_losses, har_losses, h=horizon)
    ensemble_sign = np.sign(res.mean_loss_diff)
    same_sign = 0
    for seed in seeds:
        seed_pred = per_seed_preds[seed]
        seed_losses = qlike_series(seed_pred.to_numpy(), y_true.to_numpy(), floor)
        diff = float(np.mean(seed_losses - har_losses))
        if np.sign(diff) == ensemble_sign:
            same_sign += 1
    return {
        "stat": res.stat,
        "p_value": res.p_value,
        "hac_lag": res.hac_lag,
        "mean_loss_diff": res.mean_loss_diff,
        "sign_agreement_frac": same_sign / len(seeds),
        "n": res.n,
    }


def _train_net(cfg, snap, targets, folds, target, horizon, tiers, seeds):
    features = build_features(targets, snap, target=target, horizon=horizon, tiers=tiers)
    result = train_mod.run_walk_forward(features, folds, cfg["model"]["hp"], seeds, cfg)
    ensemble = result.preds_ensemble
    y_true = features.y.reindex(ensemble.index)
    return features, result, ensemble, y_true


def _net_leaf(ensemble, y_true, floor, loss_used, model_name, target, horizon):
    q = qlike(ensemble.to_numpy(), y_true.to_numpy(), floor)
    leaf = {
        "qlike": q.value,
        "rmse": rmse(ensemble.to_numpy(), y_true.to_numpy()),
        "bind_rate": q.bind_rate,
        "n": q.n,
        "loss_used": loss_used,
    }
    if q.bind_rate > BIND_RATE_THRESHOLD:
        leaf["bind_rate_flag"] = True
        warnings.warn(
            f"{model_name}/{target}/h{horizon}: QLIKE floor bind rate "
            f"{q.bind_rate:.2%} exceeds the S8 threshold",
            UserWarning,
        )
    return leaf


def _pred_frame(ensemble, y_true, target, horizon, model, seed):
    return pd.DataFrame(
        {
            "origin_date": ensemble.index,
            "target": target,
            "horizon": horizon,
            "model": model,
            "seed": seed,
            "y_pred": ensemble.to_numpy(),
            "y_true": y_true.to_numpy(),
        }
    )


# --------------------------------------------------------------------------- aux-compare


def run_aux_compare(seeds: list[int], folds_limit: int | None = None) -> Path:
    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    targets = build_targets(snap)
    split_cfg = _split_cfg(cfg)
    folds = retrain_folds(targets.index, split_cfg, cfg["splits"]["window_len"])
    if folds_limit is not None:
        folds = folds[:folds_limit]

    stage1_dir = _latest_run("stage1_net_vs_har")
    problems = io_mod.validate_run(stage1_dir)
    if problems:
        sys.exit(f"stage1 run failed validation: {problems}")
    stage1 = io_mod.load_run(stage1_dir)
    print(f"reusing har + lstm_rvonly from {stage1_dir.name}", flush=True)

    metrics: dict = {"models": {}, "dm": {}}
    pred_frames = []

    # carry over har + lstm_rvonly verbatim (identical config -> re-running would be
    # a wasted ~5h of deterministic, byte-identical compute)
    for model in ("har", "lstm_rvonly"):
        metrics["models"][model] = stage1.metrics["models"][model]
    for horizon in ("1", "5"):
        key = f"lstm_rvonly_vs_har_{TV}_h{horizon}"
        if key in stage1.metrics["dm"]:
            metrics["dm"][key] = stage1.metrics["dm"][key]
    reused = stage1.preds[stage1.preds["model"].isin(["har", "lstm_rvonly"])].copy()
    pred_frames.append(reused)

    for horizon in (1, 5):
        print(f"--- TV h={horizon} ---", flush=True)
        floor = cfg["floors"][TV]

        print("  training lstm_aux (RV+aux: tiers=('t1','t2','t3')) ...", flush=True)
        features, result, ensemble, y_true = _train_net(
            cfg, snap, targets, folds, TV, horizon, AUX_TIERS, seeds
        )
        leaf = _net_leaf(ensemble, y_true, floor, result.loss_used, "lstm_aux", TV, horizon)
        metrics["models"].setdefault("lstm_aux", {}).setdefault(TV, {})[str(horizon)] = leaf

        for seed in seeds:
            s = result.preds_per_seed[seed]
            pred_frames.append(
                pd.DataFrame(
                    {
                        "origin_date": s.index,
                        "target": TV,
                        "horizon": horizon,
                        "model": "lstm_aux",
                        "seed": seed,
                        "y_pred": s.to_numpy(),
                        "y_true": features.y.reindex(s.index).to_numpy(),
                    }
                )
            )
        pred_frames.append(_pred_frame(ensemble, y_true, TV, horizon, "lstm_aux", -1))

        # ---- DM: aux vs HAR ----
        har_rows = stage1.preds[
            (stage1.preds["model"] == "har")
            & (stage1.preds["horizon"] == horizon)
            & (stage1.preds["seed"] == -1)
        ].set_index("origin_date")
        common = ensemble.index.intersection(har_rows.index)
        aux_losses = qlike_series(
            ensemble.loc[common].to_numpy(), y_true.loc[common].to_numpy(), floor
        )
        har_losses = qlike_series(
            har_rows.loc[common, "y_pred"].to_numpy(),
            har_rows.loc[common, "y_true"].to_numpy(),
            floor,
        )
        metrics["dm"][f"lstm_aux_vs_har_{TV}_h{horizon}"] = _dm_row(
            aux_losses,
            har_losses,
            horizon,
            seeds,
            {s: result.preds_per_seed[s].reindex(common) for s in seeds},
            y_true.reindex(common),
            floor,
        )

        # ---- DM: aux vs rv-only (the "source" cut -- aux features vs architecture) ----
        rvonly_rows = stage1.preds[
            (stage1.preds["model"] == "lstm_rvonly")
            & (stage1.preds["horizon"] == horizon)
            & (stage1.preds["seed"] == -1)
        ].set_index("origin_date")
        common2 = ensemble.index.intersection(rvonly_rows.index)
        aux_losses2 = qlike_series(
            ensemble.loc[common2].to_numpy(), y_true.loc[common2].to_numpy(), floor
        )
        rvonly_losses2 = qlike_series(
            rvonly_rows.loc[common2, "y_pred"].to_numpy(),
            rvonly_rows.loc[common2, "y_true"].to_numpy(),
            floor,
        )
        rvonly_per_seed = stage1.preds[
            (stage1.preds["model"] == "lstm_rvonly")
            & (stage1.preds["horizon"] == horizon)
            & (stage1.preds["seed"] != -1)
        ]
        res2 = dm_test(aux_losses2, rvonly_losses2, h=horizon)
        ensemble_sign2 = np.sign(res2.mean_loss_diff)
        same_sign2 = 0
        for seed in seeds:
            seed_rv = rvonly_per_seed[rvonly_per_seed["seed"] == seed].set_index(
                "origin_date"
            )
            c = common2.intersection(seed_rv.index)
            seed_aux = result.preds_per_seed[seed].reindex(c)
            seed_aux_losses = qlike_series(
                seed_aux.to_numpy(), y_true.reindex(c).to_numpy(), floor
            )
            seed_rv_losses = qlike_series(
                seed_rv.loc[c, "y_pred"].to_numpy(), seed_rv.loc[c, "y_true"].to_numpy(), floor
            )
            diff = float(np.mean(seed_aux_losses - seed_rv_losses))
            if np.sign(diff) == ensemble_sign2:
                same_sign2 += 1
        metrics["dm"][f"lstm_aux_vs_lstm_rvonly_{TV}_h{horizon}"] = {
            "stat": res2.stat,
            "p_value": res2.p_value,
            "hac_lag": res2.hac_lag,
            "mean_loss_diff": res2.mean_loss_diff,
            "sign_agreement_frac": same_sign2 / len(seeds),
            "n": res2.n,
        }

    preds = pd.concat(pred_frames).reset_index(drop=True)
    run_cfg = dict(cfg)
    run_cfg["experiment"] = AUX_EXPERIMENT
    run_cfg["target"] = TV
    run_cfg["seeds_used"] = seeds
    run_cfg["n_folds"] = len(folds)
    run_cfg["snapshot_files"] = snap.manifest["files"]
    run_cfg["reused_stage1_run"] = stage1_dir.name
    run_dir = io_mod.save_run(AUX_EXPERIMENT, run_cfg, preds, metrics)
    print(f"\nsaved run: {run_dir}")
    return run_dir


def replay_aux_compare(run_dir: Path | None = None) -> None:
    if run_dir is None:
        run_dir = _latest_run(AUX_EXPERIMENT)
    problems = io_mod.validate_run(run_dir)
    if problems:
        sys.exit(f"run failed validation: {problems}")
    arts = io_mod.load_run(run_dir)

    print(f"\n=== Stage 2 aux-compare headline (pure replay of {run_dir.name}) ===")
    for horizon in ("1", "5"):
        print(f"\n{TV} h={horizon}   (QLIKE lower is better)")
        rows = []
        for model, per_target in arts.metrics["models"].items():
            leaf = per_target[TV][horizon]
            dm_har = arts.metrics["dm"].get(f"{model}_vs_har_{TV}_h{horizon}", {})
            dm_rvonly = arts.metrics["dm"].get(
                f"{model}_vs_lstm_rvonly_{TV}_h{horizon}", {}
            )
            rows.append(
                {
                    "model": model,
                    "qlike": leaf["qlike"],
                    "rmse": leaf["rmse"],
                    "bind%": 100 * leaf["bind_rate"],
                    "n": leaf["n"],
                    "DM vs HAR": dm_har.get("stat"),
                    "p vs HAR": dm_har.get("p_value"),
                    "seed agree% (HAR)": (
                        100 * dm_har["sign_agreement_frac"]
                        if dm_har.get("sign_agreement_frac") is not None
                        else None
                    ),
                    "DM vs RVonly": dm_rvonly.get("stat"),
                    "p vs RVonly": dm_rvonly.get("p_value"),
                }
            )
        table = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}", na_rep="—"))
    print(
        "\nEdge claim requires DM p<0.05 on the ensemble-mean AND same-sign loss "
        "differential in >=80% of seeds (manifesto S6)."
    )


# --------------------------------------------------------------------------- decay


def run_decay(seeds: list[int], folds_limit: int | None = None) -> Path:
    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    targets = build_targets(snap)
    split_cfg = _split_cfg(cfg)

    metrics: dict = {"models": {}, "dm": {}, "window_lengths_td": {}}
    pred_frames = []
    horizon = 1
    floor = cfg["floors"][TV]

    for wl in WINDOW_LENGTHS_TD:
        folds = retrain_folds(targets.index, split_cfg, wl)
        resolved_wl = len(folds[0].fit_idx) if folds else wl
        if folds_limit is not None:
            folds = folds[:folds_limit]
        label = f"wl{resolved_wl}"
        print(f"--- window_len={resolved_wl}td (h=1, TV, RV-only) ---", flush=True)
        metrics["window_lengths_td"][label] = resolved_wl

        har = forecast_classical(
            "har", targets, None, folds, target=TV, horizon=horizon, cfg=cfg
        )
        q_har = qlike(har["y_pred"].to_numpy(), har["y_true"].to_numpy(), floor)
        har_model = f"har_{label}"
        metrics["models"].setdefault(har_model, {}).setdefault(TV, {})[str(horizon)] = {
            "qlike": q_har.value,
            "rmse": rmse(har["y_pred"].to_numpy(), har["y_true"].to_numpy()),
            "bind_rate": q_har.bind_rate,
            "n": q_har.n,
        }
        pred_frames.append(
            har.reset_index()
            .rename(columns={"index": "origin_date"})
            .assign(target=TV, horizon=horizon, model=har_model, seed=-1)[
                ["origin_date", "target", "horizon", "model", "seed", "y_pred", "y_true"]
            ]
        )

        features, result, ensemble, y_true = _train_net(
            cfg, snap, targets, folds, TV, horizon, RVONLY_TIERS, seeds
        )
        net_model = f"lstm_rvonly_{label}"
        leaf = _net_leaf(ensemble, y_true, floor, result.loss_used, net_model, TV, horizon)
        metrics["models"].setdefault(net_model, {}).setdefault(TV, {})[str(horizon)] = leaf
        pred_frames.append(_pred_frame(ensemble, y_true, TV, horizon, net_model, -1))
        for seed in seeds:
            s = result.preds_per_seed[seed]
            pred_frames.append(
                pd.DataFrame(
                    {
                        "origin_date": s.index,
                        "target": TV,
                        "horizon": horizon,
                        "model": net_model,
                        "seed": seed,
                        "y_pred": s.to_numpy(),
                        "y_true": features.y.reindex(s.index).to_numpy(),
                    }
                )
            )

        common = ensemble.index.intersection(har.index)
        net_losses = qlike_series(
            ensemble.loc[common].to_numpy(), y_true.loc[common].to_numpy(), floor
        )
        har_losses = qlike_series(
            har.loc[common, "y_pred"].to_numpy(), har.loc[common, "y_true"].to_numpy(), floor
        )
        dm_row = _dm_row(
            net_losses,
            har_losses,
            horizon,
            seeds,
            {s: result.preds_per_seed[s].reindex(common) for s in seeds},
            y_true.reindex(common),
            floor,
        )
        metrics["dm"][f"{net_model}_vs_{har_model}_{TV}_h{horizon}"] = dm_row

    preds = pd.concat(pred_frames).reset_index(drop=True)
    run_cfg = dict(cfg)
    run_cfg["experiment"] = DECAY_EXPERIMENT
    run_cfg["target"] = TV
    run_cfg["seeds_used"] = seeds
    run_cfg["window_lengths_td"] = metrics["window_lengths_td"]
    run_cfg["snapshot_files"] = snap.manifest["files"]
    run_dir = io_mod.save_run(DECAY_EXPERIMENT, run_cfg, preds, metrics)
    print(f"\nsaved run: {run_dir}")
    return run_dir


def replay_decay(run_dir: Path | None = None) -> None:
    if run_dir is None:
        run_dir = _latest_run(DECAY_EXPERIMENT)
    problems = io_mod.validate_run(run_dir)
    if problems:
        sys.exit(f"run failed validation: {problems}")
    arts = io_mod.load_run(run_dir)

    print(f"\n=== Stage 2 decay curve (pure replay of {run_dir.name}) ===")
    rows = []
    for label, wl in sorted(arts.metrics["window_lengths_td"].items(), key=lambda kv: kv[1]):
        net_leaf = arts.metrics["models"][f"lstm_rvonly_{label}"][TV]["1"]
        har_leaf = arts.metrics["models"][f"har_{label}"][TV]["1"]
        dm = arts.metrics["dm"].get(f"lstm_rvonly_{label}_vs_har_{label}_{TV}_h1", {})
        rows.append(
            {
                "window_td": wl,
                "lstm_qlike": net_leaf["qlike"],
                "har_qlike": har_leaf["qlike"],
                "DM stat": dm.get("stat"),
                "p vs HAR": dm.get("p_value"),
                "seed agree%": (
                    100 * dm["sign_agreement_frac"]
                    if dm.get("sign_agreement_frac") is not None
                    else None
                ),
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}", na_rep="—"))


# --------------------------------------------------------------------------- final-oc


def run_final_oc(seeds: list[int], folds_limit: int | None = None) -> Path:
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
        print(f"--- OC h={horizon} ---", flush=True)
        floor = cfg["floors"][OC]

        print("  scoring har ...", flush=True)
        har = forecast_classical(
            "har", targets, None, folds, target=OC, horizon=horizon, cfg=cfg
        )
        q = qlike(har["y_pred"].to_numpy(), har["y_true"].to_numpy(), floor)
        metrics["models"].setdefault("har", {}).setdefault(OC, {})[str(horizon)] = {
            "qlike": q.value,
            "rmse": rmse(har["y_pred"].to_numpy(), har["y_true"].to_numpy()),
            "bind_rate": q.bind_rate,
            "n": q.n,
        }
        pred_frames.append(
            har.reset_index()
            .rename(columns={"index": "origin_date"})
            .assign(target=OC, horizon=horizon, model="har", seed=-1)[
                ["origin_date", "target", "horizon", "model", "seed", "y_pred", "y_true"]
            ]
        )

        print("  training lstm_aux (RV+aux: tiers=('t1','t2','t3')) ...", flush=True)
        features, result, ensemble, y_true = _train_net(
            cfg, snap, targets, folds, OC, horizon, AUX_TIERS, seeds
        )
        leaf = _net_leaf(ensemble, y_true, floor, result.loss_used, "lstm_aux", OC, horizon)
        metrics["models"].setdefault("lstm_aux", {}).setdefault(OC, {})[str(horizon)] = leaf

        for seed in seeds:
            s = result.preds_per_seed[seed]
            pred_frames.append(
                pd.DataFrame(
                    {
                        "origin_date": s.index,
                        "target": OC,
                        "horizon": horizon,
                        "model": "lstm_aux",
                        "seed": seed,
                        "y_pred": s.to_numpy(),
                        "y_true": features.y.reindex(s.index).to_numpy(),
                    }
                )
            )
        pred_frames.append(_pred_frame(ensemble, y_true, OC, horizon, "lstm_aux", -1))

        common = ensemble.index.intersection(har.index)
        net_losses = qlike_series(
            ensemble.loc[common].to_numpy(), y_true.loc[common].to_numpy(), floor
        )
        har_losses = qlike_series(
            har.loc[common, "y_pred"].to_numpy(), har.loc[common, "y_true"].to_numpy(), floor
        )
        metrics["dm"][f"lstm_aux_vs_har_{OC}_h{horizon}"] = _dm_row(
            net_losses,
            har_losses,
            horizon,
            seeds,
            {s: result.preds_per_seed[s].reindex(common) for s in seeds},
            y_true.reindex(common),
            floor,
        )

    preds = pd.concat(pred_frames).reset_index(drop=True)
    run_cfg = dict(cfg)
    run_cfg["experiment"] = FINAL_OC_EXPERIMENT
    run_cfg["target"] = OC
    run_cfg["seeds_used"] = seeds
    run_cfg["n_folds"] = len(folds)
    run_cfg["snapshot_files"] = snap.manifest["files"]
    run_dir = io_mod.save_run(FINAL_OC_EXPERIMENT, run_cfg, preds, metrics)
    print(f"\nsaved run: {run_dir}")
    return run_dir


def replay_final_oc(run_dir: Path | None = None) -> None:
    if run_dir is None:
        run_dir = _latest_run(FINAL_OC_EXPERIMENT)
    problems = io_mod.validate_run(run_dir)
    if problems:
        sys.exit(f"run failed validation: {problems}")
    arts = io_mod.load_run(run_dir)

    print(f"\n=== Stage 2 final OC eval (pure replay of {run_dir.name}) ===")
    for horizon in ("1", "5"):
        print(f"\n{OC} h={horizon}   (QLIKE lower is better)")
        rows = []
        for model, per_target in arts.metrics["models"].items():
            leaf = per_target[OC][horizon]
            dm = arts.metrics["dm"].get(f"{model}_vs_har_{OC}_h{horizon}", {})
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

    p_aux = sub.add_parser("aux-compare", help="RV+aux net vs HAR (and vs RV-only) on TV")
    p_aux.add_argument("--seeds", choices=["dev", "headline"], default="headline")
    p_aux.add_argument("--replay", action="store_true")
    p_aux.add_argument("--smoke", action="store_true", help="tiny run: 2 folds, dev seeds")

    p_decay = sub.add_parser("decay", help="window-length sensitivity curve, TV h=1, RV-only")
    p_decay.add_argument("--seeds", choices=["dev", "headline"], default="dev")
    p_decay.add_argument("--replay", action="store_true")
    p_decay.add_argument("--smoke", action="store_true", help="tiny run: 2 folds, dev seeds")

    p_final = sub.add_parser("final-oc", help="RV+aux net vs HAR on OC, both horizons")
    p_final.add_argument("--seeds", choices=["dev", "headline"], default="headline")
    p_final.add_argument("--replay", action="store_true")
    p_final.add_argument("--smoke", action="store_true", help="tiny run: 2 folds, dev seeds")

    args = ap.parse_args()
    cfg = io_mod.load_config()

    if args.cmd == "aux-compare":
        if args.replay:
            replay_aux_compare()
        else:
            seeds = cfg["seeds"]["dev"] if args.smoke else cfg["seeds"][args.seeds]
            folds_limit = 2 if args.smoke else None
            replay_aux_compare(run_aux_compare(seeds, folds_limit))
    elif args.cmd == "decay":
        if args.replay:
            replay_decay()
        else:
            seeds = cfg["seeds"]["dev"] if args.smoke else cfg["seeds"][args.seeds]
            folds_limit = 2 if args.smoke else None
            replay_decay(run_decay(seeds, folds_limit))
    else:  # final-oc
        if args.replay:
            replay_final_oc()
        else:
            seeds = cfg["seeds"]["dev"] if args.smoke else cfg["seeds"][args.seeds]
            folds_limit = 2 if args.smoke else None
            replay_final_oc(run_final_oc(seeds, folds_limit))
