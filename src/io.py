"""Save/load contract for every run + config handling.

Spec: specs/io/SPEC.md. NOTE: this module shadows stdlib ``io`` by name (manifesto-pinned
filename). Always import it as ``from src import io`` / ``import src.io``; NEVER add
``src/`` itself to sys.path (a guard test enforces this repo-wide).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as _md
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
RESULTS_DIR = REPO_ROOT / "results"

_PRED_COLS = ["origin_date", "target", "horizon", "model", "seed", "y_pred", "y_true"]
_METRIC_LEAF_KEYS = {"qlike", "rmse", "bind_rate", "n"}
_DM_KEYS = {"stat", "p_value", "hac_lag", "mean_loss_diff", "sign_agreement_frac"}
_TRACKED_PACKAGES = ["torch", "numpy", "pandas", "statsmodels", "arch", "optuna"]


@dataclass(frozen=True)
class RunArtifacts:
    config: dict
    preds: pd.DataFrame
    metrics: dict


def _resolve_config(cfg: dict) -> dict:
    emb = cfg["splits"]["embargo_days"]
    if emb == "auto":
        parts = {
            "har_monthly_lag": cfg["lookbacks"]["har_monthly_lag"],
            "model.seq_len_max": cfg["model"]["seq_len_max"],
            "max_aux_window": cfg["lookbacks"]["max_aux_window"],
        }
        missing = [k for k, v in parts.items() if v is None]
        if missing:
            raise ValueError(f"embargo auto-resolution missing component(s): {missing}")
        cfg["splits"]["embargo_days"] = int(max(parts.values()))
    elif not isinstance(emb, int) or emb <= 0:
        raise ValueError(f"embargo_days must be 'auto' or a positive int, got {emb!r}")
    if cfg["floors"]["rv_tv"] is None or cfg["floors"]["rv_oc"] is None:
        warnings.warn("QLIKE floors are unset — run scripts/freeze_snapshot.py before scoring",
                      UserWarning)
    return cfg


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = ["snapshot", "splits", "lookbacks", "targets", "horizons", "floors",
                "calibration_band", "ewma", "model", "seeds"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config missing required key(s): {missing}")
    unknown = [k for k in cfg if k not in required + ["project"]]
    if unknown:
        warnings.warn(f"config has unknown key(s) {unknown} — ignored", UserWarning)
    return _resolve_config(cfg)


def _versions() -> dict:
    out = {}
    for p in _TRACKED_PACKAGES:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            out[p] = "absent"
    return out


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, timeout=10,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _run_id(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{hashlib.sha256(blob).hexdigest()[:8]}"


def save_run(experiment: str, config: dict, preds: pd.DataFrame, metrics: dict,
             results_dir: Path = RESULTS_DIR) -> Path:
    missing = [c for c in _PRED_COLS if c not in preds.columns]
    if missing:
        raise ValueError(f"preds missing required column(s): {missing}")

    config = dict(config)
    config.setdefault("provenance", {})
    config["provenance"].update({"versions": _versions(), "git_rev": _git_rev(),
                                 "saved_utc": datetime.now(timezone.utc).isoformat()})

    run_dir = Path(results_dir) / experiment / _run_id(config)
    if run_dir.exists():
        raise FileExistsError(f"run dir already exists (results are append-only): {run_dir}")
    tmp = run_dir.with_suffix(".tmp")
    if tmp.exists():
        raise FileExistsError(f"leftover partial write: {tmp} — inspect and remove manually")
    tmp.mkdir(parents=True)

    p = preds.copy()
    p["origin_date"] = pd.to_datetime(p["origin_date"]).dt.tz_localize(None)
    (tmp / "config.json").write_text(
        json.dumps(config, sort_keys=True, indent=2, default=str), encoding="utf-8")
    p.to_parquet(tmp / "preds.parquet", index=False)
    (tmp / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, indent=2, default=str), encoding="utf-8")
    tmp.rename(run_dir)
    return run_dir


def load_run(run_dir: Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    problems = validate_run(run_dir)
    if problems:
        raise ValueError(f"invalid run at {run_dir}: {problems}")
    return RunArtifacts(
        config=json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        preds=pd.read_parquet(run_dir / "preds.parquet"),
        metrics=json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
    )


def validate_run(run_dir: Path) -> list[str]:
    run_dir = Path(run_dir)
    problems: list[str] = []
    if run_dir.with_suffix(".tmp").exists():
        problems.append("leftover .tmp dir from a partial write")
    for fname in ("config.json", "preds.parquet", "metrics.json"):
        if not (run_dir / fname).exists():
            problems.append(f"missing {fname}")
    if problems:
        return problems

    preds = pd.read_parquet(run_dir / "preds.parquet")
    missing = [c for c in _PRED_COLS if c not in preds.columns]
    if missing:
        problems.append(f"preds missing column(s): {missing}")
        return problems

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if "models" not in metrics:
        problems.append("metrics.json missing 'models' key")
    else:
        for model, per_target in metrics["models"].items():
            for target, per_h in per_target.items():
                for h, leaf in per_h.items():
                    missing_k = _METRIC_LEAF_KEYS - set(leaf)
                    if missing_k:
                        problems.append(f"metrics[{model}][{target}][{h}] missing {sorted(missing_k)}")
    if "dm" in metrics:
        for name, leaf in metrics["dm"].items():
            missing_k = _DM_KEYS - set(leaf)
            if missing_k:
                problems.append(f"dm[{name}] missing {sorted(missing_k)}")

    # Ensemble consistency: seed == -1 rows must equal the per-seed mean (1e-12).
    for keys, grp in preds.groupby(["model", "target", "horizon"]):
        seeds = set(grp["seed"].unique())
        real = sorted(s for s in seeds if s != -1)
        if len(real) > 1:
            if -1 not in seeds:
                problems.append(f"{keys}: multi-seed group lacks ensemble (seed == -1) block")
                continue
            wide = grp[grp["seed"] != -1].pivot(index="origin_date", columns="seed",
                                                values="y_pred")
            ens = grp[grp["seed"] == -1].set_index("origin_date")["y_pred"]
            diff = (wide.mean(axis=1) - ens.reindex(wide.index)).abs().max()
            if not (diff <= 1e-12):
                problems.append(f"{keys}: ensemble block != per-seed mean (max diff {diff})")
    return problems
