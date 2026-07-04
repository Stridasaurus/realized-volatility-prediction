"""Executable success criteria S1-S6 from specs/io/SPEC.md.

Import discipline: src/io.py shadows stdlib io — always `from src import io as io_mod`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import io as io_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preds() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=5)
    rows = []
    for seed in (0, 1):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "origin_date": d,
                    "target": "rv_tv",
                    "horizon": 1,
                    "model": "lstm",
                    "seed": seed,
                    "y_pred": 1e-4 * (1 + i + seed),
                    "y_true": 1e-4 * (1 + i),
                }
            )
    frame = pd.DataFrame(rows)
    ens = frame.groupby("origin_date", as_index=False).agg(
        {
            "target": "first",
            "horizon": "first",
            "model": "first",
            "y_true": "first",
            "y_pred": "mean",
        }
    )
    ens["seed"] = -1
    return pd.concat([frame, ens[frame.columns]], ignore_index=True)


@pytest.fixture
def metrics() -> dict:
    leaf = {"qlike": 0.31, "rmse": 2e-4, "bind_rate": 0.0, "n": 5}
    dm = {
        "stat": -1.2,
        "p_value": 0.23,
        "hac_lag": 4,
        "mean_loss_diff": -0.01,
        "sign_agreement_frac": 0.9,
    }
    return {
        "models": {"lstm": {"rv_tv": {"1": leaf}}},
        "dm": {"lstm_vs_har_rv_tv_h1": dm},
    }


@pytest.fixture
def config() -> dict:
    return {
        "splits": {"embargo_days": 66},
        "floors": {"rv_tv": 1e-6, "rv_oc": 1e-6},
        "seeds": [0, 1],
    }


# ------------------------------------------------- S1: round trip


def test_s1_save_load_round_trip(tmp_path, config, preds, metrics):
    run_dir = io_mod.save_run("exp_smoke", config, preds, metrics, results_dir=tmp_path)
    arts = io_mod.load_run(run_dir)
    assert arts.metrics == metrics
    for k, v in config.items():
        assert arts.config[k] == v
    assert "provenance" in arts.config  # versions + git rev embedded (R3)
    assert set(io_mod._TRACKED_PACKAGES) <= set(arts.config["provenance"]["versions"])
    got = arts.preds.sort_values(["seed", "origin_date"]).reset_index(drop=True)
    want = preds.sort_values(["seed", "origin_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want[got.columns], check_dtype=True)


def test_s1_json_written_deterministically(tmp_path, config, preds, metrics):
    run_dir = io_mod.save_run("exp_det", config, preds, metrics, results_dir=tmp_path)
    text = (run_dir / "metrics.json").read_text(encoding="utf-8")
    keys = list(json.loads(text))
    assert keys == sorted(keys)


# ------------------------------------------------- S2: schema validation


def test_s2_good_run_validates_clean(tmp_path, config, preds, metrics):
    run_dir = io_mod.save_run("exp_ok", config, preds, metrics, results_dir=tmp_path)
    assert io_mod.validate_run(run_dir) == []


def test_s2_each_missing_piece_is_caught(tmp_path, config, preds, metrics):
    for fname in ("config.json", "preds.parquet", "metrics.json"):
        run_dir = io_mod.save_run(
            f"exp_missing_{fname}", config, preds, metrics, results_dir=tmp_path
        )
        (run_dir / fname).unlink()
        problems = io_mod.validate_run(run_dir)
        assert any(fname in p for p in problems)

    # metric leaf key deletion
    bad = {"models": {"lstm": {"rv_tv": {"1": {"qlike": 0.3, "rmse": 1.0, "n": 5}}}}}
    run_dir = io_mod.save_run("exp_leaf", config, preds, bad, results_dir=tmp_path)
    assert any("bind_rate" in p for p in io_mod.validate_run(run_dir))

    run_dir2 = io_mod.save_run(
        "exp_nomodels", config, preds, {"dm": {}}, results_dir=tmp_path
    )
    assert any("models" in p for p in io_mod.validate_run(run_dir2))


def test_s2_missing_pred_column_rejected_at_save(tmp_path, config, preds, metrics):
    with pytest.raises(ValueError, match="seed"):
        io_mod.save_run(
            "exp_col", config, preds.drop(columns="seed"), metrics, results_dir=tmp_path
        )


# ------------------------------------------------- S3: ensemble consistency


def test_s3_corrupted_ensemble_block_caught(tmp_path, config, preds, metrics):
    bad = preds.copy()
    ens_rows = bad.index[bad["seed"] == -1]
    bad.loc[ens_rows[0], "y_pred"] *= 1.5
    run_dir = io_mod.save_run("exp_ens", config, bad, metrics, results_dir=tmp_path)
    assert any("ensemble" in p for p in io_mod.validate_run(run_dir))


def test_s3_missing_ensemble_block_caught(tmp_path, config, preds, metrics):
    no_ens = preds[preds["seed"] != -1]
    run_dir = io_mod.save_run(
        "exp_noens", config, no_ens, metrics, results_dir=tmp_path
    )
    assert any("ensemble" in p for p in io_mod.validate_run(run_dir))


# ------------------------------------------------- S4: embargo auto-resolution


def test_s4_embargo_auto_resolves_to_max_lookback(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    assert cfg["splits"]["embargo_days"] == "auto"
    resolved = io_mod.load_config()
    assert resolved["floors"]["rv_tv"] > 0  # floors written at freeze
    assert resolved["floors"]["rv_oc"] > 0
    assert resolved["splits"]["embargo_days"] == max(
        cfg["lookbacks"]["har_monthly_lag"],
        cfg["model"]["seq_len_max"],
        cfg["lookbacks"]["max_aux_window"],
    )

    cfg["model"]["seq_len_max"] = None
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="missing component"):
        io_mod.load_config(p)


def test_s4_explicit_embargo_passes_through(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    cfg["splits"]["embargo_days"] = 30
    cfg["floors"] = {"rv_tv": 1e-6, "rv_oc": 1e-6}
    p = tmp_path / "explicit.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert io_mod.load_config(p)["splits"]["embargo_days"] == 30

    cfg["splits"]["embargo_days"] = -3
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="embargo"):
        io_mod.load_config(p)


# ------------------------------------------------- S5: append-only + atomicity


def test_s5_overwrite_refusal(tmp_path, config, preds, metrics, monkeypatch):
    monkeypatch.setattr(io_mod, "_run_id", lambda config: "fixed_run_id")
    io_mod.save_run("exp_dup", config, preds, metrics, results_dir=tmp_path)
    with pytest.raises(FileExistsError, match="append-only"):
        io_mod.save_run("exp_dup", config, preds, metrics, results_dir=tmp_path)


def test_s5_leftover_tmp_blocks_save_and_flags_validation(
    tmp_path, config, preds, metrics, monkeypatch
):
    monkeypatch.setattr(io_mod, "_run_id", lambda config: "fixed_run_id")
    tmp_dir = tmp_path / "exp_tmp" / "fixed_run_id.tmp"
    tmp_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="partial"):
        io_mod.save_run("exp_tmp", config, preds, metrics, results_dir=tmp_path)
    problems = io_mod.validate_run(tmp_path / "exp_tmp" / "fixed_run_id")
    assert any(".tmp" in p for p in problems)


# ------------------------------------------------- S6: stdlib-shadowing guard


def test_s6_no_sys_path_injection_of_src_anywhere():
    """src/ on sys.path would shadow stdlib io for the whole interpreter."""
    pattern = re.compile(r"sys\.path\.(?:append|insert)\([^)]*src")
    offenders = []
    for p in REPO_ROOT.rglob("*"):
        if p.suffix not in {".py", ".ipynb"} or "tests" in p.parts:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(p.relative_to(REPO_ROOT)))
    assert offenders == []


def test_s6_import_discipline_in_repo_modules():
    """No module may do `import io` expecting the project module, or vice versa."""
    for p in (REPO_ROOT / "src").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"^import io$", text, re.M), f"{p.name} imports bare io"
        assert not re.search(r"^from io import", text, re.M), f"{p.name} bare-from io"
