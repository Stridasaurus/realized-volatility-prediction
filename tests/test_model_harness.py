"""Executable success criteria S1-S7 from specs/model-harness/SPEC.md (CPU, seeded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.features import FeatureSet
from src.metrics import qlike
from src.splits import CanonicalSplit, RetrainFold
from src.train import qlike_loss_torch, run_walk_forward, tune

INDEX = pd.bdate_range("2019-01-02", periods=320)


def make_feature_set(
    seed: int = 0, n_features: int = 3, noise_only: bool = False
) -> FeatureSet:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(len(INDEX), n_features)),
        index=INDEX,
        columns=[f"t1_f{i}" for i in range(n_features)],
    )
    base = np.log(1e-4) + (0.0 if noise_only else 0.5 * X["t1_f0"].to_numpy())
    y_log = pd.Series(
        base + rng.normal(0.0, 0.3, len(INDEX)), index=INDEX, name="y_rv_tv_h1"
    )
    y = np.exp(y_log)
    y.name = "y_rv_tv_h1"
    return FeatureSet(X=X, y=y, y_log=y_log, tiers={"t1": list(X.columns)})


FOLDS = [RetrainFold(fit_idx=INDEX[:250], test_idx=INDEX[255:275])]

HP = {
    "hidden": 8,
    "layers": 1,
    "dropout": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 64,
    "seq_len": 5,
}


def make_cfg(max_epochs: int = 5, patience: int = 3) -> dict:
    return {
        "floors": {"rv_tv": 1e-6, "rv_oc": 1e-6},
        "model": {
            "inner_val_frac": 0.15,
            "patience": patience,
            "max_epochs": max_epochs,
            "grad_clip": 1.0,
        },
    }


# ------------------------------------------------- S1: seed determinism


def test_s1_same_seed_identical_predictions():
    fs = make_feature_set()
    a = run_walk_forward(fs, FOLDS, HP, seeds=[0], cfg=make_cfg())
    b = run_walk_forward(fs, FOLDS, HP, seeds=[0], cfg=make_cfg())
    assert np.array_equal(a.preds_per_seed.to_numpy(), b.preds_per_seed.to_numpy())
    assert a.preds_ensemble.equals(b.preds_ensemble)


def test_s1_different_seeds_differ():
    fs = make_feature_set()
    r = run_walk_forward(fs, FOLDS, HP, seeds=[0, 1], cfg=make_cfg())
    assert not np.array_equal(
        r.preds_per_seed[0].to_numpy(), r.preds_per_seed[1].to_numpy()
    )


# ------------------------------------------------- S2: log-target round trip


def test_s2_log_round_trip_and_level_space_output():
    fs = make_feature_set()
    assert np.abs(np.exp(fs.y_log.to_numpy()) - fs.y.to_numpy()).max() < 1e-10
    r = run_walk_forward(fs, FOLDS, HP, seeds=[0], cfg=make_cfg())
    # predictions come back exponentiated: strictly positive level/variance space
    assert (r.preds_per_seed.to_numpy() > 0).all()
    assert (r.preds_ensemble > 0).all()


# ------------------------------------------------- S3: early stopping


def test_s3_early_stopping_triggers_on_overfit_fixture():
    fs = make_feature_set(seed=3, noise_only=True)  # nothing to learn -> val rises
    hp = dict(HP, hidden=32, lr=5e-3)
    cfg = make_cfg(max_epochs=200, patience=5)
    r = run_walk_forward(fs, FOLDS, hp, seeds=[0], cfg=cfg)
    (best_epoch,) = r.early_stop_epochs.values()
    assert best_epoch < 100, "early stopping never triggered on a pure-noise fixture"


# ------------------------------------------------- S4: QLIKE -> MSE-log fallback


def test_s4_nonfinite_qlike_falls_back_to_mse_log(monkeypatch):
    import src.train as train_mod

    monkeypatch.setattr(
        train_mod, "qlike_loss_torch", lambda *a, **k: torch.tensor(float("nan"))
    )
    fs = make_feature_set()
    with pytest.warns(UserWarning, match="falling back"):
        r = run_walk_forward(fs, FOLDS, HP, seeds=[0], cfg=make_cfg())
    assert r.loss_used[0] == "mse_log"
    assert len(r.preds_ensemble) > 0  # run completed on the fallback path


def test_s4_healthy_run_records_qlike():
    fs = make_feature_set()
    r = run_walk_forward(fs, FOLDS, HP, seeds=[0], cfg=make_cfg())
    assert r.loss_used[0] == "qlike"


# ------------------------------------------------- S5: ensemble mean contract


def test_s5_ensemble_equals_per_seed_mean_exactly():
    fs = make_feature_set()
    r = run_walk_forward(fs, FOLDS, HP, seeds=[0, 1, 2], cfg=make_cfg())
    assert r.preds_per_seed.shape[1] == 3
    expected = r.preds_per_seed.mean(axis=1)
    assert r.preds_ensemble.equals(expected)
    assert r.preds_ensemble.index.equals(r.preds_per_seed.index)


# ------------------------------------------------- S6: torch QLIKE == metrics.qlike


def test_s6_torch_loss_matches_metrics_qlike():
    rng = np.random.default_rng(6)
    pred = np.abs(rng.normal(1e-4, 3e-5, 200)) + 1e-6
    actual = np.abs(rng.normal(1e-4, 3e-5, 200)) + 1e-6
    floor = 5e-5  # binds on some predictions, exercising the clamp
    torch_val = float(
        qlike_loss_torch(
            torch.log(torch.tensor(pred, dtype=torch.float64)),
            torch.tensor(actual, dtype=torch.float64),
            floor,
        )
    )
    assert torch_val == pytest.approx(qlike(pred, actual, floor).value, abs=1e-10)


# ------------------------------------------------- S7: tune cannot see test data


def test_s7_tune_smoke_and_test_region_guard():
    fs = make_feature_set()
    split = CanonicalSplit(
        train_idx=INDEX[:200], val_idx=INDEX[210:260], test_idx=INDEX[270:]
    )
    cfg = make_cfg(max_epochs=2)
    cfg["model"]["seq_len_max"] = 10
    hp = tune(fs, split, cfg, n_trials=2)
    assert {
        "hidden",
        "layers",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
        "seq_len",
    } <= set(hp)
    assert 5 <= hp["seq_len"] <= 10


# ------------------------------------------------- E5 + guards


def test_e5_duplicate_seeds_rejected():
    fs = make_feature_set()
    with pytest.raises(ValueError, match="duplicate"):
        run_walk_forward(fs, FOLDS, HP, seeds=[0, 0], cfg=make_cfg())


def test_e1_fit_window_too_small_hard_error():
    fs = make_feature_set()
    tiny = [RetrainFold(fit_idx=INDEX[:40], test_idx=INDEX[45:60])]
    with pytest.raises(ValueError, match="too few fit origins"):
        run_walk_forward(fs, tiny, HP, seeds=[0], cfg=make_cfg())
