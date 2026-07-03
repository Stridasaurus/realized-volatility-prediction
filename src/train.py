"""Seeded, leak-safe train/eval harness for the small LSTM.

Spec: specs/model-harness/SPEC.md. Owns sequence windowing, the seeded training loop
(early stopping, QLIKE-with-floor loss + MSE-of-log fallback), the walk-forward retrain
protocol, multi-seed runs, and the seed-ensemble mean prediction sequence (the DM object).
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import FeatureSet, TrainOnlyScaler
from src.models import SmallLSTM
from src.splits import CanonicalSplit, RetrainFold

_DIVERGE_FACTOR = 10.0
_DIVERGE_EPOCHS = 3


@dataclass(frozen=True)
class TrainResult:
    preds_per_seed: pd.DataFrame     # index origin dates; one column per seed (variance space)
    preds_ensemble: pd.Series        # per-origin mean over seeds (the DM object)
    loss_used: dict                  # {seed: "qlike" | "mse_log"} (worst case across folds)
    early_stop_epochs: dict          # {(fold_id, seed): epoch}


class _Diverged(RuntimeError):
    pass


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def qlike_loss_torch(pred_log: torch.Tensor, y_level: torch.Tensor,
                     floor: float) -> torch.Tensor:
    """QLIKE-with-floor on exp(pred_log) vs the level target — mirrors metrics.qlike."""
    pred = torch.exp(pred_log).clamp_min(floor)
    r = y_level / pred
    return (r - torch.log(r) - 1.0).mean()


class WindowDataset(Dataset):
    """(seq_len, n_features) windows ending at each origin — rows <= origin only."""

    def __init__(self, X: np.ndarray, y_log: np.ndarray, positions: np.ndarray,
                 seq_len: int) -> None:
        if (positions < seq_len - 1).any():
            raise ValueError("origin position(s) lack a full lookback window")
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y_log, dtype=torch.float32)
        self.positions = positions
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        p = self.positions[i]
        return self.X[p - self.seq_len + 1 : p + 1], self.y[p]


def _inner_split(origins: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(int(len(origins) * frac), 1)
    return origins[:-n_val], origins[-n_val:]


def _train_one(X: np.ndarray, y_log: np.ndarray, y_level: np.ndarray,
               fit_pos: np.ndarray, hp: dict, seed: int, floor: float, cfg: dict,
               loss_name: str) -> SmallLSTM:
    _seed_everything(seed)
    model = SmallLSTM(X.shape[1], hp["hidden"], hp["layers"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    tr_pos, va_pos = _inner_split(fit_pos, cfg["model"]["inner_val_frac"])
    ds = WindowDataset(X, y_log, tr_pos, hp["seq_len"])
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, generator=gen,
                    num_workers=0)
    va_ds = WindowDataset(X, y_log, va_pos, hp["seq_len"])
    xv = torch.stack([va_ds[i][0] for i in range(len(va_ds))])
    yv_log = torch.as_tensor(y_log[va_pos], dtype=torch.float32)
    yv_lvl = torch.as_tensor(y_level[va_pos], dtype=torch.float32)

    def _loss(pred_log, pos_mask_log, pos_mask_lvl):
        if loss_name == "qlike":
            return qlike_loss_torch(pred_log, pos_mask_lvl, floor)
        return torch.mean((pred_log - pos_mask_log) ** 2)

    best_val, best_state, best_epoch = float("inf"), None, -1
    init_val, bad_streak = None, 0
    patience = cfg["model"]["patience"]
    for epoch in range(cfg["model"]["max_epochs"]):
        model.train()
        for xb, yb_log in dl:
            opt.zero_grad()
            pred = model(xb)
            yb_lvl = torch.exp(yb_log)
            loss = _loss(pred, yb_log, yb_lvl)
            if not torch.isfinite(loss):
                raise _Diverged(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["grad_clip"])
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(_loss(model(xv), yv_log, yv_lvl))
        if not np.isfinite(val):
            bad_streak += 1
        else:
            if init_val is None:
                init_val = val
            bad_streak = bad_streak + 1 if val > _DIVERGE_FACTOR * init_val else 0
        if bad_streak >= _DIVERGE_EPOCHS:
            raise _Diverged(f"validation loss diverged at epoch {epoch}")

        if val < best_val:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model._early_stop_epoch = best_epoch  # type: ignore[attr-defined]
    return model


def run_walk_forward(features: FeatureSet, folds: list[RetrainFold], hp: dict,
                     seeds: list[int], cfg: dict) -> TrainResult:
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    floor = cfg["floors"][features.y.name.split("_h")[0].removeprefix("y_")]
    horizon = int(features.y.name.rsplit("_h", 1)[1])
    seq_len = hp["seq_len"]
    index = features.X.index

    per_seed: dict[int, list[pd.Series]] = {s: [] for s in seeds}
    loss_used = {s: "qlike" for s in seeds}
    early_stop: dict = {}

    for fold_id, fold in enumerate(folds):
        fit_idx = index.intersection(fold.fit_idx)
        test_idx = index.intersection(fold.test_idx)
        if len(test_idx) == 0:
            continue
        fit_pos_all = index.get_indexer(fit_idx)
        # drop last h origins (label window must end inside the fit window) and origins
        # lacking a full lookback
        fit_pos = fit_pos_all[: max(len(fit_pos_all) - horizon, 0)]
        fit_pos = fit_pos[fit_pos >= seq_len - 1]
        if len(fit_pos) < 50:
            raise ValueError(f"fold {fold_id}: too few fit origins ({len(fit_pos)})")

        scaler = TrainOnlyScaler().fit(features.X, index[fit_pos])
        Xs = scaler.transform(features.X).to_numpy(dtype=np.float64)
        y_log = features.y_log.to_numpy(dtype=np.float64)
        y_lvl = features.y.to_numpy(dtype=np.float64)
        test_pos = index.get_indexer(test_idx)
        test_pos = test_pos[test_pos >= seq_len - 1]

        for seed in seeds:
            try:
                model = _train_one(Xs, y_log, y_lvl, fit_pos, hp, seed, floor, cfg,
                                   loss_used[seed])
            except _Diverged as e:
                warnings.warn(f"fold {fold_id} seed {seed}: {e} — falling back to "
                              "MSE-of-log loss", UserWarning)
                loss_used[seed] = "mse_log"
                model = _train_one(Xs, y_log, y_lvl, fit_pos, hp, seed, floor, cfg,
                                   "mse_log")
            early_stop[(fold_id, seed)] = model._early_stop_epoch
            ds = WindowDataset(Xs, y_log, test_pos, seq_len)
            xb = torch.stack([ds[i][0] for i in range(len(ds))])
            model.eval()
            with torch.no_grad():
                pred_lvl = torch.exp(model(xb)).numpy().astype(np.float64)
            per_seed[seed].append(pd.Series(pred_lvl, index=index[test_pos]))

    preds = pd.DataFrame({s: pd.concat(chunks) for s, chunks in per_seed.items()})
    return TrainResult(
        preds_per_seed=preds,
        preds_ensemble=preds.mean(axis=1),
        loss_used=loss_used,
        early_stop_epochs=early_stop,
    )


def tune(features: FeatureSet, split: CanonicalSplit, cfg: dict, n_trials: int) -> dict:
    """Stage-1 Optuna search on train/val ONLY (structurally cannot see test data)."""
    import optuna

    embargo_guard = split.test_idx[0]
    usable = features.X.index[features.X.index < split.val_idx[-1]]
    assert usable.max() < embargo_guard, "tune() must never see test-region dates"

    X_idx = features.X.index
    train_pos = X_idx.get_indexer(X_idx.intersection(split.train_idx))
    val_pos = X_idx.get_indexer(X_idx.intersection(split.val_idx))

    def objective(trial: "optuna.Trial") -> float:
        hp = {
            "hidden": trial.suggest_int("hidden", 8, 128, log=True),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "seq_len": trial.suggest_int("seq_len", 5, cfg["model"]["seq_len_max"]),
        }
        floor = cfg["floors"][features.y.name.split("_h")[0].removeprefix("y_")]
        fit_pos = train_pos[train_pos >= hp["seq_len"] - 1]
        scaler = TrainOnlyScaler().fit(features.X, X_idx[fit_pos])
        Xs = scaler.transform(features.X).to_numpy(dtype=np.float64)
        y_log = features.y_log.to_numpy(dtype=np.float64)
        y_lvl = features.y.to_numpy(dtype=np.float64)
        try:
            model = _train_one(Xs, y_log, y_lvl, fit_pos, hp, seed=0, floor=floor,
                               cfg=cfg, loss_name="qlike")
        except _Diverged:
            return float("inf")
        vp = val_pos[val_pos >= hp["seq_len"] - 1]
        ds = WindowDataset(Xs, y_log, vp, hp["seq_len"])
        xb = torch.stack([ds[i][0] for i in range(len(ds))])
        with torch.no_grad():
            pred_log = model(xb)
        val = float(qlike_loss_torch(pred_log, torch.as_tensor(y_lvl[vp],
                    dtype=torch.float32), floor))
        return val

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)
