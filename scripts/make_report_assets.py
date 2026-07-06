"""Report assets: pre-registered regime cut + figures, replayed from committed artifacts.

Pure post-processing (manifesto S4): reads saved ``preds.parquet``/``metrics.json``
from results/, re-runs NO training, and touches no data outside the frozen test
predictions. Outputs:

  results/report/regime_cut.json   - per-regime QLIKE + DM (aux vs HAR, RV-only vs HAR)
  results/report/figures/fig1_headline_qlike.png
  results/report/figures/fig2_decay_curve.png
  results/report/figures/fig3_test_span_vol.png
  results/report/figures/fig4_cum_qlike_diff.png

Regime boundaries are calendar sub-periods matching the manifesto's own description
of the test span ("COVID, 2022 bear, calm"); they were not pre-registered as dates,
so the per-regime DM p-values are reported as descriptive, not confirmatory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.metrics import dm_test, qlike_series  # noqa: E402

RUNS = {
    "stage0": REPO / "results/stage0_classical/20260704T005951Z_2713c8da",
    "stage1": REPO / "results/stage1_net_vs_har/20260706T025918Z_ff2c8d44",
    "aux": REPO / "results/stage2_aux_compare/20260706T072223Z_5ec6ae6d",
    "decay": REPO / "results/stage2_decay/20260706T085826Z_db2a5ced",
    "oc": REPO / "results/stage2_final_oc/20260706T142051Z_d8c4a176",
}

REGIMES = {
    "covid_2020": ("2020-01-01", "2020-12-31"),
    "calm_2021": ("2021-01-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023_2026": ("2023-01-01", "2026-06-30"),
}

HAC_LAG = 7  # matches every committed DM run

# palette (dataviz reference instance, validated for the light surface)
C_HAR = "#2a78d6"
C_RVONLY = "#1baf7a"
C_AUX = "#c98500"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

LABELS = {"har": "HAR", "lstm_rvonly": "LSTM (RV-only)", "lstm_aux": "LSTM (RV+aux)"}


def load_preds(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "preds.parquet")
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def ensemble_mean(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Per-date seed-ensemble mean prediction per model (the object DM tests)."""
    sub = df[(df["target"] == target) & (df["horizon"] == horizon)]
    ens = (
        sub.groupby(["model", "origin_date"])
        .agg(y_pred=("y_pred", "mean"), y_true=("y_true", "first"))
        .reset_index()
    )
    return ens


def regime_mask(dates: pd.Series, regime: str) -> pd.Series:
    lo, hi = REGIMES[regime]
    return (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))


def per_seed_sign_agreement(
    df: pd.DataFrame, target: str, horizon: int, model: str, bench: str,
    floor: float, regime: str,
) -> float:
    """Fraction of seeds whose mean regime loss diff has the ensemble's sign (S6 style)."""
    sub = df[(df["target"] == target) & (df["horizon"] == horizon)]
    sub = sub[regime_mask(sub["origin_date"], regime)]
    b = sub[sub["model"] == bench].set_index("origin_date").sort_index()
    if (b["seed"] != b["seed"].iloc[0]).any():
        b = b[b["seed"] == b["seed"].iloc[0]]
    bench_loss = qlike_series(b["y_pred"].to_numpy(), b["y_true"].to_numpy(), floor)
    bench_loss = pd.Series(bench_loss, index=b.index)
    diffs = []
    per_seed = sub[(sub["model"] == model) & (sub["seed"] >= 0)]
    for seed, g in per_seed.groupby("seed"):
        g = g.set_index("origin_date").sort_index()
        loss = qlike_series(g["y_pred"].to_numpy(), g["y_true"].to_numpy(), floor)
        diffs.append(float(np.mean(loss - bench_loss.loc[g.index].to_numpy())))
    signs = np.sign(diffs)
    ens_sign = np.sign(np.mean(diffs))
    return float(np.mean(signs == ens_sign))


def regime_cut(floors: dict) -> dict:
    out = {"regimes": {k: list(v) for k, v in REGIMES.items()}, "hac_lag": HAC_LAG, "blocks": {}}
    blocks = [
        ("rv_tv", 1, "aux", ["lstm_aux", "lstm_rvonly"]),
        ("rv_tv", 5, "aux", ["lstm_aux", "lstm_rvonly"]),
        ("rv_oc", 1, "oc", ["lstm_aux"]),
        ("rv_oc", 5, "oc", ["lstm_aux"]),
    ]
    for target, h, run_key, models in blocks:
        df = load_preds(RUNS[run_key])
        ens = ensemble_mean(df, target, h)
        floor = floors[target]
        wide_p = ens.pivot(index="origin_date", columns="model", values="y_pred").sort_index()
        wide_t = ens.pivot(index="origin_date", columns="model", values="y_true").sort_index()
        y_true = wide_t.iloc[:, 0].to_numpy()
        losses = {
            m: qlike_series(wide_p[m].to_numpy(), y_true, floor) for m in wide_p.columns
        }
        block = {}
        for regime in REGIMES:
            mask = regime_mask(pd.Series(wide_p.index), regime).to_numpy()
            n = int(mask.sum())
            entry = {"n": n, "qlike": {m: float(np.mean(losses[m][mask])) for m in losses}}
            entry["dm_vs_har"] = {}
            for m in models:
                r = dm_test(losses[m][mask], losses["har"][mask], h=h, hac_lag=HAC_LAG)
                entry["dm_vs_har"][m] = {
                    "stat": r.stat,
                    "p_value": r.p_value,
                    "mean_loss_diff": r.mean_loss_diff,
                    "seed_sign_agreement": per_seed_sign_agreement(
                        df, target, h, m, "har", floor, regime
                    ),
                }
            block[regime] = entry
        out["blocks"][f"{target}_h{h}"] = block
    return out


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1_headline(figdir: Path):
    metrics = {k: json.loads((RUNS[k] / "metrics.json").read_text()) for k in ("aux", "oc")}
    blocks = ["TV  h=1", "TV  h=5", "OC  h=1", "OC  h=5"]
    q = {
        "har": [
            metrics["aux"]["models"]["har"]["rv_tv"]["1"]["qlike"],
            metrics["aux"]["models"]["har"]["rv_tv"]["5"]["qlike"],
            metrics["oc"]["models"]["har"]["rv_oc"]["1"]["qlike"],
            metrics["oc"]["models"]["har"]["rv_oc"]["5"]["qlike"],
        ],
        "lstm_rvonly": [
            metrics["aux"]["models"]["lstm_rvonly"]["rv_tv"]["1"]["qlike"],
            metrics["aux"]["models"]["lstm_rvonly"]["rv_tv"]["5"]["qlike"],
            None,
            None,
        ],
        "lstm_aux": [
            metrics["aux"]["models"]["lstm_aux"]["rv_tv"]["1"]["qlike"],
            metrics["aux"]["models"]["lstm_aux"]["rv_tv"]["5"]["qlike"],
            metrics["oc"]["models"]["lstm_aux"]["rv_oc"]["1"]["qlike"],
            metrics["oc"]["models"]["lstm_aux"]["rv_oc"]["5"]["qlike"],
        ],
    }
    colors = {"har": C_HAR, "lstm_rvonly": C_RVONLY, "lstm_aux": C_AUX}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    style_ax(ax)
    x = np.arange(len(blocks))
    w = 0.26
    for i, m in enumerate(["har", "lstm_rvonly", "lstm_aux"]):
        vals = q[m]
        for j, v in enumerate(vals):
            if v is None:
                continue
            b = ax.bar(x[j] + (i - 1) * (w + 0.02), v, width=w, color=colors[m],
                       label=LABELS[m] if j == 0 else None)
            ax.annotate(f"{v:.3f}", (b[0].get_x() + w / 2, v), ha="center",
                        va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, color=INK2, fontsize=9)
    ax.set_ylabel("QLIKE (test span, lower is better)", color=INK2, fontsize=8)
    ax.set_title("Headline out-of-sample QLIKE — 10-seed ensemble mean, frozen HPs",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_headline_qlike.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig2_decay(figdir: Path):
    m = json.loads((RUNS["decay"] / "metrics.json").read_text())
    wls = [504, 1260, 3167]
    har = [m["models"][f"har_wl{w}"]["rv_tv"]["1"]["qlike"] for w in wls]
    net = [m["models"][f"lstm_rvonly_wl{w}"]["rv_tv"]["1"]["qlike"] for w in wls]
    pvals = [m["dm"][f"lstm_rvonly_wl{w}_vs_har_wl{w}_rv_tv_h1"]["p_value"] for w in wls]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    style_ax(ax)
    ax.plot(wls, har, color=C_HAR, linewidth=2, marker="o", markersize=6)
    ax.plot(wls, net, color=C_RVONLY, linewidth=2, marker="o", markersize=6)
    ax.annotate("HAR", (wls[-1], har[-1]), xytext=(10, 8), textcoords="offset points",
                color=C_HAR, fontsize=9, fontweight="bold")
    ax.annotate("LSTM (RV-only)", (wls[-1], net[-1]), xytext=(10, -12),
                textcoords="offset points", color="#0d7a52", fontsize=9, fontweight="bold")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.012, ymax + 0.018)
    for wl, p in zip(wls, pvals):
        lab = f"p={p:.1e}" if p < 0.05 else f"p={p:.2f} (n.s.)"
        ax.annotate(lab, (wl, max(har[wls.index(wl)], net[wls.index(wl)])),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=7.5, color=MUTED)
    ax.set_xticks(wls)
    ax.set_xticklabels(["504\n(~2y)", "1260\n(~5y)", "3167\n(~12.6y, frozen default)"],
                       color=INK2, fontsize=8)
    ax.set_xlabel("rolling training-window length (trading days)", color=INK2, fontsize=8)
    ax.set_ylabel("QLIKE (TV h=1, dev seeds)", color=INK2, fontsize=8)
    ax.set_title("Training-window decay curve — the net needs nearly all available history",
                 color=INK, fontsize=10, loc="left")
    ax.set_xlim(300, 4400)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_decay_curve.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def shade_regimes(ax, label_y=None):
    spans = {"covid_2020": "COVID", "bear_2022": "2022 bear"}
    for key, name in spans.items():
        lo, hi = REGIMES[key]
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=GRID, alpha=0.45, zorder=0)
        if label_y is not None:
            mid = pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2
            ax.annotate(name, (mid, label_y), ha="center", fontsize=7.5, color=MUTED)


def fig3_test_span(figdir: Path, floors: dict):
    ens = ensemble_mean(load_preds(RUNS["aux"]), "rv_tv", 1)
    wide = ens.pivot(index="origin_date", columns="model", values="y_pred").sort_index()
    y_true = ens.pivot(index="origin_date", columns="model", values="y_true").iloc[:, 0].sort_index()
    ann = lambda v: np.sqrt(252.0 * np.maximum(v, 0)) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    style_ax(ax)
    ax.plot(y_true.index, ann(y_true.to_numpy()), color=BASE, linewidth=0.8,
            label="realized (TV proxy)")
    ax.plot(wide.index, ann(wide["har"].to_numpy()), color=C_HAR, linewidth=1.1,
            label="HAR forecast")
    ax.plot(wide.index, ann(wide["lstm_aux"].to_numpy()), color=C_AUX, linewidth=1.1,
            label="LSTM (RV+aux) forecast")
    shade_regimes(ax, label_y=ax.get_ylim()[1] * 0.94)
    ax.set_ylabel("annualized volatility (%)", color=INK2, fontsize=8)
    ax.set_title("Test span (2020 → 2026-06): forecasts vs realized, TV h=1",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_test_span_vol.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig4_cum_diff(figdir: Path, floors: dict):
    ens = ensemble_mean(load_preds(RUNS["aux"]), "rv_tv", 1)
    wide = ens.pivot(index="origin_date", columns="model", values="y_pred").sort_index()
    y_true = ens.pivot(index="origin_date", columns="model", values="y_true").iloc[:, 0].sort_index()
    floor = floors["rv_tv"]
    l_har = qlike_series(wide["har"].to_numpy(), y_true.to_numpy(), floor)
    l_aux = qlike_series(wide["lstm_aux"].to_numpy(), y_true.to_numpy(), floor)
    cum = np.cumsum(l_har - l_aux)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    style_ax(ax)
    ax.plot(wide.index, cum, color=C_HAR, linewidth=1.6)
    ax.axhline(0, color=BASE, linewidth=0.8)
    shade_regimes(ax, label_y=cum.max() * 0.96)
    ax.set_ylabel("cumulative QLIKE saved vs HAR", color=INK2, fontsize=8)
    ax.set_title("Where the edge is earned — cumulative loss differential, LSTM (RV+aux) vs HAR (TV h=1)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_cum_qlike_diff.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    cfg = yaml.safe_load((REPO / "configs/default.yaml").read_text())
    floors = {k: float(v) for k, v in cfg["floors"].items()}
    outdir = REPO / "results/report"
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

    # consistency check: replayed full-span QLIKE must match the committed metrics.json
    committed = json.loads((RUNS["aux"] / "metrics.json").read_text())
    ens = ensemble_mean(load_preds(RUNS["aux"]), "rv_tv", 1)
    wide = ens.pivot(index="origin_date", columns="model", values="y_pred").sort_index()
    y_true = ens.pivot(index="origin_date", columns="model", values="y_true").iloc[:, 0].sort_index()
    replayed = float(np.mean(qlike_series(wide["lstm_aux"].to_numpy(), y_true.to_numpy(), floors["rv_tv"])))
    expected = committed["models"]["lstm_aux"]["rv_tv"]["1"]["qlike"]
    assert abs(replayed - expected) < 1e-10, f"replay mismatch: {replayed} vs {expected}"
    print(f"replay check OK: lstm_aux TV h=1 QLIKE {replayed:.10f} == committed {expected:.10f}")

    cut = regime_cut(floors)
    (outdir / "regime_cut.json").write_text(json.dumps(cut, indent=2, sort_keys=True))
    print("wrote", outdir / "regime_cut.json")

    fig1_headline(figdir)
    fig2_decay(figdir)
    fig3_test_span(figdir, floors)
    fig4_cum_diff(figdir, floors)
    print("wrote 4 figures to", figdir)


if __name__ == "__main__":
    main()
