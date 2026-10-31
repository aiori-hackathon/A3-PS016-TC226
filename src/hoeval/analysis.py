"""Statistics and plots for the experiment results."""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KPI_LABELS = {
    "failures": "HO failures + RLF per run",
    "rlf_count": "Radio link failures per run",
    "hof_count": "HO failures per run",
    "too_late": "Too-late HO per run",
    "ping_pong_rate": "Ping-pong rate",
    "ho_count": "Handovers per run",
    "unnecessary_ho": "Unnecessary HOs per run",
    "interruption_ms": "Interruption time per run (ms)",
    "mean_throughput_mbps": "Mean throughput (Mbit/s)",
}

POLICY_STYLE = {
    "A3 (fixed)": dict(color="#c0392b", marker="o"),
    "Speed-adaptive A3": dict(color="#1f5fbf", marker="s"),
    "CHO + adaptive": dict(color="#138808", marker="^"),
    "CHO": dict(color="#7a7a7a", marker="v"),
}

# two-sided 95 % Student-t critical values
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045, 39: 2.023, 49: 2.010}


def t95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if dof <= k:
            return _T95[k]
    return 1.96


def save_rows(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_rows(path) -> list[dict]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "policy":
                r[k] = float(v)
    return rows


def summarise(rows: list[dict], kpis=None) -> list[dict]:
    """Mean and 95 % confidence interval per (policy, speed, KPI)."""
    kpis = kpis or list(KPI_LABELS)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], float(r["speed_kmh"]))].append(r)
    out = []
    for (pol, v), rs in sorted(groups.items(), key=lambda g: (g[0][1], g[0][0])):
        n = len(rs)
        rec = {"policy": pol, "speed_kmh": v, "runs": n}
        for k in kpis:
            x = np.array([float(r[k]) for r in rs])
            m = x.mean()
            ci = t95(n - 1) * x.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
            rec[f"{k}_mean"] = round(m, 4)
            rec[f"{k}_ci95"] = round(ci, 4)
        out.append(rec)
    return out


def plot_kpis(summary: list[dict], out_dir, kpis=None, title_suffix="") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kpis = kpis or ["failures", "ping_pong_rate", "interruption_ms", "mean_throughput_mbps", "ho_count", "too_late"]
    policies = list(dict.fromkeys(r["policy"] for r in summary))
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, k in zip(axes.flat, kpis):
        for pol in policies:
            rs = sorted((r for r in summary if r["policy"] == pol), key=lambda r: r["speed_kmh"])
            v = [r["speed_kmh"] for r in rs]
            m = [r[f"{k}_mean"] for r in rs]
            ci = [r[f"{k}_ci95"] for r in rs]
            ax.errorbar(v, m, yerr=ci, label=pol, capsize=3, lw=1.8, ms=5, **POLICY_STYLE.get(pol, {}))
        ax.set_title(KPI_LABELS[k], fontsize=11)
        ax.set_xlabel("UE speed (km/h)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(f"Handover KPIs vs UE speed (mean ± 95% CI){title_suffix}", fontsize=13)
    fig.tight_layout()
    p = out_dir / "kpi_vs_speed.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_timeline(kpi, ch, cfg, path=None, max_cells=None):
    """RSRP of each cell along the track, serving-cell SINR, and HO / failure events."""
    x = ch.x_m / 1000.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    lo, hi = cfg.start_cell - 1, cfg.end_cell + 1
    for c in range(max(lo, 0), min(hi + 1, cfg.n_cells)):
        a1.plot(x, ch.rsrp[:, c], lw=0.8, alpha=0.8, label=str(c))
    a1.set_ylabel("RSRP (dBm)")
    a1.set_title(f"{kpi.policy} at {kpi.speed_kmh:.0f} km/h (seed {kpi.seed})")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0), title="cell", title_fontsize=7)

    # serving cell over time, reconstructed from the events
    serving = np.empty(len(ch.t_ms), dtype=int)
    cur = int(np.argmax(ch.meas[0]))
    evs = sorted(kpi.events, key=lambda e: e.t_ms)
    j = 0
    for k, t in enumerate(ch.t_ms):
        while j < len(evs) and evs[j].t_ms <= t:
            cur = evs[j].dst
            j += 1
        serving[k] = cur
    s_sinr = ch.sinr[np.arange(len(serving)), serving]
    a2.plot(x, s_sinr, lw=0.8, color="#333")
    a2.axhline(cfg.qout_db, color="#c0392b", ls="--", lw=1, label=f"Qout = {cfg.qout_db} dB")
    a2.set_ylabel("Serving SINR (dB)")
    a2.set_xlabel("Position along track (km)")
    a2.grid(alpha=0.3)

    colors = {"HO": "#138808", "RLF": "#c0392b", "HOF": "#e8871e"}
    seen = set()
    for e in evs:
        lab = e.kind if e.kind not in seen else None
        seen.add(e.kind)
        for ax in (a1, a2):
            ax.axvline(e.x_m / 1000.0, color=colors[e.kind], lw=1.1, ls="-" if e.kind == "HO" else ":",
                       label=lab if ax is a2 else None)
    a2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_sweep(grid: np.ndarray, hys_values, ttt_values, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(ttt_values)), [str(t) for t in ttt_values])
    ax.set_yticks(range(len(hys_values)), [f"{h:g}" for h in hys_values])
    ax.set_xlabel("TTT (ms)")
    ax.set_ylabel("Hysteresis (dB)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="failures + ping-pongs per run")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
