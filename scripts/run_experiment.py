#!/usr/bin/env python3
"""Run all handover policies over a speed sweep and save KPIs and plots.

Example:
    python scripts/run_experiment.py --config configs/hst.yaml --runs 20
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoeval import load_config, run_experiment  # noqa: E402
from hoeval.analysis import plot_kpis, save_rows, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "hst.yaml"))
    ap.add_argument("--speeds", default="30,60,120,200,300,400,500", help="comma-separated km/h")
    ap.add_argument("--runs", type=int, default=20, help="independent runs (seeds) per speed")
    ap.add_argument("--seed", type=int, default=1, help="first seed")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None, help="output folder (default results/<scenario>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    speeds = [float(s) for s in args.speeds.split(",")]
    out = Path(args.out or ROOT / "results" / cfg.scenario)

    t0 = time.time()
    print(f"Scenario '{cfg.scenario}': {len(speeds)} speeds x {args.runs} runs, "
          f"track {cfg.track_length_m / 1000:.1f} km, workers={args.workers}")
    rows = run_experiment(cfg, speeds, args.runs, base_seed=args.seed, workers=args.workers)
    summary = summarise(rows)
    save_rows(rows, out / "kpi_runs.csv")
    save_rows(summary, out / "kpi_summary.csv")
    plot_kpis(summary, out, title_suffix=f" - {cfg.scenario}, {args.runs} runs per point")
    print(f"Done in {time.time() - t0:.1f} s -> {out}")

    print(f"\n{'speed':>6}  {'policy':<18} {'fail/run':>9} {'ping-pong':>10} {'interr. ms':>11} {'thr Mbit/s':>11}")
    for r in summary:
        print(f"{r['speed_kmh']:>6.0f}  {r['policy']:<18} {r['failures_mean']:>9.2f} "
              f"{r['ping_pong_rate_mean']:>10.3f} {r['interruption_ms_mean']:>11.0f} {r['mean_throughput_mbps_mean']:>11.1f}")


if __name__ == "__main__":
    main()
