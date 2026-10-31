#!/usr/bin/env python3
"""Sweep Hys and TTT for the fixed A3 baseline at one speed.

Shows why one fixed setting cannot suit every speed: the best (Hys, TTT)
cell moves as speed changes.

Example:
    python scripts/param_sweep.py --speed 300 --runs 10
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoeval import A3Policy, load_config, run_experiment  # noqa: E402
from hoeval.analysis import plot_sweep, save_rows  # noqa: E402

def a3_only(cfg):
    return [A3Policy(cfg)]


HYS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
TTT = [0, 40, 80, 160, 320, 640]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "hst.yaml"))
    ap.add_argument("--speed", type=float, default=300)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    out = Path(args.out or ROOT / "results" / base.scenario)
    out.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((len(HYS), len(TTT)))
    rows = []
    for i, h in enumerate(HYS):
        for j, t in enumerate(TTT):
            cfg = base.with_overrides(hys_db=h, ttt_ms=t)
            res = run_experiment(cfg, [args.speed], args.runs, policy_factory=a3_only,
                                 workers=args.workers)
            score = np.mean([r["failures"] + r["ping_pong_count"] for r in res])
            grid[i, j] = score
            rows.append({"speed_kmh": args.speed, "hys_db": h, "ttt_ms": t,
                         "failures": np.mean([r["failures"] for r in res]),
                         "ping_pongs": np.mean([r["ping_pong_count"] for r in res]),
                         "score": score})
            print(f"Hys {h:3.1f} dB  TTT {t:4d} ms  ->  {score:.2f}")

    save_rows(rows, out / f"a3_sweep_{int(args.speed)}kmh.csv")
    plot_sweep(grid, HYS, TTT, f"Fixed A3 at {args.speed:.0f} km/h: failures + ping-pongs per run",
               out / f"a3_sweep_{int(args.speed)}kmh.png")
    best = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"Best setting at {args.speed:.0f} km/h: Hys {HYS[best[0]]} dB, TTT {TTT[best[1]]} ms")


if __name__ == "__main__":
    main()
