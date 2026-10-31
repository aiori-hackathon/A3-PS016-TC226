#!/usr/bin/env python3
"""Plot one run of every policy: RSRP per cell, serving SINR and HO / failure events.

Example:
    python scripts/plot_timeline.py --speed 300 --seed 3
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoeval import build_channel, default_policies, load_config  # noqa: E402
from hoeval.analysis import plot_timeline  # noqa: E402
from hoeval.executor import HOExecutor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "hst.yaml"))
    ap.add_argument("--speed", type=float, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or ROOT / "results" / cfg.scenario)
    out.mkdir(parents=True, exist_ok=True)
    ch = build_channel(cfg, args.speed, args.seed)
    for p in default_policies(cfg):
        kpi = HOExecutor(cfg).run(p, ch)
        s = kpi.summary()
        slug = p.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        path = out / f"timeline_{int(args.speed)}kmh_{slug}.png"
        plot_timeline(kpi, ch, cfg, path)
        print(f"{p.name:<18} HO={s['ho_count']:2d}  ping-pong={s['ping_pong_count']}  "
              f"RLF={s['rlf_count']}  HOF={s['hof_count']}  -> {path.name}")


if __name__ == "__main__":
    main()
