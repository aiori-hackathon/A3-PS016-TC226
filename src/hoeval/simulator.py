"""Simulation engine: builds a channel realisation and runs every policy on it.

Common random numbers: for each (speed, seed) the trajectory, shadowing and
measurement noise are generated once and shared by all policies, so any KPI
difference comes from the policy and not from a luckier channel.
"""
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

from .channel import rsrp_dbm, shannon_throughput_mbps, sinr_db_all
from .config import ScenarioConfig
from .executor import ChannelRun, HOExecutor
from .kpi import KPICollector
from .measurement import l3_filtered_rsrp
from .mobility import ue_trajectory
from .policies import HandoverPolicy, default_policies


def build_channel(cfg: ScenarioConfig, speed_kmh: float, seed: int) -> ChannelRun:
    rng = np.random.default_rng([seed, int(round(speed_kmh * 10))])
    t_ms, x = ue_trajectory(cfg, speed_kmh)
    rsrp = rsrp_dbm(cfg, x, rng)
    sinr = sinr_db_all(cfg, rsrp)
    thr = shannon_throughput_mbps(cfg, sinr)
    meas_steps, meas = l3_filtered_rsrp(cfg, rsrp, rng)
    return ChannelRun(speed_kmh, seed, t_ms, x, rsrp, sinr, thr, meas_steps, meas)


def run_single(cfg: ScenarioConfig, policy: HandoverPolicy, speed_kmh: float, seed: int = 0) -> tuple[KPICollector, ChannelRun]:
    """One policy, one speed, one seed. Returns the KPI collector (with events) and the channel."""
    ch = build_channel(cfg, speed_kmh, seed)
    return HOExecutor(cfg).run(policy, ch), ch


def _run_task(args):
    cfg, speed, seed, factory = args
    ch = build_channel(cfg, speed, seed)
    ex = HOExecutor(cfg)
    return [ex.run(p, ch).summary() for p in factory(cfg)]


def run_experiment(cfg: ScenarioConfig, speeds, n_runs: int, base_seed: int = 1,
                   policy_factory: Callable = default_policies, workers: int = 1) -> list[dict]:
    """Run every policy for every speed and `n_runs` independent seeds."""
    tasks = [(cfg, float(v), base_seed + r, policy_factory) for v in speeds for r in range(n_runs)]
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(_run_task, tasks, chunksize=1):
                rows.extend(out)
    else:
        for task in tasks:
            rows.extend(_run_task(task))
    return rows
