"""Mobility model: a UE moving at constant speed along a straight track.

gNBs sit on a line at x = i * ISD. The track runs parallel to that line at a
fixed perpendicular offset, which is the usual layout for railway and highway
coverage (3GPP TR 38.913 high-speed-train scenario).
"""
import numpy as np

from .config import ScenarioConfig


def gnb_positions(cfg: ScenarioConfig) -> np.ndarray:
    """x-coordinates (m) of all gNBs."""
    return np.arange(cfg.n_cells) * cfg.isd_m


def ue_trajectory(cfg: ScenarioConfig, speed_kmh: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ms, x_m) for every simulation step.

    The UE starts at the position of `start_cell` and stops at `end_cell`, so
    every run covers the same distance whatever the speed.
    """
    v = speed_kmh / 3.6                        # m/s
    x0 = cfg.start_cell * cfg.isd_m
    x1 = cfg.end_cell * cfg.isd_m
    duration_ms = (x1 - x0) / v * 1000.0
    n_steps = int(np.ceil(duration_ms / cfg.dt_ms)) + 1
    t_ms = np.arange(n_steps) * cfg.dt_ms
    x = np.minimum(x0 + v * t_ms / 1000.0, x1)
    return t_ms, x
