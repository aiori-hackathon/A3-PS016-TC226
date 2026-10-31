"""UE measurement model: L1 sampling with error, then L3 filtering (TS 38.331 5.5.3.2).

    F_n = (1 - a) * F_{n-1} + a * M_n,   a = 1 / 2^(k/4)

The filter runs on every cell regardless of which policy is used, so it is
computed once per run and shared by all policies (common random numbers).
"""
import numpy as np

from .config import ScenarioConfig


def l3_filtered_rsrp(cfg: ScenarioConfig, rsrp: np.ndarray, rng: np.random.Generator):
    """Return (meas_steps, filtered) where filtered has shape [n_meas, cells].

    meas_steps are the simulation step indices at which a measurement is available.
    """
    every = max(1, cfg.meas_period_ms // cfg.dt_ms)
    meas_steps = np.arange(0, rsrp.shape[0], every)
    m = rsrp[meas_steps] + rng.normal(0.0, cfg.meas_error_db, (len(meas_steps), rsrp.shape[1]))
    a = 1.0 / 2 ** (cfg.l3_filter_k / 4.0)
    f = np.empty_like(m)
    f[0] = m[0]
    for i in range(1, len(m)):
        f[i] = (1 - a) * f[i - 1] + a * m[i]
    return meas_steps, f
