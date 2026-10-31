"""Radio channel: path loss, shadowing, RSRP and SINR.

Path loss follows the 3GPP TR 38.901 UMa LOS formula (PL1):
    PL = 28.0 + 22 log10(d3D) + 20 log10(fc[GHz])
Shadowing is log-normal and spatially correlated along the UE path using the
Gudmundson model, with an independent process per cell.
"""
import numpy as np

from .config import ScenarioConfig
from .mobility import gnb_positions

BOLTZMANN_DBM_HZ = -174.0


def path_loss_db(d3d_m: np.ndarray, fc_ghz: float) -> np.ndarray:
    d = np.maximum(d3d_m, 10.0)               # model is valid from 10 m
    return 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(fc_ghz)


def n_subcarriers(cfg: ScenarioConfig) -> int:
    # number of PRBs for the bandwidth (approx. TS 38.101 table), 12 SC per PRB
    n_prb = int(cfg.bandwidth_mhz * 1000 * 0.98 / (12 * cfg.scs_khz))
    return n_prb * 12


def tx_power_per_re_dbm(cfg: ScenarioConfig) -> float:
    """RSRP is defined per resource element, so spread the total power."""
    return cfg.tx_power_dbm - 10 * np.log10(n_subcarriers(cfg))


def noise_per_re_dbm(cfg: ScenarioConfig) -> float:
    return BOLTZMANN_DBM_HZ + 10 * np.log10(cfg.scs_khz * 1e3) + cfg.noise_figure_db


def correlated_shadowing(cfg: ScenarioConfig, x_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) shadowing per cell, correlated over travelled distance. Shape [steps, cells]."""
    n_steps, n_cells = len(x_m), cfg.n_cells
    dx = np.diff(x_m, prepend=x_m[0])
    rho = np.exp(-np.abs(dx) / cfg.shadow_decorr_m)          # [steps]
    # part of the shadowing is common to all sites (same local environment of the UE)
    rc = cfg.site_shadow_corr
    w = np.sqrt(rc) * rng.standard_normal((n_steps, 1)) + np.sqrt(1 - rc) * rng.standard_normal((n_steps, n_cells))
    s = np.empty((n_steps, n_cells))
    s[0] = w[0] * cfg.shadow_std_db
    innov = np.sqrt(1.0 - rho ** 2)[:, None] * cfg.shadow_std_db
    for k in range(1, n_steps):
        s[k] = rho[k] * s[k - 1] + innov[k] * w[k]
    return s


def rsrp_dbm(cfg: ScenarioConfig, x_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """True (noise-free) RSRP of every cell at every step. Shape [steps, cells]."""
    gx = gnb_positions(cfg)
    dx = x_m[:, None] - gx[None, :]
    d2d = np.sqrt(dx ** 2 + cfg.track_offset_m ** 2)
    d3d = np.sqrt(d2d ** 2 + (cfg.gnb_height_m - cfg.ue_height_m) ** 2)
    pl = path_loss_db(d3d, cfg.fc_ghz)
    shadow = correlated_shadowing(cfg, x_m, rng)
    return tx_power_per_re_dbm(cfg) - pl + shadow


def sinr_db_all(cfg: ScenarioConfig, rsrp: np.ndarray) -> np.ndarray:
    """SINR if each cell were the serving cell; other cells interfere at `interference_load`.

    Shape [steps, cells]. Column j is the SINR the UE would see when served by j.
    """
    p = 10 ** (rsrp / 10.0)
    noise = 10 ** (noise_per_re_dbm(cfg) / 10.0)
    total = p.sum(axis=1, keepdims=True)
    interference = cfg.interference_load * (total - p)
    return 10 * np.log10(p / (interference + noise))


def shannon_throughput_mbps(cfg: ScenarioConfig, sinr_db: np.ndarray) -> np.ndarray:
    """Truncated Shannon bound (TR 36.942 style): 0.75 efficiency, capped at 5.5 bit/s/Hz."""
    se = np.minimum(0.75 * np.log2(1 + 10 ** (sinr_db / 10.0)), 5.5)
    return se * cfg.bandwidth_mhz
