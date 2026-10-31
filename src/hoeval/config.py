"""Scenario configuration.

All parameters live in a YAML file under configs/. Loading it into a dataclass
keeps the rest of the code free of string keys and gives every value a type.
"""
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str = "hst"

    # deployment
    n_cells: int = 12
    isd_m: float = 500.0
    track_offset_m: float = 30.0
    gnb_height_m: float = 25.0
    ue_height_m: float = 3.0
    start_cell: int = 1
    end_cell: int = 10

    # radio
    fc_ghz: float = 3.5
    bandwidth_mhz: float = 100.0
    scs_khz: float = 30.0
    tx_power_dbm: float = 46.0
    noise_figure_db: float = 7.0
    shadow_std_db: float = 4.0
    shadow_decorr_m: float = 37.0
    site_shadow_corr: float = 0.5
    interference_load: float = 0.5
    meas_error_db: float = 1.0

    # measurement
    dt_ms: int = 10
    meas_period_ms: int = 40
    l3_filter_k: int = 4

    # A3 baseline
    a3_offset_db: float = 1.0
    hys_db: float = 2.0
    ttt_ms: int = 320

    # speed-adaptive A3
    speed_ref_kmh: float = 60.0
    min_scale: float = 0.25
    ttt_min_ms: int = 40
    hys_min_db: float = 0.5

    # conditional HO
    cho_prep_margin_db: float = -2.0

    # HO execution / RLF
    prep_delay_ms: int = 50
    ho_interruption_ms: int = 50
    qout_db: float = -8.0
    qin_db: float = -6.0
    t310_ms: int = 1000
    reestablish_ms: int = 500
    ping_pong_mts_ms: int = 1000

    @property
    def track_length_m(self) -> float:
        return (self.end_cell - self.start_cell) * self.isd_m

    @property
    def boundaries_crossed(self) -> int:
        return self.end_cell - self.start_cell

    def with_overrides(self, **kwargs) -> "ScenarioConfig":
        return replace(self, **kwargs)


def load_config(path) -> ScenarioConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(ScenarioConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return ScenarioConfig(**data)
