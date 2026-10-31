"""Basic checks of the simulator building blocks. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoeval import (A3Policy, ConditionalHO, ScenarioConfig, SpeedAdaptiveA3,  # noqa: E402
                    build_channel, load_config, run_experiment)
from hoeval.channel import path_loss_db  # noqa: E402
from hoeval.executor import HOExecutor  # noqa: E402
from hoeval.policies import snap_ttt, speed_scale  # noqa: E402

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hst.yaml"


def test_config_loads():
    cfg = load_config(CFG_PATH)
    assert cfg.scenario == "hst"
    assert cfg.track_length_m == 4500


def test_path_loss_increases_with_distance():
    pl = path_loss_db(np.array([10, 100, 1000]), 3.5)
    assert np.all(np.diff(pl) > 0)
    # 22 dB per decade
    assert abs((pl[2] - pl[1]) - 22.0) < 1e-9


def test_ttt_snaps_to_standard_values():
    assert snap_ttt(300) == 256
    assert snap_ttt(40) == 40
    assert snap_ttt(10) == 0


def test_speed_scaling():
    cfg = ScenarioConfig()
    assert speed_scale(cfg, 30) == 1.0
    assert speed_scale(cfg, 120) == 0.5
    assert speed_scale(cfg, 1000) == cfg.min_scale


def test_a3_triggers_only_after_ttt():
    cfg = ScenarioConfig(a3_offset_db=1, hys_db=2, ttt_ms=160)
    p = A3Policy(cfg)
    meas = np.array([-80.0, -76.0])          # neighbour 4 dB better > off + hys
    assert p.decide(0, meas, 0, 100) == []
    assert p.decide(120, meas, 0, 100) == []
    out = p.decide(160, meas, 0, 100)
    assert len(out) == 1 and out[0].target == 1


def test_a3_resets_when_condition_breaks():
    cfg = ScenarioConfig(a3_offset_db=1, hys_db=2, ttt_ms=160)
    p = A3Policy(cfg)
    p.decide(0, np.array([-80.0, -76.0]), 0, 100)
    p.decide(80, np.array([-80.0, -79.0]), 0, 100)       # condition fails
    assert p.decide(160, np.array([-80.0, -76.0]), 0, 100) == []


def test_common_random_numbers():
    cfg = load_config(CFG_PATH)
    a = build_channel(cfg, 200, 5)
    b = build_channel(cfg, 200, 5)
    assert np.array_equal(a.meas, b.meas)


def test_low_speed_baseline_has_no_failures():
    cfg = load_config(CFG_PATH)
    rows = run_experiment(cfg, [30], n_runs=3, policy_factory=lambda c: [A3Policy(c)])
    assert all(r["failures"] == 0 for r in rows)
    assert all(r["ho_count"] >= cfg.boundaries_crossed for r in rows)


def test_all_policies_run_at_high_speed():
    cfg = load_config(CFG_PATH)
    ch = build_channel(cfg, 500, 2)
    for p in (A3Policy(cfg), SpeedAdaptiveA3(cfg), ConditionalHO(cfg)):
        s = HOExecutor(cfg).run(p, ch).summary()
        assert s["ho_count"] + s["failures"] > 0
        assert 0 <= s["ping_pong_rate"] <= 1
