"""KPI collection and failure classification (3GPP TR 36.839 definitions).

Per run we record every successful HO and every failure (RLF or HOF), then
derive:
    ho_count, ping_pong_count, ping_pong_rate, unnecessary_ho
    rlf_count, hof_count, failures, too_late, too_early, wrong_cell
    interruption_ms (time with no data: HO / re-establishment / out-of-sync)
    mean_throughput_mbps
"""
from dataclasses import dataclass, field

from .config import ScenarioConfig

MOBILITY_TIMER_MS = 1000   # failure within 1 s after a HO counts as caused by that HO


@dataclass
class Event:
    t_ms: int
    x_m: float
    kind: str      # HO, RLF, HOF
    src: int
    dst: int
    label: str = ""


@dataclass
class KPICollector:
    policy: str
    speed_kmh: float
    seed: int
    cfg: ScenarioConfig
    events: list = field(default_factory=list)
    interruption_ms: int = 0
    thr_sum: float = 0.0
    steps: int = 0
    _last_ho: Event = None

    def handover(self, t, x, src, dst):
        ev = Event(t, float(x), "HO", src, dst)
        last = self._last_ho
        if (last is not None and last.src == dst and last.dst == src
                and t - last.t_ms < self.cfg.ping_pong_mts_ms):
            ev.label = "ping-pong"
        self.events.append(ev)
        self._last_ho = ev

    def failure(self, t, x, cause, src, reest_cell):
        last = self._last_ho
        recent = last is not None and t - last.t_ms < MOBILITY_TIMER_MS
        if recent and reest_cell == last.src:
            label = "too-early"
        elif recent and reest_cell not in (last.src, last.dst):
            label = "wrong-cell"
        elif reest_cell != src:
            label = "too-late"
        else:
            label = "coverage"
        self.events.append(Event(t, float(x), cause, src, reest_cell, label))
        self._last_ho = None

    def summary(self) -> dict:
        ho = [e for e in self.events if e.kind == "HO"]
        fails = [e for e in self.events if e.kind != "HO"]
        pp = sum(e.label == "ping-pong" for e in ho)
        n_ho = len(ho)
        return {
            "policy": self.policy,
            "speed_kmh": self.speed_kmh,
            "seed": self.seed,
            "ho_count": n_ho,
            "ping_pong_count": pp,
            "ping_pong_rate": pp / n_ho if n_ho else 0.0,
            "unnecessary_ho": max(0, n_ho - self.cfg.boundaries_crossed),
            "rlf_count": sum(e.kind == "RLF" for e in fails),
            "hof_count": sum(e.kind == "HOF" for e in fails),
            "failures": len(fails),
            "too_late": sum(e.label == "too-late" for e in fails),
            "too_early": sum(e.label == "too-early" for e in fails),
            "wrong_cell": sum(e.label == "wrong-cell" for e in fails),
            "interruption_ms": self.interruption_ms,
            "mean_throughput_mbps": self.thr_sum / self.steps if self.steps else 0.0,
        }
