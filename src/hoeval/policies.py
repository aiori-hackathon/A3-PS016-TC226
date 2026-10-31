"""Handover decision policies.

Every policy implements the same HandoverPolicy interface, so the simulator can
run them on identical scenarios and compare them fairly.

    A3Policy           - standard A3 event with fixed Off / Hys / TTT (baseline)
    SpeedAdaptiveA3    - A3 with TTT and Hys scaled down as UE speed increases
    ConditionalHO      - Rel-16 CHO: prepare early, execute on the A3 condition

A3 entering condition (TS 38.331 5.5.4.4), with Ofn = Ocn = Ofp = Ocp = 0:
    Mn - Hys > Mp + Off
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig

# TimeToTrigger values allowed by TS 38.331 (ms)
TTT_VALUES_MS = (0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120)


@dataclass(frozen=True)
class Decision:
    kind: str      # "HO" (legacy), "PREPARE" or "EXECUTE" (CHO)
    target: int


def snap_ttt(ttt_ms: float) -> int:
    """Round down to the nearest TTT value allowed by the standard."""
    allowed = [v for v in TTT_VALUES_MS if v <= ttt_ms]
    return allowed[-1] if allowed else 0


def speed_scale(cfg: ScenarioConfig, speed_kmh: float) -> float:
    """1.0 up to the reference speed, then shrinks as v_ref / v, never below min_scale."""
    if speed_kmh <= cfg.speed_ref_kmh:
        return 1.0
    return max(cfg.min_scale, cfg.speed_ref_kmh / speed_kmh)


class HandoverPolicy(ABC):
    name = "policy"

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Clear internal state (called at start of run and after every HO / re-establishment)."""

    def on_prepared(self, cell: int) -> None:
        """CHO only: the HO command for `cell` has been received by the UE."""

    def on_prepare_failed(self, cell: int) -> None:
        """CHO only: preparation of `cell` failed, so it may be requested again."""

    @abstractmethod
    def decide(self, t_ms: int, meas: np.ndarray, serving: int, speed_kmh: float) -> list[Decision]:
        """Called at every measurement instant with L3-filtered RSRP of all cells."""


class _A3Tracker:
    """Keeps the TTT timer for the A3 entering condition."""

    def __init__(self):
        self.cand = None
        self.since = None

    def reset(self):
        self.cand, self.since = None, None

    def update(self, t_ms, meas, serving, off, hys, ttt, allowed=None):
        m = meas.copy()
        m[serving] = -np.inf
        if allowed is not None:
            mask = np.full(m.shape, True)
            mask[list(allowed)] = False
            m[mask] = -np.inf
        n = int(np.argmax(m))
        if not np.isfinite(m[n]) or not (m[n] - hys > meas[serving] + off):
            self.reset()
            return None
        if n != self.cand:
            self.cand, self.since = n, t_ms
        if t_ms - self.since >= ttt:
            return n
        return None


class A3Policy(HandoverPolicy):
    name = "A3 (fixed)"

    def reset(self):
        self.a3 = _A3Tracker()

    def params(self, speed_kmh):
        return self.cfg.a3_offset_db, self.cfg.hys_db, self.cfg.ttt_ms

    def decide(self, t_ms, meas, serving, speed_kmh):
        off, hys, ttt = self.params(speed_kmh)
        target = self.a3.update(t_ms, meas, serving, off, hys, ttt)
        return [Decision("HO", target)] if target is not None else []


class SpeedAdaptiveA3(A3Policy):
    name = "Speed-adaptive A3"

    def params(self, speed_kmh):
        c = self.cfg
        s = speed_scale(c, speed_kmh)
        ttt = snap_ttt(max(c.ttt_min_ms, c.ttt_ms * s))
        hys = max(c.hys_min_db, c.hys_db * s)
        return c.a3_offset_db, hys, ttt


class ConditionalHO(HandoverPolicy):
    """Conditional handover with speed-adaptive execution condition.

    Preparation: when a neighbour comes within `cho_prep_margin_db` of the serving
    cell, ask the network to prepare it. The UE receives the HO command early,
    while the serving link is still good.
    Execution: when the A3 condition holds for a prepared cell for TTT, the UE
    switches on its own, without waiting for the serving cell.
    """
    name = "CHO + adaptive"

    def __init__(self, cfg, adaptive: bool = True):
        self.adaptive = adaptive
        if not adaptive:
            self.name = "CHO"
        super().__init__(cfg)

    def reset(self):
        self.a3 = _A3Tracker()
        self.requested: set[int] = set()
        self.prepared: set[int] = set()

    def on_prepared(self, cell):
        self.prepared.add(cell)

    def on_prepare_failed(self, cell):
        self.requested.discard(cell)

    def decide(self, t_ms, meas, serving, speed_kmh):
        c = self.cfg
        out = []
        m = meas.copy()
        m[serving] = -np.inf
        n = int(np.argmax(m))
        if meas[n] > meas[serving] + c.cho_prep_margin_db and n not in self.requested:
            self.requested.add(n)
            out.append(Decision("PREPARE", n))

        if self.prepared:
            if self.adaptive:
                s = speed_scale(c, speed_kmh)
                ttt = snap_ttt(max(c.ttt_min_ms, c.ttt_ms * s))
                hys = max(c.hys_min_db, c.hys_db * s)
            else:
                ttt, hys = c.ttt_ms, c.hys_db
            target = self.a3.update(t_ms, meas, serving, c.a3_offset_db, hys, ttt, allowed=self.prepared)
            if target is not None:
                out.append(Decision("EXECUTE", target))
        return out


def default_policies(cfg: ScenarioConfig) -> list[HandoverPolicy]:
    return [A3Policy(cfg), SpeedAdaptiveA3(cfg), ConditionalHO(cfg, adaptive=True)]
