"""HO execution and radio link monitoring.

Runs one policy over one pre-computed channel realisation, step by step, and
records every event in a KPICollector.

Failure model (simplified from TS 38.331 / TR 36.839):
  * Legacy HO: the measurement report and the HO command both travel over the
    serving link. If serving SINR < Qout when the command is due, the command is
    lost -> HO failure (HOF).
  * Execution (legacy and CHO): if the target SINR < Qout, RACH fails -> HOF.
  * RLF: serving SINR below Qout continuously for T310 -> radio link failure.
  * After HOF or RLF the UE re-establishes on the strongest cell, which costs
    `reestablish_ms` of interruption.
"""
from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig
from .kpi import KPICollector
from .policies import HandoverPolicy


@dataclass
class ChannelRun:
    """Everything that does not depend on the policy, computed once per (speed, seed)."""
    speed_kmh: float
    seed: int
    t_ms: np.ndarray        # [steps]
    x_m: np.ndarray         # [steps]
    rsrp: np.ndarray        # [steps, cells] true RSRP
    sinr: np.ndarray        # [steps, cells] SINR if served by each cell
    thr: np.ndarray         # [steps, cells] throughput if served by each cell
    meas_steps: np.ndarray  # [n_meas] step index of each measurement
    meas: np.ndarray        # [n_meas, cells] L3-filtered RSRP


class HOExecutor:
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg

    def run(self, policy: HandoverPolicy, ch: ChannelRun) -> KPICollector:
        cfg = self.cfg
        dt = cfg.dt_ms
        kpi = KPICollector(policy.name, ch.speed_kmh, ch.seed, cfg)

        policy.reset()
        serving = int(np.argmax(ch.meas[0]))
        blocked_until = -1          # interruption (HO or re-establishment) ends at this time
        t310_start = None
        pending = None              # legacy HO: (target, command_time)
        cho_pending = []            # CHO preparations: [(target, ready_time)]
        meas_ptr = 0
        meas_steps = ch.meas_steps
        n_meas = len(meas_steps)

        def reestablish(k, t, cause):
            nonlocal serving, blocked_until, t310_start, pending, cho_pending
            new = int(np.argmax(ch.rsrp[k]))
            kpi.failure(t, ch.x_m[k], cause, serving, new)
            serving = new
            blocked_until = t + cfg.reestablish_ms
            t310_start, pending, cho_pending = None, None, []
            policy.reset()

        def execute(k, t, target):
            nonlocal serving, blocked_until, t310_start, pending, cho_pending
            if ch.sinr[k, target] < cfg.qout_db:
                reestablish(k, t, "HOF")
                return
            kpi.handover(t, ch.x_m[k], serving, target)
            serving = target
            blocked_until = t + cfg.ho_interruption_ms
            t310_start, pending, cho_pending = None, None, []
            policy.reset()

        for k in range(len(ch.t_ms)):
            t = int(ch.t_ms[k])
            interrupted = t < blocked_until

            # --- data delivered in this step ---
            s = ch.sinr[k, serving]
            if interrupted or s < cfg.qout_db:
                kpi.interruption_ms += dt
                kpi.thr_sum += 0.0
            else:
                kpi.thr_sum += ch.thr[k, serving]
            kpi.steps += 1

            if interrupted:
                while meas_ptr < n_meas and meas_steps[meas_ptr] <= k:
                    meas_ptr += 1
                continue

            # --- legacy HO command due ---
            if pending is not None and t >= pending[1]:
                target = pending[0]
                pending = None
                if s < cfg.qout_db:
                    reestablish(k, t, "HOF")
                    continue
                execute(k, t, target)
                continue

            # --- CHO preparation completes (command delivered over serving link) ---
            if cho_pending:
                still = []
                for target, ready in cho_pending:
                    if t >= ready:
                        if s >= cfg.qout_db:
                            policy.on_prepared(target)
                        else:
                            policy.on_prepare_failed(target)
                    else:
                        still.append((target, ready))
                cho_pending = still

            # --- radio link monitoring ---
            if s < cfg.qout_db:
                if t310_start is None:
                    t310_start = t
                elif t - t310_start >= cfg.t310_ms:
                    reestablish(k, t, "RLF")
                    continue
            elif s > cfg.qin_db:
                t310_start = None

            # --- measurement instant: ask the policy ---
            if meas_ptr < n_meas and meas_steps[meas_ptr] == k:
                meas = ch.meas[meas_ptr]
                meas_ptr += 1
                if pending is None:
                    for d in policy.decide(t, meas, serving, ch.speed_kmh):
                        if d.kind == "HO":
                            # report must reach the serving gNB
                            if s >= cfg.qout_db:
                                pending = (d.target, t + cfg.prep_delay_ms)
                        elif d.kind == "PREPARE":
                            if s >= cfg.qout_db:
                                cho_pending.append((d.target, t + cfg.prep_delay_ms))
                            else:
                                policy.on_prepare_failed(d.target)
                        elif d.kind == "EXECUTE":
                            execute(k, t, d.target)
                            break

        return kpi
