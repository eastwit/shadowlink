"""Wall-clock to fixed-timestep conversion for MuJoCo."""

from __future__ import annotations


class FixedStepClock:
    """Accumulate real time and emit a bounded number of physics steps.

    A long stall (for example, a debugger breakpoint) does not cause the
    simulator to chase hundreds of missed steps; the accumulator is reset and
    simulation time is allowed to lag wall-clock time.
    """

    def __init__(
        self,
        timestep: float,
        max_catchup_s: float = 0.05,
        reset_stall_s: float = 0.5,
    ) -> None:
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        if max_catchup_s <= 0:
            raise ValueError("max_catchup_s must be positive")
        self.timestep = float(timestep)
        self.max_steps = max(1, int(max_catchup_s / self.timestep))
        self.reset_stall_s = float(reset_stall_s)
        self._last: float | None = None
        self._accumulator = 0.0

    @property
    def pending_s(self) -> float:
        return self._accumulator

    def reset(self, now: float | None = None) -> None:
        self._last = now
        self._accumulator = 0.0

    def advance(self, now: float) -> int:
        if self._last is None:
            self._last = now
            return 0
        delta = now - self._last
        self._last = now
        if delta < 0:
            self._accumulator = 0.0
            return 0
        if self.reset_stall_s > 0 and delta > self.reset_stall_s:
            self._accumulator = 0.0
            return 0
        self._accumulator += delta
        max_sim_s = self.max_steps * self.timestep
        if self._accumulator > max_sim_s:
            self._accumulator = max_sim_s
        steps = int(self._accumulator / self.timestep)
        self._accumulator -= steps * self.timestep
        return steps
