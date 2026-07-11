"""Per-split schedules saying when an eval dataset's metrics are computed.

``EvalSchedule`` is the config value: one modality (``steps``, ``time``, or
``total``) plus an ``at_end`` flag. ``ScheduleTracker`` is its mutable
counterpart, asked once per training step whether the split is due. A
``time`` tracker reads ``time.monotonic`` itself, anchoring the run start
when it is built.
"""

import dataclasses
import time

_UNIT_SECONDS = {"min": 60.0, "s": 1.0, "h": 3600.0}


def _parse_duration(spec: str) -> float:
    """Convert a duration like ``"30s"``, ``"10min"``, or ``"1.5h"`` to seconds.

    Args:
        spec: A positive number followed by one of the suffixes ``s``,
            ``min``, or ``h``.

    Returns:
        The duration in seconds.

    Raises:
        ValueError: If the suffix is missing or unknown, the number is
            malformed, or the duration is not positive.
    """
    for suffix, seconds in _UNIT_SECONDS.items():
        if not spec.endswith(suffix):
            continue
        try:
            value = float(spec.removesuffix(suffix))
        except ValueError:
            raise ValueError(f"malformed duration: {spec!r}") from None
        if value <= 0:
            raise ValueError(f"duration must be positive: {spec!r}")
        return value * seconds
    raise ValueError(f"duration needs an s/min/h suffix: {spec!r}")


@dataclasses.dataclass(frozen=True)
class EvalSchedule:
    """When one eval split's metrics are computed during a training run.

    At most one of ``steps``, ``time``, and ``total`` may be set; they are
    three ways to express the same in-training cadence. ``at_end`` combines
    with any of them, and a schedule with no modality must set ``at_end`` or
    it would never fire.

    Attributes:
        steps: Evaluate every this many global steps.
        time: Evaluate every wall-clock interval, e.g. ``"30s"``, ``"10min"``,
            ``"1.5h"``.
        total: Evaluate this many times, evenly spaced across the whole run.
        at_end: Evaluate once after the last training step.
    """

    steps: int | None = None
    time: str | None = None
    total: int | None = None
    at_end: bool = False

    def __post_init__(self) -> None:
        """Reject contradictory or never-firing schedules.

        Raises:
            ValueError: If more than one modality is set, a set modality is
                not positive, ``time`` doesn't parse, or nothing is set at
                all (no modality and ``at_end`` false).
        """
        modalities = {"steps": self.steps, "time": self.time, "total": self.total}
        set_names = [name for name, value in modalities.items() if value is not None]
        if len(set_names) > 1:
            raise ValueError(f"set at most one of steps/time/total, got {set_names}")
        if not set_names and not self.at_end:
            raise ValueError("schedule never fires: set steps, time, total, or at_end")
        if self.steps is not None and self.steps <= 0:
            raise ValueError(f"steps must be positive: {self.steps}")
        if self.total is not None and self.total <= 0:
            raise ValueError(f"total must be positive: {self.total}")
        if self.time is not None:
            _parse_duration(self.time)


class ScheduleTracker:
    """Stateful firing decisions for one ``EvalSchedule`` over one run.

    Bind a schedule to a run of ``total_steps``, then call ``due`` once per
    global step to learn when the split fires: every N steps, once per
    wall-clock interval, or a fixed number of times spread evenly over the
    run. The ``time`` modality reads ``time.monotonic``, anchoring the run
    start at construction and recording the last-fired timestamp between
    calls. Evaluation after the last step is reported by ``due_at_end``.
    """

    def __init__(self, schedule: EvalSchedule, *, total_steps: int) -> None:
        """Bind a schedule to a run of ``total_steps`` global steps.

        Args:
            schedule: The split's schedule.
            total_steps: Length of the run in global steps; a ``total``
                schedule spaces its firings over exactly this many steps.

        Raises:
            ValueError: If the schedule's ``total`` exceeds ``total_steps``,
                which cannot yield ``total`` distinct firings.
        """
        assert total_steps > 0, f"total_steps must be positive: {total_steps}"
        if schedule.total is not None and schedule.total > total_steps:
            raise ValueError(
                f"total={schedule.total} exceeds the run's {total_steps} steps"
            )
        self._schedule = schedule
        self._total_steps = total_steps
        self._interval_seconds = (
            _parse_duration(schedule.time) if schedule.time is not None else None
        )
        self._start_seconds = (
            time.monotonic() if self._interval_seconds is not None else None
        )
        self._last_fired_seconds = 0.0

    def due(self, *, step: int) -> bool:
        """Report whether the split should be evaluated at this step.

        A ``steps`` schedule fires on every multiple of its cadence. A
        ``time`` schedule reads the clock and fires when at least its
        interval has passed since the last firing (or since the run start,
        for the first one); the firing is recorded, so calling ``due`` more
        than once per step skews a ``time`` schedule. A ``total`` schedule
        fires ``total`` times, evenly spaced, the last on the final step. A
        schedule with no modality (``at_end`` only) never fires here.

        Args:
            step: The 1-based global step, between 1 and ``total_steps``.

        Returns:
            True if the split's metrics should be computed now.
        """
        assert 1 <= step <= self._total_steps, (
            f"step {step} outside the run of {self._total_steps}"
        )
        if self._schedule.steps is not None:
            return step % self._schedule.steps == 0
        if self._interval_seconds is not None:
            assert self._start_seconds is not None
            elapsed_seconds = time.monotonic() - self._start_seconds
            if elapsed_seconds - self._last_fired_seconds >= self._interval_seconds:
                self._last_fired_seconds = elapsed_seconds
                return True
            return False
        if self._schedule.total is not None:
            previous = (step - 1) * self._schedule.total // self._total_steps
            current = step * self._schedule.total // self._total_steps
            return current > previous
        return False

    def due_at_end(self) -> bool:
        """Report whether the split gets one more evaluation after training.

        Returns the schedule's ``at_end`` flag with no memory of past
        firings: a schedule that also fired on the final step is evaluated
        again.
        """
        return self._schedule.at_end
