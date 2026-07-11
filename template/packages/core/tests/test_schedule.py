"""Tests for eval schedules and their trackers."""

import itertools
import time
from typing import Any

import pytest

from core import EvalSchedule
from core import ScheduleTracker


class _Clock:
    """A hand-advanced monotonic clock; patch over `time.monotonic` in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _fired_steps(tracker: ScheduleTracker, total_steps: int) -> list[int]:
    """The steps at which the tracker fires over a run."""
    return [step for step in range(1, total_steps + 1) if tracker.due(step=step)]


def test_steps_schedule_fires_on_every_multiple() -> None:
    """A `steps: n` schedule fires exactly on the multiples of n."""
    tracker = ScheduleTracker(EvalSchedule(steps=5), total_steps=17)
    assert _fired_steps(tracker, 17) == [5, 10, 15]


@pytest.mark.parametrize(
    ("total", "total_steps"),
    [(4, 10), (1, 7), (10, 10), (3, 1000)],
    ids=["four-of-ten", "once", "every-step", "three-of-thousand"],
)
def test_total_schedule_fires_exactly_total_times(total: int, total_steps: int) -> None:
    """A `total: n` schedule fires n times, evenly, ending on the final step."""
    tracker = ScheduleTracker(EvalSchedule(total=total), total_steps=total_steps)
    fired = _fired_steps(tracker, total_steps)
    assert len(fired) == total
    assert fired[-1] == total_steps
    gaps = [after - before for before, after in itertools.pairwise(fired)]
    assert not gaps or max(gaps) - min(gaps) <= 1


def test_total_exceeding_the_run_length_raises() -> None:
    """A run too short to fire `total` times is rejected at tracker creation."""
    with pytest.raises(ValueError, match="exceeds"):
        ScheduleTracker(EvalSchedule(total=20), total_steps=10)


@pytest.mark.parametrize(
    ("spec", "seconds"),
    [("30s", 30.0), ("10min", 600.0), ("1.5h", 5400.0)],
    ids=["seconds", "minutes", "fractional-hours"],
)
def test_time_schedule_fires_when_the_interval_elapses(
    spec: str, seconds: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `time` schedule stays quiet below its interval and fires at it."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    tracker = ScheduleTracker(EvalSchedule(time=spec), total_steps=3)
    clock.now = seconds - 0.5
    assert not tracker.due(step=1)
    clock.now = seconds
    assert tracker.due(step=2)


def test_time_schedule_measures_from_the_last_firing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a firing, the next one needs a full interval from that firing."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    tracker = ScheduleTracker(EvalSchedule(time="30s"), total_steps=10)
    clock.now = 30.0
    assert tracker.due(step=1)
    clock.now = 45.0
    assert not tracker.due(step=2)
    clock.now = 60.0
    assert tracker.due(step=3)


def test_time_schedule_overshoot_fires_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One slow step that spans several intervals fires once, not repeatedly."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    tracker = ScheduleTracker(EvalSchedule(time="30s"), total_steps=10)
    clock.now = 95.0
    assert tracker.due(step=1)
    clock.now = 96.0
    assert not tracker.due(step=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 5, "total": 3},
        {"steps": 5, "time": "30s"},
        {"time": "30s", "total": 3},
    ],
    ids=["steps+total", "steps+time", "time+total"],
)
def test_schedule_rejects_combined_modalities(kwargs: dict[str, Any]) -> None:
    """Setting more than one of steps/time/total is a config bug."""
    with pytest.raises(ValueError, match="at most one"):
        EvalSchedule(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"steps": 0}, {"steps": -3}, {"total": 0}, {"time": "0s"}, {"time": "-5min"}],
    ids=["zero-steps", "negative-steps", "zero-total", "zero-time", "negative-time"],
)
def test_schedule_rejects_nonpositive_values(kwargs: dict[str, Any]) -> None:
    """A modality set to zero or a negative value raises."""
    with pytest.raises(ValueError, match="positive"):
        EvalSchedule(**kwargs)


@pytest.mark.parametrize(
    "spec",
    ["10", "5days", "s", "min"],
    ids=["bare", "unknown-unit", "no-value-s", "no-value-min"],
)
def test_malformed_durations_raise(spec: str) -> None:
    """A duration without a value or with an unknown unit raises."""
    with pytest.raises(ValueError, match="duration"):
        EvalSchedule(time=spec)


def test_empty_schedule_raises() -> None:
    """A schedule with no modality and no at_end would never fire."""
    with pytest.raises(ValueError, match="never fires"):
        EvalSchedule()


def test_at_end_only_schedule_never_fires_during_the_run() -> None:
    """An `at_end`-only schedule is silent in-loop and due once at the end."""
    tracker = ScheduleTracker(EvalSchedule(at_end=True), total_steps=20)
    assert _fired_steps(tracker, 20) == []
    assert tracker.due_at_end()


def test_due_at_end_mirrors_at_end_even_after_a_final_step_firing() -> None:
    """`at_end` holds regardless of what fired during the run."""
    tracker = ScheduleTracker(EvalSchedule(steps=5, at_end=True), total_steps=10)
    assert tracker.due(step=10)
    assert tracker.due_at_end()
    quiet = ScheduleTracker(EvalSchedule(steps=5), total_steps=10)
    assert not quiet.due_at_end()
