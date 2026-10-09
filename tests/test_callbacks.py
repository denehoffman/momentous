import threading
from unittest.mock import Mock

import pytest

from momentous import (
    CheckProgress,
    CheckResult,
    Covariance,
    MomentData,
    Status,
    Waveset,
    analyze,
)
from momentous._engine import Engine


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("limits", [(1, None), (1, 2), (2, 2)])
def test_callback_reports_every_numerical_check_on_the_calling_thread(
    workers: int, limits: tuple[int, int | None]
) -> None:
    reports: list[tuple[Waveset, CheckResult, CheckProgress, int]] = []
    data = MomentData({(0, 0): 1, (1, 0): 0.2}, covariance=Covariance.exact())
    pool = Waveset.from_max_l(1)
    caller = threading.get_ident()
    analysis = analyze(
        data,
        pool,
        on_check=lambda waves, result, progress: reports.append(
            (waves, result, progress, threading.get_ident())
        ),
    )
    assert len(reports) == 1 and reports[0][0] == pool
    search = analysis.search(min_size=limits[0], max_size=limits[1], workers=workers)
    search.minimal_wavesets()
    evaluated = [report for report in reports if report[2].evaluated]
    assert len(evaluated) == analysis.stats.checks
    assert all(thread == caller for _, _, _, thread in reports)
    assert len({waves for waves, _, _, _ in evaluated}) == len(evaluated)
    for waves, result, _, _ in reports:
        assert result.status is analyze(data, pool).check(waves).status
    snapshots = [
        progress for _, _, progress, _ in reports if progress.phase == "search"
    ]
    assert {progress.total for progress in snapshots} == {search.candidate_count}
    completed = [progress.completed for progress in snapshots]
    assert completed == sorted(completed)
    assert all(0 <= count <= search.candidate_count for count in completed)
    assert completed[-1] == search.candidate_count
    assert reports[0][2] == CheckProgress(1, 1)
    reference = analyze(data, pool).search(
        min_size=limits[0], max_size=limits[1], workers=1
    )
    assert list(search.wavesets()) == list(reference.wavesets())


def test_cached_and_inferred_checks_are_silent_but_new_diagnostics_are_reported() -> (
    None
):
    reports: list[tuple[Waveset, CheckResult]] = []
    data = MomentData({(0, 0): 1, (1, 0): 0.2}, covariance=Covariance.exact())
    pool = Waveset([(0, 0), (1, 0)])
    candidate = Waveset([(0, 0)])
    analysis = analyze(
        data,
        pool,
        on_check=lambda waves, result, progress: reports.append((waves, result)),
    )
    analysis.check(candidate)
    assert [waves for waves, _ in reports] == [pool, candidate]
    analysis.check(candidate)
    analysis.check(pool)
    analysis.check(Waveset([]))
    assert len(reports) == 2
    detailed = analysis.for_waves(candidate).check(diagnostics=True)
    assert reports[-1] == (candidate, detailed) and detailed.diagnostics
    analysis.check(candidate, diagnostics=True)
    assert len(reports) == 3


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("evaluated", [True, False], ids=["evaluation", "pruning"])
def test_callback_exception_aborts_search_without_losing_the_completed_check(
    workers: int,
    evaluated: bool,
) -> None:
    pool = Waveset.from_max_l(1)
    cancelled: list[Waveset] = []
    snapshots: list[CheckProgress] = []

    def cancel_once(
        waves: Waveset, result: CheckResult, progress: CheckProgress
    ) -> None:
        snapshots.append(progress)
        if waves != pool and not cancelled and progress.evaluated == evaluated:
            cancelled.append(waves)
            raise RuntimeError("cancel")

    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): 0.2}, covariance=Covariance.exact()),
        pool,
        on_check=cancel_once,
    )
    with pytest.raises(RuntimeError, match="cancel"):
        analysis.search(workers=workers)
    assert analysis.check(cancelled[0]).status is not Status.UNRESOLVED
    analysis.check(cancelled[0], diagnostics=True)
    assert snapshots[-1] == CheckProgress(1, 1)
    assert analysis.search(workers=workers).complete
    assert snapshots[-1].completed == snapshots[-1].total == 15


def test_unresolved_evaluations_are_delivered_to_the_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Engine, "evaluate", Mock(return_value=CheckResult(Status.UNRESOLVED))
    )
    reports: list[tuple[CheckResult, CheckProgress]] = []
    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 0)]),
        on_check=lambda waves, result, progress: reports.append((result, progress)),
    )
    search = analysis.search(workers=2)
    assert not search.complete
    assert sum(progress.evaluated for _, progress in reports) == 3
    assert all(result.status is Status.UNRESOLVED for result, _ in reports)
    assert reports[-1][1].completed == reports[-1][1].total == 3


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("value", [0.0, 2.0], ids=["passing", "excluded"])
def test_progress_counts_pruned_and_cached_domains_without_extra_evaluations(
    workers: int, value: float
) -> None:
    pool = Waveset.from_max_l(1)
    snapshots: list[CheckProgress] = []
    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): value}, covariance=Covariance.exact()),
        pool,
        on_check=lambda waves, result, progress: snapshots.append(progress),
    )
    found = analysis.search(min_size=2, max_size=3, workers=workers)
    assert snapshots[-1].completed == snapshots[-1].total == found.candidate_count == 10
    snapshots.clear()
    repeated = analysis.search(min_size=2, max_size=3, workers=workers)
    assert snapshots and all(not progress.evaluated for progress in snapshots)
    assert snapshots[-1].completed == snapshots[-1].total == 10
    assert list(found.wavesets()) == list(repeated.wavesets())


def test_non_callable_check_callback_is_rejected() -> None:
    with pytest.raises(TypeError, match="on_check"):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
            Waveset([(0, 0)]),
            on_check=1,  # ty: ignore[invalid-argument-type]
        )
