import threading
from unittest.mock import Mock

import pytest

from momentous import CheckResult, Covariance, MomentData, Status, Waveset, analyze
from momentous._engine import Engine


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("limits", [(1, None), (1, 2), (2, 2)])
def test_callback_reports_every_numerical_check_on_the_calling_thread(
    workers: int, limits: tuple[int, int | None]
) -> None:
    reports: list[tuple[Waveset, CheckResult, int]] = []
    data = MomentData({(0, 0): 1, (1, 0): 0.2}, covariance=Covariance.exact())
    pool = Waveset.from_max_l(1)
    caller = threading.get_ident()
    analysis = analyze(
        data,
        pool,
        on_check=lambda waves, result: reports.append(
            (waves, result, threading.get_ident())
        ),
    )
    assert len(reports) == 1 and reports[0][0] == pool
    search = analysis.search(min_size=limits[0], max_size=limits[1], workers=workers)
    search.minimal_wavesets()
    assert len(reports) == analysis.stats.checks
    assert all(thread == caller for _, _, thread in reports)
    assert len({waves for waves, _, _ in reports}) == len(reports)
    for waves, result, _ in reports:
        assert result.status is analyze(data, pool).check(waves).status
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
        data, pool, on_check=lambda waves, result: reports.append((waves, result))
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
def test_callback_exception_aborts_search_without_losing_the_completed_check(
    workers: int,
) -> None:
    pool = Waveset.from_max_l(1)
    cancelled: list[Waveset] = []

    def cancel_once(waves: Waveset, result: CheckResult) -> None:
        if waves != pool and not cancelled:
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
    assert analysis.search(workers=workers).complete


def test_unresolved_evaluations_are_delivered_to_the_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Engine, "evaluate", Mock(return_value=CheckResult(Status.UNRESOLVED))
    )
    reports: list[CheckResult] = []
    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 0)]),
        on_check=lambda waves, result: reports.append(result),
    )
    search = analysis.search(workers=2)
    assert not search.complete
    assert len(reports) == 3
    assert all(result.status is Status.UNRESOLVED for result in reports)


def test_non_callable_check_callback_is_rejected() -> None:
    with pytest.raises(TypeError, match="on_check"):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
            Waveset([(0, 0)]),
            on_check=1,  # ty: ignore[invalid-argument-type]
        )
