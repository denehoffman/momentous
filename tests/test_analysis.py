from unittest.mock import Mock

import numpy as np
import pytest

from momentous import (
    CheckResult,
    Covariance,
    MomentData,
    ProjectionFailure,
    Status,
    Waveset,
    analyze,
)
from momentous._engine import Engine


def test_individual_components_pass_but_the_complex_pair_excludes() -> None:
    pool = Waveset([(0, 0), (1, 1)])
    moments = {(0, 0): 1, (1, 1): (0.25 + 0.25j)}
    assert (
        analyze(
            MomentData(moments, covariance=Covariance.exact()),
            pool,
            max_combination_size=1,
        )
        .check()
        .valid
    )
    result = analyze(MomentData(moments, covariance=Covariance.exact()), pool)
    assert not result.check().valid
    failure = result.check(diagnostics=True).diagnostics[0]
    assert isinstance(failure, ProjectionFailure)
    assert failure.components == ((1, 1, "real"), (1, 1, "imag"))
    assert failure.value > failure.bounds.upper


def test_real_imaginary_correlations_change_the_pair_verdict() -> None:
    pool = Waveset([(0, 0), (1, 1)])
    moments = {(0, 0): 1, (1, 1): (0.25 + 0.25j)}
    positive = [[0.01, 0.009], [0.009, 0.01]]
    negative = [[0.01, -0.009], [-0.009, 0.01]]
    labels = [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")]
    cov_positive = Covariance(np.pad(np.array(positive), (2, 0)), labels)
    cov_negative = Covariance(np.pad(np.array(negative), (2, 0)), labels)
    assert (
        analyze(MomentData(moments, covariance=cov_positive), pool, n_sigma=1)
        .check()
        .valid
    )
    result = analyze(MomentData(moments, covariance=cov_negative), pool, n_sigma=1)
    assert not result.check().valid
    failure = result.check(diagnostics=True).diagnostics[0]
    assert isinstance(failure, ProjectionFailure)
    weights = np.asarray(failure.coefficients)
    assert failure.uncertainty == pytest.approx(
        np.sqrt(weights[2:] @ negative @ weights[2:])
    )


def test_polarized_cross_variant_checks_use_singular_joint_covariance() -> None:
    pool = Waveset([(0, 0, "+"), (1, 0, "+")])
    moments = {(0, 0, 0): 1, (1, 0, 0): 0.4, (1, 0, 1): -0.4}
    labels = [(0, 0, 0), (1, 0, 0), (1, 0, 1)]
    positive = Covariance(np.pad([[0.25, 0.25], [0.25, 0.25]], (1, 0)), labels)
    negative = Covariance(np.pad([[0.25, -0.25], [-0.25, 0.25]], (1, 0)), labels)
    assert (
        analyze(
            MomentData(moments, covariance=positive),
            pool,
            max_combination_size=1,
            n_sigma=1,
        )
        .check()
        .valid
    )
    assert (
        not analyze(MomentData(moments, covariance=positive), pool, n_sigma=1)
        .check()
        .valid
    )
    assert (
        analyze(MomentData(moments, covariance=negative), pool, n_sigma=1).check().valid
    )


def test_duplicate_observables_still_apply_the_joint_euclidean_tolerance() -> None:
    pool = Waveset([(0, 0, "+"), (1, 0, "+")])
    tolerance = 1e-7
    value = 1 / np.sqrt(3) + 0.9 * tolerance
    data = MomentData(
        {(0, 0, 0): 1, (1, 0, 0): value, (1, 0, 1): value},
        covariance=Covariance.exact(),
    )
    assert (
        analyze(data, pool, max_combination_size=1, tolerance=tolerance).check().valid
    )
    assert not analyze(data, pool, tolerance=tolerance).check().valid


def test_full_diagnostics_collect_pairs_after_individual_failures() -> None:
    result = analyze(
        MomentData(
            {(0, 0): 1, (2, 0): 0.2, (4, 0): 0.9},
            covariance=Covariance.exact(),
        ),
        Waveset([(0, 0)]),
    )
    failures = result.check(diagnostics=True).diagnostics
    assert sum(len(item.components) == 1 for item in failures) == 2
    assert any(len(item.components) == 2 for item in failures)


def test_diagnostics_recompute_bounds_for_an_inferred_candidate() -> None:
    pool = Waveset([(0, 0), (1, 0)])
    result = analyze(
        MomentData({(0, 0): 1, (2, 0): 1}, covariance=Covariance.exact()), pool
    ).search()
    candidate = Waveset([(0, 0)])
    assert result.status(candidate) is Status.INCOMPATIBLE
    failure = result.analysis.check(candidate, diagnostics=True).diagnostics[0]
    assert isinstance(failure, ProjectionFailure)
    assert failure.bounds.upper == 0
    larger = result.analysis.check(pool, diagnostics=True).diagnostics[0]
    assert isinstance(larger, ProjectionFailure)
    assert larger.bounds != failure.bounds


@pytest.mark.parametrize("workers", [1, 3])
def test_search_matches_fresh_candidate_checks_and_certifies_minimal_sets(
    workers: int,
) -> None:
    pool = Waveset([(0, 0), (1, 0), (1, 1), (2, 0)])
    moments = {(0, 0): 1, (1, 0): 1 / np.sqrt(3), (2, 0): 0.2}
    expected = {
        candidate
        for candidate in pool
        if analyze(
            MomentData(moments, covariance=Covariance.exact()), candidate, n_sigma=0
        )
        .check()
        .valid
    }
    result = analyze(
        MomentData(moments, covariance=Covariance.exact()), pool, n_sigma=0
    ).search(workers=workers)
    passing = list(result.wavesets())
    assert result.complete and set(passing) == expected
    assert passing == [candidate for candidate in pool if candidate in expected]
    assert len(passing) == len(set(passing)) == result.count
    expected_minimal = {
        candidate
        for candidate in expected
        if not any(set(other.waves) < set(candidate.waves) for other in expected)
    }
    assert set(result.minimal_wavesets()) == expected_minimal
    for candidate in pool:
        assert (result.status(candidate) is Status.COMPATIBLE) == (
            candidate in expected
        )
    assert set(result.wavesets("incompatible")) == set(pool) - expected


def test_minimality_is_relative_to_the_search_size_domain() -> None:
    pool = Waveset.from_max_l(1)
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()), pool
    ).search(min_size=2, max_size=2)
    assert result.count == 6
    assert result.size_limits == (2, 2) and result.candidate_count == 6
    assert set(result.minimal_wavesets()) == set(pool.powerset(2, 2))
    with pytest.raises(ValueError, match="size domain"):
        result.status(Waveset([(0, 0)]))
    assert result.analysis.check(Waveset([(0, 0)])).valid


@pytest.mark.parametrize("workers", [1, 3])
def test_unresolved_decisions_do_not_prune_search(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    monkeypatch.setattr(
        Engine, "evaluate", Mock(return_value=CheckResult(Status.UNRESOLVED))
    )
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 0), (2, 0)]),
    ).search(workers=workers)
    assert not result.complete
    assert len(result.unresolved) == 7 and result.count == 0
    assert result.minimal_wavesets() == ()


@pytest.mark.parametrize("workers", [1, 3])
def test_unresolved_smaller_candidate_prevents_minimal_certification(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    def verdict(self: Engine, mask: int, *, diagnostics: bool = False) -> CheckResult:
        return CheckResult(
            {
                0: Status.INCOMPATIBLE,
                1: Status.UNRESOLVED,
                2: Status.INCOMPATIBLE,
                3: Status.COMPATIBLE,
            }[mask]
        )

    monkeypatch.setattr(Engine, "evaluate", verdict)
    pool = Waveset([(0, 0), (1, 0)])
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()), pool
    ).search(workers=workers)
    assert result.status(pool) is Status.COMPATIBLE
    assert not result.complete and result.minimal_wavesets() == ()


@pytest.mark.parametrize("workers", [1, 3])
def test_later_pass_certificate_resolves_an_earlier_unknown_superset(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    def verdict(self: Engine, mask: int, *, diagnostics: bool = False) -> CheckResult:
        return CheckResult(Status.COMPATIBLE if mask == 2 else Status.UNRESOLVED)

    monkeypatch.setattr(Engine, "evaluate", verdict)
    pool = Waveset([(0, 0), (1, 0)])
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()), pool
    ).search(workers=workers)
    assert result.status(pool) is Status.COMPATIBLE
    assert result.count == 2
    assert result.unresolved == (Waveset([(0, 0)]),)


def test_one_prepared_analysis_can_search_multiple_size_domains() -> None:
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 0)]),
    )
    assert result.check().valid
    first = result.search()
    second = result.search(min_size=1, max_size=1)
    assert first.analysis is result and second.analysis is result
    assert first.size_limits == (1, 2) and second.size_limits == (1, 1)
    assert first.count == 3 and second.count == 2
    with pytest.raises(ValueError, match="size domain"):
        second.status(result.pool)
    assert result.status(result.pool) is Status.COMPATIBLE


def test_parallel_solver_checks_preserve_correlated_pair_verdicts() -> None:
    pool = Waveset.from_max_l(1)
    labels = [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")]
    covariance = Covariance(np.pad([[0.01, -0.009], [-0.009, 0.01]], (2, 0)), labels)
    data = MomentData({(0, 0): 1, (1, 1): 0.25 + 0.25j}, covariance=covariance)
    serial = analyze(data, pool, n_sigma=1).search(workers=1)
    analysis = analyze(data, pool, n_sigma=1)
    parallel = analysis.search(workers=3)
    assert serial.complete == parallel.complete
    assert [serial.status(candidate) for candidate in pool] == [
        parallel.status(candidate) for candidate in pool
    ]
    assert set(serial.minimal_wavesets()) == set(parallel.minimal_wavesets())
    assert analysis.search(workers=3).count == parallel.count


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_search_workers_require_a_positive_integer(workers: int) -> None:
    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0)]),
    )
    with pytest.raises((ValueError, TypeError), match="workers"):
        analysis.search(workers=workers)


def test_candidate_metric_is_restricted_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "momentous.data.normalization_metric",
        Mock(return_value=np.array([[2, 0.5], [0.5, 1]], dtype=complex)),
    )
    monkeypatch.setattr(
        "momentous.data.moment_operators",
        Mock(
            side_effect=[
                np.array([[[2, 0.5], [0.5, 1]]], complex),
                np.array([[[0.2, 0.1], [0.1, 0.7]]], complex),
            ]
        ),
    )
    pool = Waveset([(0, 0, "+"), (1, 0, "+")])
    result = analyze(
        MomentData({(0, 0, 0): 1, (1, 0, 0): 0.1}, covariance=Covariance.exact()), pool
    )
    candidate = Waveset([(0, 0, "+")])
    assert result.check(candidate).valid
    assert result.operators(candidate)[1, 0, 0].real == pytest.approx(0.1)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_combination_sizes_are_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="max_combination_size"):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
            Waveset([(0, 0)]),
            max_combination_size=size,
        )


def test_future_combination_sizes_are_not_silently_downgraded() -> None:
    with pytest.raises(NotImplementedError):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
            Waveset([(0, 0)]),
            max_combination_size=3,
        )


@pytest.mark.parametrize("status", list(Status))
def test_status_requires_explicit_comparison(status: Status) -> None:
    with pytest.raises(TypeError, match="Compare Status explicitly"):
        bool(status)


@pytest.mark.parametrize("minimum,maximum", [(0, 2), (3, 2), (1, 5), (True, 2)])
def test_search_requires_a_nonempty_admissible_size_domain(
    minimum: int, maximum: int
) -> None:
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset.from_max_l(1),
    )
    with pytest.raises(ValueError):
        result.search(min_size=minimum, max_size=maximum)
