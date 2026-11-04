from unittest.mock import Mock

import numpy as np
import pytest

from momentous import (
    Covariance,
    MomentData,
    Status,
    UnresolvedCompatibilityError,
    Waveset,
    analyze,
)
from momentous.geometry import _ConicSolver


def disk_data(covariance: np.ndarray, scale: float = 1) -> MomentData:
    labels = [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")]
    return MomentData(
        {(0, 0): scale, (1, 1): scale * (0.25 + 0.25j)},
        covariance=Covariance(np.pad(scale**2 * covariance, (2, 0)), labels),
    )


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize("tolerance", [1e-8, 1e-3])
def test_direct_pair_radius_matches_disk_distance_and_passes(
    scale: float, tolerance: float
) -> None:
    waves = Waveset([(0, 0), (1, 1)])
    data = disk_data(np.eye(2) * 1e-4, scale)
    analysis = analyze(data, waves, n_sigma=1, tolerance=tolerance)
    accuracy = 1e-3
    required = analysis.required_n_sigma(accuracy=accuracy)
    expected = (np.sqrt(0.125) - 1 / (2 * np.sqrt(3)) - tolerance) / 0.01
    assert expected <= required <= expected + accuracy
    assert analyze(data, waves, n_sigma=required, tolerance=tolerance).check().valid
    assert (
        not analyze(data, waves, n_sigma=required - 2 * accuracy, tolerance=tolerance)
        .check()
        .valid
    )


@pytest.mark.parametrize("correlation", [-0.9, 0.9, 1.0])
def test_correlated_and_singular_pair_radii(correlation: float) -> None:
    waves = Waveset([(0, 0), (1, 1)])
    data = disk_data(np.array([[1, correlation], [correlation, 1]]) * 1e-4)
    analysis = analyze(data, waves)
    expected = (np.sqrt(0.125) - 1 / (2 * np.sqrt(3)) - 1e-8) / np.sqrt(
        1e-4 * (1 + correlation)
    )
    required = analysis.required_n_sigma()
    assert expected <= required <= expected + 1e-3
    assert analyze(data, waves, n_sigma=required).check().valid


def test_scalar_only_radius_and_candidate_membership() -> None:
    pool = Waveset([(0, 0), (1, 0)])
    candidate = Waveset([(0, 0)])
    data = MomentData(
        {(0, 0): 10, (1, 0): 2},
        covariance=Covariance.from_uncertainties({(0, 0): 0, (1, 0): 0.5}),
    )
    analysis = analyze(data, pool, max_combination_size=1)
    assert analysis.required_n_sigma() == 0
    required = analysis.for_waves(candidate).required_n_sigma()
    assert required == pytest.approx(4, abs=1e-6)
    assert (
        analyze(data, candidate, n_sigma=required, max_combination_size=1).check().valid
    )
    with pytest.raises(ValueError, match="outside the pool"):
        analysis.required_n_sigma(Waveset([(2, 0)]))


@pytest.mark.parametrize("imaginary", [0.1, 0.2, 1 / (2 * np.sqrt(3))])
def test_singular_pair_with_an_exact_tangent_direction(imaginary: float) -> None:
    waves = Waveset([(0, 0), (1, 1)])
    data = MomentData(
        {(0, 0): 1, (1, 1): 0.3 + imaginary * 1j},
        covariance=Covariance(
            np.diag([0, 0, 0.01, 0]),
            [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")],
        ),
    )
    required = analyze(data, waves).required_n_sigma()
    expected = (0.3 - np.sqrt((1 / (2 * np.sqrt(3)) + 1e-8) ** 2 - imaginary**2)) / 0.1
    assert expected <= required <= expected + 1e-3
    assert analyze(data, waves, n_sigma=required).check().valid


def test_scalar_mode_does_not_require_the_pair_radius() -> None:
    waves = Waveset([(0, 0), (1, 1)])
    data = disk_data(np.eye(2) * 1e-4)
    assert analyze(data, waves, max_combination_size=1).required_n_sigma() == 0
    assert analyze(data, waves).required_n_sigma() > 6


def test_exact_centers_exclusions_and_empty_candidates() -> None:
    waves = Waveset([(0, 0), (1, 1)])
    center = MomentData({(0, 0): 1, (1, 1): 0j}, covariance=Covariance.exact())
    assert analyze(center, waves).required_n_sigma() == 0
    assert np.isinf(analyze(center, waves).required_n_sigma(Waveset([])))
    outside = MomentData(
        {(0, 0): 1, (1, 1): 0.25 + 0.25j}, covariance=Covariance.exact()
    )
    assert np.isinf(analyze(outside, waves).required_n_sigma())
    assert np.isinf(analyze(outside, Waveset([(0, 0)])).required_n_sigma())


def test_singular_exact_direction_can_exclude_every_radius() -> None:
    waves = Waveset([(0, 0, "+"), (1, 0, "+")])
    data = MomentData(
        {(0, 0, 0): 1, (1, 0, 0): 0.4, (1, 0, 1): -0.4},
        covariance=Covariance(
            np.pad([[0.25, 0.25], [0.25, 0.25]], (1, 0)),
            [(0, 0, 0), (1, 0, 0), (1, 0, 1)],
        ),
    )
    assert np.isinf(analyze(data, waves).required_n_sigma())


def test_repeated_exact_observables_keep_joint_tolerance() -> None:
    tolerance = 1e-7
    waves = Waveset([(0, 0, "+"), (1, 0, "+")])
    value = 1 / np.sqrt(3) + 0.9 * tolerance
    data = MomentData(
        {(0, 0, 0): 1, (1, 0, 0): value, (1, 0, 1): value},
        covariance=Covariance.exact(),
    )
    assert (
        analyze(
            data, waves, tolerance=tolerance, max_combination_size=1
        ).required_n_sigma()
        == 0
    )
    assert np.isinf(analyze(data, waves, tolerance=tolerance).required_n_sigma())


def test_radius_cache_and_refinement_preserve_fixed_radius_checks() -> None:
    waves = Waveset([(0, 0), (1, 1)])
    data = disk_data(np.eye(2) * 1e-4)
    callback = Mock()
    analysis = analyze(data, waves, n_sigma=1, on_check=callback)
    search_before = analysis.search().count
    calls = callback.call_count
    required = analysis.required_n_sigma()
    stats = analysis.stats
    assert analysis.for_waves(waves).required_n_sigma() == required
    assert analysis.stats == stats
    refined = analysis.required_n_sigma(accuracy=1e-5)
    expected = (np.sqrt(0.125) - 1 / (2 * np.sqrt(3)) - 1e-8) / 0.01
    assert expected <= refined <= expected + 1e-5
    assert analysis.required_n_sigma() == refined
    assert callback.call_count == calls
    assert analysis.n_sigma == 1
    assert analysis.check().status is Status.INCOMPATIBLE
    assert analysis.search().count == search_before


@pytest.mark.parametrize("accuracy", [0, -1, np.nan, np.inf, True])
def test_invalid_accuracy_is_rejected_even_after_caching(accuracy: float) -> None:
    analysis = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0)]),
    )
    analysis.required_n_sigma()
    with pytest.raises(ValueError, match="accuracy"):
        analysis.required_n_sigma(accuracy=accuracy)


def test_failed_direct_solve_is_unresolved_and_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waves = Waveset([(0, 0), (1, 1)])
    analysis = analyze(disk_data(np.eye(2) * 1e-4), waves)
    with monkeypatch.context() as patch:
        patch.setattr(_ConicSolver, "solve", Mock(return_value=False))
        with pytest.raises(UnresolvedCompatibilityError):
            analysis.required_n_sigma()
    assert analysis.required_n_sigma() > 6


def test_tiny_positive_variance_is_never_certified_as_exact() -> None:
    analysis = analyze(disk_data(np.eye(2) * 1e-40), Waveset([(0, 0), (1, 1)]))
    try:
        radius = analysis.required_n_sigma()
    except UnresolvedCompatibilityError:
        return  # Absolute radius accuracy may exceed floating-point precision.
    assert np.isfinite(radius)


def test_tiny_positive_scalar_variance_has_a_finite_large_radius() -> None:
    waves = Waveset([(0, 0)])
    data = MomentData(
        {(0, 0): 1, (1, 0): 0.2},
        covariance=Covariance.from_uncertainties({(0, 0): 0, (1, 0): 1e-20}),
    )
    analysis = analyze(data, waves, max_combination_size=1)
    radius = analysis.required_n_sigma(accuracy=1e5)
    assert np.isfinite(radius) and radius == pytest.approx(2e19, rel=1e-6)
    assert analyze(data, waves, max_combination_size=1, n_sigma=radius).check().valid


def test_radius_solve_does_not_trust_an_inaccurate_optimizer_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waves = Waveset([(0, 0), (1, 1)])
    analysis = analyze(disk_data(np.eye(2) * 1e-4), waves)
    monkeypatch.setattr(_ConicSolver, "minimum_radius", Mock(return_value=1.0))
    radius = analysis.required_n_sigma()
    expected = (np.sqrt(0.125) - 1 / (2 * np.sqrt(3)) - 1e-8) / 0.01
    assert expected <= radius <= expected + 1e-3
