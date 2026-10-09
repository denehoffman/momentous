from unittest.mock import Mock

import numpy as np
import pytest

from momentous import (
    CheckResult,
    Covariance,
    MomentData,
    Status,
    UnresolvedCompatibilityError,
    Waveset,
    analyze,
)
from momentous.geometry import (
    _ConicSolver,
    _geometry,
    _HullBatch,
    _point_hull_witness,
    check_pair,
    region_data,
)
from momentous.results import ComplexArray


@pytest.fixture
def disk_operators() -> tuple[ComplexArray, ComplexArray]:
    return np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]])


def test_batched_hull_witnesses_preserve_tangency_and_exact_constraints() -> None:
    references = np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]])
    pairs = [
        _geometry(np.array(point), np.array(covariance))
        for point, covariance in [
            ([0, 0], [[0, 0], [0, 0]]),
            ([2, 0], [[1, 0], [0, 1]]),
            ([2.01, 0], [[1, 0], [0, 1]]),
            ([1.01, 0], [[0, 0], [0, 1]]),
            ([1.01, 0], [[1e-40, 0], [0, 1]]),
        ]
    ]
    witnesses = _HullBatch(pairs).witnesses(references, 1, 1e-8)
    verdicts = [
        bool(passed) or _point_hull_witness(references, pair, 1, 1e-8)
        for passed, pair in zip(witnesses, pairs, strict=True)
    ]
    assert verdicts == [True, True, False, False, False]


@pytest.mark.parametrize(
    "point,expected",
    [([0.6, 0.6], True), ([0.8, 0.8], False), ([1, 0], True)],
    ids=["interior", "exterior", "boundary"],
)
def test_continuous_disk_contains_exact_points(
    disk_operators: tuple[ComplexArray, ComplexArray],
    point: list[float],
    expected: bool,
) -> None:
    assert check_pair(*disk_operators, point, np.zeros((2, 2))).valid is expected


def test_covariance_ellipse_tangency_and_a_separated_neighbor(
    disk_operators: tuple[ComplexArray, ComplexArray],
) -> None:
    assert check_pair(*disk_operators, [1.1, 0], np.eye(2) * 0.01, n_sigma=1).valid
    assert not check_pair(
        *disk_operators, [1.101, 0], np.eye(2) * 0.01, n_sigma=1
    ).valid


def test_continuous_separator_is_found_between_coarse_display_angles(
    disk_operators: tuple[ComplexArray, ComplexArray],
) -> None:
    angle = np.deg2rad(0.5)
    point = (1 + 1e-6) * np.array([np.cos(angle), np.sin(angle)])
    angles = np.deg2rad(np.arange(360))
    sampled = np.column_stack((np.cos(angles), np.sin(angles)))
    assert np.all(sampled @ point < 1)
    result = check_pair(*disk_operators, point, np.zeros((2, 2)))
    assert result.status is Status.INCOMPATIBLE
    assert result.normal is not None
    normal = np.asarray(result.normal)
    assert (
        normal @ point
        - np.linalg.eigvalsh(
            normal[0] * disk_operators[0] + normal[1] * disk_operators[1]
        )[-1]
        > 0
    )


def test_singular_covariance_retains_exact_null_direction(
    disk_operators: tuple[ComplexArray, ComplexArray],
) -> None:
    assert not check_pair(*disk_operators, [0, 1.1], np.diag([1, 0]), n_sigma=1).valid
    assert check_pair(*disk_operators, [1.1, 0], np.diag([0.04, 0]), n_sigma=1).valid


def test_tiny_positive_variance_is_not_replaced_by_an_exact_constraint(
    disk_operators: tuple[ComplexArray, ComplexArray],
) -> None:
    point = [1 + 1e-10, 0]
    assert check_pair(
        *disk_operators, point, np.diag([1e-20, 1]), n_sigma=1, tolerance=1e-12
    ).valid
    assert not check_pair(
        *disk_operators, point, np.diag([0, 1]), n_sigma=1, tolerance=1e-12
    ).valid


def test_solver_failure_is_explicitly_unresolved(
    disk_operators: tuple[ComplexArray, ComplexArray], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_ConicSolver, "solve", Mock(return_value=False))
    result = check_pair(*disk_operators, [0.95, 0.28], np.zeros((2, 2)))
    assert result.status is Status.UNRESOLVED
    with pytest.raises(UnresolvedCompatibilityError):
        _ = result.valid
    with pytest.raises(UnresolvedCompatibilityError):
        bool(result)
    with pytest.raises(UnresolvedCompatibilityError):
        bool(CheckResult(result.status))
    assert not CheckResult(Status.INCOMPATIBLE)
    assert CheckResult(Status.COMPATIBLE)


def test_complex_moment_region_has_analytic_disk_support_and_immutable_data() -> None:
    result = analyze(
        MomentData({(0, 0): 1, (1, 1): 0j}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 1)]),
    )
    region = result.region((1, 1, "real"), (1, 1, "imag"), n_directions=16)
    radius = 1 / (2 * np.sqrt(3))
    np.testing.assert_allclose(region.support, radius, atol=1e-15)
    np.testing.assert_allclose(
        np.linalg.norm(region.boundary, axis=1), radius, atol=1e-15
    )
    np.testing.assert_allclose(
        region.normals @ region.boundary.T <= region.support[:, None] + 1e-15, True
    )
    with pytest.raises(ValueError):
        region.boundary[0, 0] = 100


def test_region_handles_point_segment_and_flat_faces() -> None:
    point = analyze(
        MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0)]),
    ).region((1, 0, "real"), (1, 0, "imag"))
    np.testing.assert_array_equal(point.boundary, [[0, 0]])
    segment = analyze(
        MomentData({(0, 0): 1, (2, 0): 0}, covariance=Covariance.exact()),
        Waveset([(0, 0), (1, 0)]),
    ).region((2, 0, "real"), (2, 0, "imag"), n_directions=8)
    np.testing.assert_allclose(segment.boundary, [[0, 0], [0.4, 0]], atol=1e-15)


def test_region_retains_both_extremizers_of_degenerate_flat_faces() -> None:
    vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    operators = np.array([np.diag(vertices[:, 0]), np.diag(vertices[:, 1])], complex)
    region = region_data(
        operators,
        ((1, 0, "real"), (2, 0, "real")),
        np.zeros(2),
        np.zeros((2, 2)),
        3,
        4,
    )
    np.testing.assert_allclose(region.boundary, vertices, atol=1e-15)
    np.testing.assert_allclose(region.support, 1, atol=1e-15)


def test_boundary_resolution_does_not_change_compatibility() -> None:
    result = analyze(
        MomentData(
            {(0, 0): 1, (1, 1): (0.25 + 0.25j)},
            covariance=Covariance.exact(),
        ),
        Waveset([(0, 0), (1, 1)]),
    )
    before = result.check().status
    coarse = result.region((1, 1, "real"), (1, 1, "imag"), n_directions=4)
    fine = result.region((1, 1, "real"), (1, 1, "imag"), n_directions=64)
    assert len(coarse.boundary) < len(fine.boundary)
    assert before is result.check().status is Status.INCOMPATIBLE
