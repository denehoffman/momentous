"""Spectral bounds and covariance propagation behind the validation interface."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from moments.bounds import Bound, ComplexBound, get_bound
from moments.cg import c_matrix, polarized_c_matrix
from moments.waves import (
    LinearlyPolarizedMoment,
    Measurement,
    Moment,
    PartialWave,
    PolarizedMeasurement,
    ReflectivityPartialWave,
)


@dataclass
class ValidationFailure:
    """One rejected scalar projection and the measurements it involves.

    ``coefficients`` uses the same scalar order as the covariance matrix.
    It identifies the failing projection, whose value, standard deviation,
    and distance from the allowed interval in standard deviations are returned.
    ``bounds`` is the real interval for this projection.
    ``sigma_distance`` is infinite for an excluded exact measurement.
    """

    measurements: tuple[Measurement | PolarizedMeasurement, ...]
    bounds: Bound
    coefficients: tuple[float, ...]
    value: float
    uncertainty: float
    sigma_distance: float
    measurement: Measurement | PolarizedMeasurement | None = None


@dataclass
class ValidationResult:
    """All rejected projections, with a deduplicated list of involved measurements.

    ``invalid_measurements`` includes both individual failures and measurements
    involved in failing pairwise combinations, in input order. A pairwise failure
    concerns their combination, not necessarily either measurement on its own.
    The legacy ``measurement`` and ``bounds`` attributes expose the first failure.
    """

    valid: bool
    measurement: Measurement | PolarizedMeasurement | None = None
    bounds: ComplexBound | Bound | None = None
    failures: tuple[ValidationFailure, ...] = ()
    invalid_measurements: tuple[Measurement | PolarizedMeasurement, ...] = ()


def _covariance_factor(covariance: ArrayLike, size: int) -> NDArray[np.float64]:
    """Validate a real PSD covariance and factor it, including singular cases."""
    raw = np.asarray(covariance)
    if np.iscomplexobj(raw):
        raise ValueError("Covariance must be a real matrix of scalar components")
    cov = np.array(raw, dtype=float, copy=True)
    if cov.shape != (size, size):
        raise ValueError(f"Covariance must have shape ({size}, {size})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance must contain only finite values")
    if size == 0:
        return cov
    scale = float(np.max(np.abs(cov)))
    tolerance = 64 * np.finfo(float).eps * size * scale
    if np.max(np.abs(cov - cov.T)) > tolerance:
        raise ValueError("Covariance must be symmetric")
    cov = (cov + cov.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(np.diag(cov) < 0) or eigenvalues[0] < -tolerance:
        raise ValueError("Covariance must be positive semidefinite")
    # Factor norms avoid cancellation in variances of strongly correlated data.
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))


def _inputs(
    waves: Sequence[PartialWave | ReflectivityPartialWave],
    measurements: Sequence[Measurement | PolarizedMeasurement],
    covariance: ArrayLike | None,
) -> tuple[bool, NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Flatten measured observables and put their operators in a common metric."""
    if not waves:
        raise ValueError("A waveset must contain at least one partial wave")
    polarized = isinstance(waves[0], ReflectivityPartialWave)
    polarized_waves: list[ReflectivityPartialWave] = []
    unpolarized_waves: list[PartialWave] = []
    for wave in waves:
        if polarized:
            if not isinstance(wave, ReflectivityPartialWave):
                raise ValueError("A waveset cannot mix polarized and unpolarized waves")
            polarized_waves.append(wave)
        else:
            if not isinstance(wave, PartialWave):
                raise ValueError("A waveset cannot mix polarized and unpolarized waves")
            unpolarized_waves.append(wave)
    keys = [
        (
            wave.L.value,
            wave.M.value,
            wave.r.value if isinstance(wave, ReflectivityPartialWave) else None,
        )
        for wave in waves
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("A waveset cannot contain duplicate partial waves")

    scalar_values: list[float] = []
    scalar_errors: list[float] = []
    operators: list[NDArray[np.float64] | NDArray[np.complex128]] = []
    for item in measurements:
        if polarized:
            if not isinstance(item, PolarizedMeasurement):
                raise ValueError("Measurements must match the waveset's polarization")
            if not isinstance(item.moment, LinearlyPolarizedMoment):
                raise ValueError("Polarized measurements need polarized moments")
            if np.iscomplexobj(item.value):
                raise ValueError("Polarized measurements must be real observables")
            scalar_values.append(item.value)
            operators.append(polarized_c_matrix(item.moment, polarized_waves))
            if covariance is None:
                if item.uncertainty is not None and np.iscomplexobj(item.uncertainty):
                    raise ValueError("Polarized uncertainties must be real")
                scalar_errors.append(
                    0 if item.uncertainty is None else item.uncertainty
                )
        else:
            if not isinstance(item, Measurement):
                raise ValueError("Measurements must match the waveset's polarization")
            if not isinstance(item.moment, Moment):
                raise ValueError("Unpolarized measurements need unpolarized moments")
            value = complex(item.value)
            scalar_values.extend((value.real, value.imag))
            operators.extend(c_matrix(item.moment, unpolarized_waves))
            if covariance is None:
                error = 0j if item.uncertainty is None else complex(item.uncertainty)
                scalar_errors.extend((error.real, error.imag))

    values = np.asarray(scalar_values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Measured values must be finite")
    if covariance is None:
        errors = np.asarray(scalar_errors, dtype=np.float64)
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError(
                "Uncertainties must be finite, nonnegative standard deviations"
            )
        factor = np.diag(errors)
    else:
        factor = _covariance_factor(covariance, len(values))

    matrices = np.asarray(operators, dtype=np.complex128).reshape(
        -1, len(waves), len(waves)
    )
    if polarized:
        normalization = polarized_c_matrix(
            LinearlyPolarizedMoment(0, 0, 0), polarized_waves
        )
        eigenvalues, eigenvectors = np.linalg.eigh(normalization)
        if np.any(eigenvalues <= 0):
            raise ValueError("H^0(0, 0) must define a positive definite normalization")
        transform = eigenvectors / np.sqrt(eigenvalues)
        matrices = transform.conj().T @ matrices @ transform
    return polarized, values, factor, matrices


def _failure(
    coefficients: NDArray[np.float64],
    values: NDArray[np.float64],
    factor: NDArray[np.float64],
    bound: Bound,
    n_sigma: float,
    measurement: Measurement | PolarizedMeasurement,
) -> ValidationFailure | None:
    value = float(coefficients @ values)
    uncertainty = float(np.linalg.norm(coefficients @ factor))
    distance = max(bound.lower - value, value - bound.upper, 0.0)
    tolerance = (
        64
        * np.finfo(float).eps
        * max(
            1.0, abs(value), abs(bound.lower), abs(bound.upper), n_sigma * uncertainty
        )
    )
    if distance <= n_sigma * uncertainty + tolerance:
        return None
    return ValidationFailure(
        measurements=(measurement,),
        measurement=measurement,
        bounds=bound,
        coefficients=tuple(float(weight) for weight in coefficients),
        value=value,
        uncertainty=uncertainty,
        sigma_distance=distance / uncertainty if uncertainty else float("inf"),
    )


@overload
def validate(
    waves: Sequence[PartialWave],
    moment_measurements: Sequence[Measurement],
    *,
    covariance: ArrayLike | None = None,
    n_sigma: float = 1.0,
    pairwise: bool = True,
    n_angles: int = 360,
) -> ValidationResult: ...


@overload
def validate(
    waves: Sequence[ReflectivityPartialWave],
    moment_measurements: Sequence[PolarizedMeasurement],
    *,
    covariance: ArrayLike | None = None,
    n_sigma: float = 1.0,
    pairwise: bool = True,
    n_angles: int = 360,
) -> ValidationResult: ...


def validate(
    waves: Sequence[PartialWave | ReflectivityPartialWave],
    moment_measurements: Sequence[Measurement | PolarizedMeasurement],
    *,
    covariance: ArrayLike | None = None,
    n_sigma: float = 1.0,
    pairwise: bool = True,
    n_angles: int = 360,
) -> ValidationResult:
    """Screen a waveset against normalized moments using covariance-aware bounds.

    Unpolarized covariance has shape (2*n, 2*n), ordered
    [Re H_0, Im H_0, Re H_1, Im H_1, ...] in measurement order.
    Polarized covariance has shape (n, n), ordered like the real observables
    (including Im H^2). It must describe the *normalized* measurements.
    A supplied covariance supersedes measurement uncertainties. Without it,
    uncertainties give independent standard deviations; omitted errors are zero.

    Each projection is allowed n_sigma * sqrt(w.T @ covariance @ w) away from
    its spectral interval. Pairwise mode tests every pair of scalar components,
    at n_angles equally spaced directions in [0, pi), using both interval ends.
    It includes mixed Re/Im and mixed polarized-variant projections.

    A failure excludes the waveset at the requested projected error tolerance.
    Success means the sampled necessary conditions passed; it does not prove a
    simultaneous amplitude solution or supply a joint confidence level/p-value.
    Larger n_angles reduces gaps between checked pairwise directions. Floating
    point comparisons include a small numerical tolerance at interval edges.
    All failed individual components and sampled pairwise directions are collected
    in result.failures; result.invalid_measurements lists the measurements involved.
    """
    if not np.isfinite(n_sigma) or n_sigma < 0:
        raise ValueError("n_sigma must be finite and nonnegative")
    if pairwise and (
        not isinstance(n_angles, Integral) or isinstance(n_angles, bool) or n_angles < 4
    ):
        raise ValueError("n_angles must be an integer of at least 4")
    partial_waves, measurements = list(waves), list(moment_measurements)
    polarized, values, factor, matrices = _inputs(
        partial_waves, measurements, covariance
    )
    size = len(values)
    stride = 1 if polarized else 2
    failures: list[ValidationFailure] = []
    invalid_indices: set[int] = set()
    legacy_bounds: Bound | ComplexBound | None = None
    for index, measurement in enumerate(measurements):
        bounds = [
            get_bound(mat) for mat in matrices[stride * index : stride * (index + 1)]
        ]
        for part, bound in enumerate(bounds):
            coefficients = np.zeros(size)
            coefficients[stride * index + part] = 1
            result = _failure(coefficients, values, factor, bound, n_sigma, measurement)
            if result is not None:
                if legacy_bounds is None:
                    legacy_bounds = bound if polarized else ComplexBound(*bounds)
                failures.append(result)
                invalid_indices.add(index)

    if pairwise and size > 1:
        angles = np.arange(n_angles) * np.pi / n_angles
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        # Axis-aligned directions are already checked individually. Omitting them
        # avoids duplicate diagnostics that name a second, unused measurement.
        directions = directions[np.all(np.abs(directions) > 1e-14, axis=1)]
        for first, second in combinations(range(size), 2):
            operators = (
                directions[:, 0, None, None] * matrices[first]
                + directions[:, 1, None, None] * matrices[second]
            )
            eigenvalues = np.linalg.eigvalsh(operators)
            projected = directions @ values[[first, second]]
            errors = np.linalg.norm(directions @ factor[[first, second]], axis=1)
            distances = np.maximum(
                np.maximum(
                    eigenvalues[:, 0] - projected, projected - eigenvalues[:, -1]
                ),
                0,
            )
            tolerances = (
                64
                * np.finfo(float).eps
                * np.maximum.reduce(
                    [
                        np.ones(len(directions)),
                        np.abs(projected),
                        np.max(np.abs(eigenvalues), axis=1),
                        n_sigma * errors,
                    ]
                )
            )
            failed = np.flatnonzero(distances > n_sigma * errors + tolerances)
            for angle in failed:
                coefficients = np.zeros(size)
                coefficients[[first, second]] = directions[angle]
                bound = Bound(
                    float(eigenvalues[angle, 0]), float(eigenvalues[angle, -1])
                )
                indices = sorted({first // stride, second // stride})
                failures.append(
                    ValidationFailure(
                        measurements=tuple(measurements[index] for index in indices),
                        bounds=bound,
                        coefficients=tuple(float(weight) for weight in coefficients),
                        value=float(projected[angle]),
                        uncertainty=float(errors[angle]),
                        sigma_distance=float(distances[angle] / errors[angle])
                        if errors[angle]
                        else float("inf"),
                    )
                )
                invalid_indices.update(indices)
    return ValidationResult(
        valid=not failures,
        measurement=failures[0].measurement if failures else None,
        bounds=legacy_bounds
        if legacy_bounds is not None
        else failures[0].bounds
        if failures
        else None,
        failures=tuple(failures),
        invalid_measurements=tuple(
            measurements[index] for index in sorted(invalid_indices)
        ),
    )
