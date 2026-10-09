"""Continuous two-operator numerical ranges and checked numerical witnesses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from momentous.covariance import covariance_factor, readonly
from momentous.domain import Component
from momentous.results import (
    ComplexArray,
    PairResult,
    RealArray,
    Region2D,
    SolverStats,
    Status,
)

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass(frozen=True)
class _Pair:
    """Prepared two-component measurement geometry."""

    indices: tuple[int, int]
    point: RealArray
    root: RealArray
    whitener: RealArray | None


class _HullBatch:
    """Batch conservative attainable-hull witnesses for fixed pair geometries.

    Notes
    -----
    False entries require the individual witness or solver. Singular ellipses
    retain their exact constraints in that fallback; no small variance is dropped.
    """

    def __init__(self, pairs: Sequence[_Pair]) -> None:
        """Collect pair indices, measured points, and covariance transformations."""
        self.indices = np.asarray([pair.indices for pair in pairs], dtype=int).reshape(
            -1, 2
        )
        self.points = np.asarray([pair.point for pair in pairs]).reshape(-1, 2)
        self.full_rank = np.asarray(
            [pair.whitener is not None for pair in pairs], dtype=bool
        )
        self.whiteners = np.asarray(
            [np.eye(2) if pair.whitener is None else pair.whitener for pair in pairs]
        ).reshape(-1, 2, 2)

    def witnesses(
        self, references: RealArray, tau: float, tolerance: float
    ) -> np.ndarray:
        """Certify passes together, leaving tangent and singular cases to fallback."""
        if not len(self.indices):
            return np.zeros(0, dtype=bool)
        delta = references[:, self.indices].transpose(1, 0, 2) - self.points[:, None]
        passed = np.any(np.linalg.norm(delta, axis=2) <= tolerance, axis=1)
        remaining = np.flatnonzero(~passed)
        if not len(remaining):
            return passed
        coordinates = np.einsum(
            "pij,psj->psi", self.whiteners[remaining], delta[remaining]
        )
        angles = np.arctan2(coordinates[:, :, 1], coordinates[:, :, 0])
        order = np.argsort(angles, axis=1)
        angles = np.take_along_axis(angles, order, axis=1)
        gaps = np.diff(
            np.concatenate((angles, angles[:, :1] + 2 * np.pi), axis=1), axis=1
        )
        contained = np.max(gaps, axis=1) < np.pi - 1e-12
        passed[remaining[contained]] = True
        # Only positive definite ellipses admit this whitened distance witness.
        pending = np.flatnonzero(~contained & self.full_rank[remaining])
        if not len(pending):
            return passed
        coordinates = np.take_along_axis(
            coordinates[pending], order[pending, :, None], axis=1
        )
        edges = np.roll(coordinates, -1, axis=1) - coordinates
        lengths = np.einsum("psi,psi->ps", edges, edges)
        fractions = np.clip(
            np.divide(
                -np.einsum("psi,psi->ps", coordinates, edges),
                lengths,
                out=np.zeros_like(lengths),
                where=lengths > 0,
            ),
            0,
            1,
        )
        distances = np.linalg.norm(coordinates + fractions[:, :, None] * edges, axis=2)
        passed[remaining[pending]] = (
            np.min(distances, axis=1) < tau - 256 * np.finfo(float).eps
        )
        return passed


def _geometry(
    point: RealArray, covariance: RealArray, indices: tuple[int, int] = (0, 1)
) -> _Pair:
    """Prepare ellipse axes and a whitener only for full-rank covariance."""
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    whitener = None
    if np.all(eigenvalues > 0):
        whitener = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    return _Pair(indices, point, root, whitener)


def _arc(alpha: float, beta: float) -> list[tuple[float, float]]:
    """Split a wrapped circular arc into ordinary intervals."""
    period = 2 * np.pi
    alpha %= period
    left, right = alpha - beta, alpha + beta
    if left < 0:
        return [(0.0, right), (left + period, period)]
    if right > period:
        return [(left, period), (0.0, right - period)]
    return [(left, right)]


def _intersect(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersect closed angle intervals, including their touching endpoints."""
    # Retain touching endpoints: an extra normal is harmless, while dropping a
    # narrow interval through rounded endpoint comparisons can hide a separator.
    result = []
    for a, b in first:
        for c, d in second:
            left, right = max(a, c), min(b, d)
            if left <= right:
                result.append((left, right))
    return sorted(result)


def angular_wedge(
    points: ArrayLike, point: ArrayLike, covariance: ArrayLike, n_sigma: float = 3.0
) -> list[tuple[float, float]]:
    """Find possible separating normal angles in whitened coordinates.

    Parameters
    ----------
    points : array_like, shape (n, 2)
        Known attainable reference points.
    point : array_like, shape (2,)
        Measured point.
    covariance : array_like, shape (2, 2)
        Positive definite real covariance of the measured point.
    n_sigma : float, optional
        Radius of the uncertainty ellipse in standard deviations.

    Returns
    -------
    list of tuple of float
        Closed intervals of possible normal angles in ``[0, 2*pi]``.
        An empty list rules out separation from the reference points.

    Raises
    ------
    ValueError
        If inputs are invalid or the covariance is singular.

    Notes
    -----
    Angles describe covariance-whitened coordinates. Reference points must
    be attainable; this function alone does not certify compatibility.
    """
    if not np.isfinite(n_sigma) or n_sigma < 0:
        raise ValueError("n_sigma must be finite and nonnegative")
    p, reference = np.asarray(point, dtype=float), np.asarray(points, dtype=float)
    if (
        p.shape != (2,)
        or reference.ndim != 2
        or reference.shape[1] != 2
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "Supply a finite point of shape (2,) and references of shape (n, 2)"
        )
    factor = covariance_factor(covariance, 2)
    geometry = _geometry(p, factor @ factor.T)
    if geometry.whitener is None:
        raise ValueError("Wedge construction requires positive definite covariance")
    regions = [(0.0, 2 * np.pi)]
    for q in reference:
        delta = geometry.whitener @ (geometry.point - q)
        radius = float(np.linalg.norm(delta))
        if radius <= n_sigma:
            return []
        regions = _intersect(
            regions,
            _arc(
                float(np.arctan2(delta[1], delta[0])),
                float(np.arccos(n_sigma / radius)),
            ),
        )
    return regions


def _separator(
    operators: ComplexArray,
    pair: _Pair,
    tau: float,
    tolerance: float,
    normal: RealArray,
) -> tuple[float, float] | None:
    """Verify a separating gap with Euclidean slack and rounding allowance."""
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
        return None
    normal = normal / np.linalg.norm(normal)
    eigenvalues = np.linalg.eigvalsh(
        normal[0] * operators[0] + normal[1] * operators[1]
    )
    support = float(eigenvalues[-1])
    gap = float(
        normal @ pair.point - support - tau * np.linalg.norm(pair.root @ normal)
    )
    rounding = (
        128
        * np.finfo(float).eps
        * max(
            1.0,
            abs(support),
            float(np.linalg.norm(pair.point)),
            float(np.max(np.abs(eigenvalues))),
        )
    )
    if gap > tolerance + rounding:
        return float(normal[0]), float(normal[1])
    return None


class _ConicSolver:
    """DPP templates reused across pairs and candidates of the same dimension."""

    def __init__(self, size: int, real: bool) -> None:
        """Construct reusable primal-distance and dual-separation templates."""
        import cvxpy as cp

        self.real = real
        self.a = cp.Parameter((size, size), symmetric=real, hermitian=not real)
        self.b = cp.Parameter((size, size), symmetric=real, hermitian=not real)
        self.p = cp.Parameter(2)
        self.root = cp.Parameter((2, 2))
        self.tau = cp.Parameter(nonneg=True)
        self.rho = cp.Variable((size, size), symmetric=real, hermitian=not real)
        self.t = cp.Variable(2)
        traces = [cp.trace(self.a @ self.rho), cp.trace(self.b @ self.rho)]
        x = cp.hstack(traces if real else [cp.real(trace) for trace in traces])
        # Minimum Euclidean distance is zero exactly when the original
        # feasibility problem is feasible, and also supplies a separator.
        self.primal = cp.Problem(
            cp.Minimize(cp.norm(x - self.p - self.root @ self.t)),
            [self.rho >> 0, cp.trace(self.rho) == 1, cp.norm(self.t) <= self.tau],
        )
        self.normal = cp.Variable(2)
        s, r = cp.Variable(), cp.Variable()
        self.dual = cp.Problem(
            cp.Maximize(self.p @ self.normal - s - self.tau * r),
            [
                s * np.eye(size) - self.normal[0] * self.a - self.normal[1] * self.b
                >> 0,
                cp.norm(self.root @ self.normal) <= r,
                cp.norm(self.normal) <= 1,
            ],
        )

    def solve(self, problem: cp.Problem, stats: SolverStats) -> bool:
        """Attempt Clarabel optimization and record time even on failure."""
        import cvxpy as cp

        start = perf_counter()
        stats.solver_calls += 1
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
                tol_feas=1e-10,
                max_iter=250,
            )
            return True
        except cp.error.SolverError:
            return False
        finally:
            stats.solver_seconds += perf_counter() - start

    def check(
        self,
        operators: ComplexArray,
        pair: _Pair,
        tau: float,
        tolerance: float,
        stats: SolverStats,
    ) -> PairResult:
        """Repair primal witnesses and verify primal or dual separating normals."""
        self.a.value = operators[0].real if self.real else operators[0]
        self.b.value = operators[1].real if self.real else operators[1]
        self.p.value, self.root.value, self.tau.value = pair.point, pair.root, tau
        self.rho.value = None
        self.t.value = None
        self.solve(self.primal, stats)
        if self.rho.value is not None and self.t.value is not None:
            rho = np.asarray(self.rho.value, dtype=complex)
            if np.all(np.isfinite(rho)) and np.all(np.isfinite(self.t.value)):
                eigenvalues, vectors = np.linalg.eigh((rho + rho.conj().T) / 2)
                eigenvalues = np.maximum(eigenvalues, 0)
                if eigenvalues.sum() > 0:
                    rho = (
                        vectors * (eigenvalues / eigenvalues.sum())
                    ) @ vectors.conj().T
                    x = np.einsum("aij,ji->a", operators, rho).real
                    y = pair.point + _ellipse_projection(x - pair.point, pair.root, tau)
                    if np.linalg.norm(x - y) <= tolerance:
                        return PairResult(Status.COMPATIBLE)
                    normal = _separator(operators, pair, tau, tolerance, y - x)
                    if normal is not None:
                        return PairResult(Status.INCOMPATIBLE, normal=normal)
        self.normal.value = None
        self.solve(self.dual, stats)
        if self.normal.value is not None:
            normal = _separator(
                operators,
                pair,
                tau,
                tolerance,
                np.asarray(self.normal.value, dtype=float),
            )
            if normal is not None:
                return PairResult(Status.INCOMPATIBLE, normal=normal)
        return PairResult(Status.UNRESOLVED)


def _ellipse_projection(delta: RealArray, root: RealArray, tau: float) -> RealArray:
    """Nearest point in a centered ellipse, including points and line segments."""
    from scipy.optimize import brentq

    radii, vectors = np.linalg.eigh(root)
    axes_squared = (tau * np.maximum(radii, 0)) ** 2
    active = axes_squared > 0
    coordinates = vectors.T @ delta
    projected = np.zeros(2)
    if np.any(active):
        axes, values = axes_squared[active], coordinates[active]
        if np.sum(values**2 / axes) <= 1:
            projected[active] = values
        else:
            scale = float(np.max(axes))
            axes = axes / scale
            values_scaled = values / np.sqrt(scale)
            upper = float(np.linalg.norm(np.sqrt(axes) * values_scaled))
            multiplier = brentq(
                lambda x: float(np.sum(axes * values_scaled**2 / (axes + x) ** 2) - 1),
                0,
                upper,
                xtol=1e-14,
            )
            projected[active] = axes * values / (axes + multiplier)
    return vectors @ projected


def _inner_hull_witness(
    operators: ComplexArray, pair: _Pair, tau: float, tolerance: float
) -> bool:
    """Certify passes from an inner hull without rejecting on a missed hull.

    Each vertex is produced by an actual normalized amplitude. Convexity of the
    two-operator numerical range makes every point in their hull attainable.
    """
    angles = np.arange(8) * np.pi / 4
    combinations = (
        np.cos(angles)[:, None, None] * operators[0]
        + np.sin(angles)[:, None, None] * operators[1]
    )
    _, vectors = np.linalg.eigh(combinations)
    z = vectors[:, :, -1]
    points = np.einsum("si,aij,sj->sa", z.conj(), operators, z).real
    return _point_hull_witness(points, pair, tau, tolerance)


def _point_hull_witness(
    points: RealArray, pair: _Pair, tau: float, tolerance: float
) -> bool:
    """Certify a pass from an attainable convex hull, never an exclusion."""
    padding = 256 * np.finfo(float).eps
    delta = points - pair.point
    if np.any(np.linalg.norm(delta, axis=1) <= tolerance):
        return True
    coordinates = delta if pair.whitener is None else delta @ pair.whitener.T
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    order = np.argsort(angles)
    angles = angles[order]
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    # The origin lies in the hull exactly when the directions cannot fit in
    # an open semicircle. A strict margin keeps this a conservative witness.
    if np.max(gaps) < np.pi - 1e-12:
        return True
    points, coordinates = points[order], coordinates[order]
    edges = np.roll(coordinates, -1, axis=0) - coordinates
    lengths = np.einsum("ij,ij->i", edges, edges)
    fractions = np.clip(
        np.divide(
            -np.einsum("ij,ij->i", coordinates, edges),
            lengths,
            out=np.zeros(len(points)),
            where=lengths > 0,
        ),
        0,
        1,
    )
    distances = np.linalg.norm(coordinates + fractions[:, None] * edges, axis=1)
    index = int(np.argmin(distances))
    if pair.whitener is not None and distances[index] < tau - padding:
        return True
    candidate = points[index] + fractions[index] * (
        points[(index + 1) % len(points)] - points[index]
    )
    projected = pair.point + _ellipse_projection(candidate - pair.point, pair.root, tau)
    return bool(np.linalg.norm(candidate - projected) <= tolerance)


def _check_pair(
    operators: ComplexArray,
    pair: _Pair,
    tau: float,
    tolerance: float,
    stats: SolverStats,
    solvers: dict[tuple[int, bool], _ConicSolver],
) -> PairResult:
    """Check a continuous pair using shortcuts followed by verified conic solves."""
    stats.pair_checks += 1
    points = np.diagonal(operators, axis1=1, axis2=2).real.T
    if pair.whitener is not None:
        regions = [(0.0, 2 * np.pi)]
        for point in points:
            delta = pair.whitener @ (pair.point - point)
            radius = float(np.linalg.norm(delta))
            if radius <= tau:
                stats.wedge_passes += 1
                return PairResult(Status.COMPATIBLE)
            # The original ellipse's wedge is a superset of normals that could
            # separate even after numerical slack. Keep endpoints conservatively.
            regions = _intersect(
                regions,
                _arc(
                    float(np.arctan2(delta[1], delta[0])),
                    float(np.arccos(tau / radius)) + 1e-12,
                ),
            )
            if not regions:
                stats.wedge_passes += 1
                return PairResult(Status.COMPATIBLE)
        for left, right in regions:
            angle = (left + right) / 2
            normal = _separator(
                operators,
                pair,
                tau,
                tolerance,
                pair.whitener.T @ np.array([np.cos(angle), np.sin(angle)]),
            )
            if normal is not None:
                stats.midpoint_exclusions += 1
                return PairResult(Status.INCOMPATIBLE, normal=normal)
    # A singleton numerical range needs no optimization, even for singular V.
    if operators.shape[1] == 1:
        y = pair.point + _ellipse_projection(points[0] - pair.point, pair.root, tau)
        if np.linalg.norm(points[0] - y) <= tolerance:
            return PairResult(Status.COMPATIBLE)
        normal = _separator(operators, pair, tau, tolerance, y - points[0])
        if normal is not None:
            return PairResult(Status.INCOMPATIBLE, normal=normal)
    if _inner_hull_witness(operators, pair, tau, tolerance):
        stats.hull_passes += 1
        return PairResult(Status.COMPATIBLE)
    key = (operators.shape[1], bool(np.all(operators.imag == 0)))
    if key not in solvers:
        solvers[key] = _ConicSolver(*key)
    return solvers[key].check(operators, pair, tau, tolerance, stats)


def check_pair(
    first: ArrayLike,
    second: ArrayLike,
    point: ArrayLike,
    covariance: ArrayLike,
    *,
    n_sigma: float = 3.0,
    tolerance: float = 1e-8,
) -> PairResult:
    """Check continuous ellipse intersection for arbitrary Hermitian operators.

    Parameters
    ----------
    first, second : array_like, shape (n, n)
        Nonempty finite Hermitian operators in a common normalized basis.
    point : array_like, shape (2,)
        Measured real scalar components.
    covariance : array_like, shape (2, 2)
        Real positive semidefinite covariance, including singular matrices.
    n_sigma : float, default 3
        Covariance ellipse radius multiplier, not a global confidence level.
    tolerance : float, default 1e-8
        Absolute Euclidean slack in normalized moment coordinates.

    Returns
    -------
    PairResult
        Verified compatible/incompatible status or an unresolved decision.

    Examples
    --------
    >>> check_pair([[0]], [[0]], [0, 0], [[0, 0], [0, 0]]).valid
    True
    """
    _settings(n_sigma, tolerance)
    operators = np.asarray([first, second], dtype=np.complex128)
    if (
        operators.ndim != 3
        or operators.shape[1] != operators.shape[2]
        or not operators.shape[1]
    ):
        raise ValueError("Operators must be nonempty square matrices of equal size")
    if not np.all(np.isfinite(operators)) or not np.allclose(
        operators, operators.conj().transpose(0, 2, 1), rtol=0, atol=1e-14
    ):
        raise ValueError("Operators must be finite and Hermitian")
    operators = (operators + operators.conj().transpose(0, 2, 1)) / 2
    if np.iscomplexobj(point):
        raise ValueError("Point must contain two finite real components")
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError("Point must contain two finite real components")
    factor = covariance_factor(covariance, 2)
    return _check_pair(
        operators,
        _geometry(p, factor @ factor.T),
        n_sigma,
        tolerance,
        SolverStats(),
        {},
    )


def _settings(n_sigma: float, tolerance: float) -> None:
    """Validate finite numerical settings shared by all checking paths."""
    if not np.isfinite(n_sigma) or n_sigma < 0:
        raise ValueError("n_sigma must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")


def _boundary_hull(points: RealArray) -> RealArray:
    """Order attainable vertices, collapsing collinear and point degeneracies."""
    ordered = np.unique(points, axis=0)
    if len(ordered) <= 1:
        return ordered
    scale = float(np.max(np.ptp(ordered, axis=0)))
    rounding = 128 * np.finfo(float).eps * scale**2

    def half(vertices: RealArray) -> list[RealArray]:
        """Build one monotone chain with a scale-aware collinearity allowance."""
        chain: list[RealArray] = []
        for point in vertices:
            while len(chain) >= 2:
                first, second = chain[-1] - chain[-2], point - chain[-1]
                cross = float(first[0] * second[1] - first[1] * second[0])
                if cross > rounding:
                    break
                chain.pop()
            chain.append(point)
        return chain

    lower, upper = half(ordered), half(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def region_data(
    operators: ComplexArray,
    components: tuple[Component, Component],
    point: RealArray,
    covariance: RealArray,
    n_sigma: float,
    n_directions: int,
) -> Region2D:
    """Compute support values and attainable extremizers, including flat faces."""
    from numbers import Integral

    if (
        isinstance(n_directions, bool)
        or not isinstance(n_directions, Integral)
        or n_directions < 4
    ):
        raise ValueError("n_directions must be an integer of at least four")
    angles = np.arange(n_directions) * (2 * np.pi / n_directions)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    support = np.empty(n_directions)
    points: list[RealArray] = []
    for index, normal in enumerate(normals):
        matrix = normal[0] * operators[0] + normal[1] * operators[1]
        eigenvalues, vectors = np.linalg.eigh(matrix)
        support[index] = eigenvalues[-1]
        slack = (
            64
            * np.finfo(float).eps
            * len(matrix)
            * max(1.0, np.max(np.abs(eigenvalues)))
        )
        space = vectors[:, eigenvalues >= eigenvalues[-1] - slack]
        # In a degenerate supporting eigenspace, tangent extrema give both
        # endpoints of the exposed face instead of one arbitrary eigenvector.
        tangent = -normal[1] * operators[0] + normal[0] * operators[1]
        _, face_vectors = np.linalg.eigh(space.conj().T @ tangent @ space)
        for column in (0, -1):
            state = space @ face_vectors[:, column]
            points.append(np.einsum("i,aij,j->a", state.conj(), operators, state).real)
    return Region2D(
        components,
        readonly(point),
        readonly(covariance),
        readonly(normals),
        readonly(support),
        readonly(_boundary_hull(np.asarray(points, dtype=np.float64))),
        n_sigma,
    )
