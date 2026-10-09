"""Prepared candidate evaluation, witness reuse, and monotonic inference."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import fields
from itertools import combinations

import numpy as np

from momentous.data import PreparedData
from momentous.domain import Waveset
from momentous.geometry import (
    _check_pair,
    _ConicSolver,
    _geometry,
    _HullBatch,
    _point_hull_witness,
)
from momentous.results import (
    Bound,
    CheckResult,
    ComplexArray,
    Diagnostic,
    ProjectionFailure,
    RealArray,
    SolverStats,
    Status,
    UnresolvedPair,
)


class Engine:
    """Own candidate caches and numerical templates for fixed analysis inputs."""

    def __init__(
        self,
        pool: Waveset,
        data: PreparedData,
        n_sigma: float,
        tolerance: float,
        max_combination_size: int,
        on_check: Callable[[Waveset, CheckResult], None] | None = None,
    ) -> None:
        """Prepare shared covariance geometry and canonical wave indices."""
        self.pool, self.data = pool, data
        self.n_sigma, self.tolerance = n_sigma, tolerance
        self.max_combination_size = max_combination_size
        self.on_check = on_check
        self.errors = np.linalg.norm(data.factor, axis=1)
        self.stats = SolverStats()
        self._indices = {wave: index for index, wave in enumerate(pool.waves)}
        self._cache: dict[int, CheckResult] = {}
        self._detailed: dict[int, CheckResult] = {}
        self._passing: set[int] = set()
        self._failing: set[int] = set()
        self._solvers: dict[tuple[int, bool], _ConicSolver] = {}
        active: list[int] = []
        for index in range(len(data.values)):
            matrix = data.operators[index]
            constant = float((matrix[0, 0] / data.metric[0, 0]).real)
            redundant = (
                data.values[index] == constant
                and self.errors[index] == 0
                and np.array_equal(matrix, constant * data.metric)
            )
            if not redundant:
                active.append(index)
        self._active = active
        groups: dict[int, int] = {}
        representatives: list[int] = []
        for index in active:
            for representative in representatives:
                if any(
                    data.values[index] == sign * data.values[representative]
                    and np.array_equal(
                        data.operators[index], sign * data.operators[representative]
                    )
                    and np.array_equal(
                        data.factor[index], sign * data.factor[representative]
                    )
                    and np.array_equal(
                        data.covariance[index], sign * data.covariance[representative]
                    )
                    for sign in (1, -1)
                ):
                    groups[index] = representative
                    break
            else:
                groups[index] = index
                representatives.append(index)
        seen: set[tuple[int, int]] = set()
        indices = []
        for i, j in combinations(active, 2):
            key = (min(groups[i], groups[j]), max(groups[i], groups[j]))
            if key not in seen:
                seen.add(key)
                indices.append((i, j))
        # Keep one repeated-component pair as well: its joint Euclidean tolerance
        # is stricter than either scalar tolerance, even with exact correlation.
        self.pairs = (
            [
                _geometry(
                    data.values[[i, j]], data.covariance[np.ix_([i, j], [i, j])], (i, j)
                )
                for i, j in indices
            ]
            if max_combination_size == 2
            else []
        )
        self._hulls = _HullBatch(self.pairs)

    def mask(self, waves: Waveset) -> int:
        """Validate a candidate's membership and map it to a private bitmask."""
        if not isinstance(waves, Waveset):
            raise TypeError("Candidates must be Waveset objects")
        if len(waves) and waves.polarized != self.pool.polarized:
            raise ValueError("Candidate polarization differs from the pool")
        try:
            return sum(1 << self._indices[wave] for wave in waves.waves)
        except KeyError as error:
            raise ValueError("Candidate contains a wave outside the pool") from error

    def members(self, mask: int) -> Waveset:
        """Construct an immutable candidate from a private bitmask."""
        return Waveset(
            (wave for index, wave in enumerate(self.pool.waves) if mask & (1 << index)),
            polarized=self.pool.polarized,
        )

    def operators(self, mask: int) -> ComplexArray:
        """Restrict raw operators before transforming the candidate metric."""
        indices = [i for i in range(len(self.pool)) if mask & (1 << i)]
        if not indices:
            raise ValueError("Empty candidates have no moment operators or geometry")
        raw = self.data.operators[:, indices][:, :, indices]
        metric = self.data.metric[np.ix_(indices, indices)]
        if np.array_equal(metric, np.diag(np.diag(metric))):
            inverse = 1 / np.sqrt(np.diag(metric).real)
            return np.asarray(
                raw * inverse[None, :, None] * inverse[None, None, :],
                dtype=np.complex128,
            )
        eigenvalues, vectors = np.linalg.eigh(metric)
        transform = (vectors / np.sqrt(eigenvalues)) @ vectors.conj().T
        return np.asarray(transform @ raw @ transform, dtype=np.complex128)

    def check(self, mask: int, *, diagnostics: bool = False) -> CheckResult:
        """Check or infer a status; evaluate actual candidates for full diagnostics."""
        if diagnostics:
            if mask not in self._detailed:
                actual = self.evaluate(mask, diagnostics=True)
                certified = self._cache.get(mask)
                if actual.status is Status.UNRESOLVED and certified is not None:
                    actual = CheckResult(certified.status, actual.diagnostics)
                self._detailed[mask] = actual
                self._remember(mask, actual)
                self._notify(mask, actual)
            return self._detailed[mask]
        cached = self.lookup(mask)
        if cached is not None:
            return cached
        result = self.evaluate(mask)
        self._remember(mask, result)
        self._notify(mask, result)
        return result

    def _notify(self, mask: int, result: CheckResult) -> None:
        """Deliver a completed evaluation after updating coordinating caches."""
        if self.on_check is not None:
            self.on_check(self.members(mask), result)

    def lookup(self, mask: int) -> CheckResult | None:
        """Return cached or inferred decisions without starting numerical work."""
        cached = self._cache.get(mask)
        if cached is not None and cached.status is not Status.UNRESOLVED:
            self.stats.cache_hits += 1
            return self._cache[mask]
        status: Status | None = None
        if any(known & mask == known for known in self._passing):
            status = Status.COMPATIBLE
        elif any(known | mask == known for known in self._failing):
            status = Status.INCOMPATIBLE
        if status is not None:
            self.stats.inferred_checks += 1
            result = CheckResult(status)
            self._cache[mask] = result
            return result
        if cached is not None:
            return cached
        return None

    def worker(self) -> Engine:
        """Share prepared arrays with a worker that owns its mutable solver state."""
        worker = copy(self)
        worker.stats = SolverStats()
        worker._cache, worker._detailed = {}, {}
        worker._passing, worker._failing = set(), set()
        worker._solvers = {}
        worker.on_check = None
        return worker

    def accept(self, mask: int, result: CheckResult, stats: SolverStats) -> None:
        """Merge a completed worker evaluation on the coordinating thread."""
        for field in fields(SolverStats):
            setattr(
                self.stats,
                field.name,
                getattr(self.stats, field.name) + getattr(stats, field.name),
            )
        certified = self.lookup(mask)
        if result.status is Status.UNRESOLVED and certified is not None:
            result = certified
        self._remember(mask, result)
        self._notify(mask, result)

    def _remember(self, mask: int, result: CheckResult) -> None:
        """Keep only minimal pass and maximal exclusion inference certificates."""
        self._cache[mask] = result
        if result.status is Status.COMPATIBLE:
            self._passing = {known for known in self._passing if known & mask != mask}
            self._passing.add(mask)
        elif result.status is Status.INCOMPATIBLE:
            self._failing = {known for known in self._failing if known | mask != mask}
            self._failing.add(mask)

    def _diagnostic(
        self,
        operators: ComplexArray,
        indices: tuple[int, ...],
        weights: RealArray,
    ) -> ProjectionFailure:
        """Recompute projection bounds and covariance for this candidate."""
        coefficients = np.zeros(len(self.data.values))
        coefficients[list(indices)] = weights
        matrix = np.einsum("i,ijk->jk", coefficients, operators)
        spectrum = np.linalg.eigvalsh(matrix)
        bound = Bound(float(spectrum[0]), float(spectrum[-1]))
        value = float(coefficients @ self.data.values)
        uncertainty = float(np.linalg.norm(coefficients @ self.data.factor))
        distance = max(bound.lower - value, value - bound.upper, 0.0)
        return ProjectionFailure(
            tuple(self.data.labels[index] for index in indices),
            tuple(float(x) for x in coefficients),
            bound,
            value,
            uncertainty,
            distance / uncertainty if uncertainty else float("inf"),
            tuple(float(weight) for weight in weights),
        )

    def evaluate(self, mask: int, *, diagnostics: bool = False) -> CheckResult:
        """Evaluate scalar and continuous pair conditions for a single candidate."""
        self.stats.checks += 1
        if not mask:
            return CheckResult(Status.INCOMPATIBLE)
        operators = self.operators(mask)
        eigenvalues, eigenvectors = np.linalg.eigh(operators)
        rounding = (
            128
            * np.finfo(float).eps
            * np.maximum(
                1.0,
                np.maximum(
                    np.abs(self.data.values), np.max(np.abs(eigenvalues), axis=1)
                ),
            )
        )
        allowances = self.n_sigma * self.errors + self.tolerance + rounding
        failed = np.flatnonzero(
            (self.data.values + allowances < eigenvalues[:, 0])
            | (self.data.values - allowances > eigenvalues[:, -1])
        )
        records: list[Diagnostic] = []
        incompatible = bool(len(failed))
        if incompatible:
            self.stats.scalar_exclusions += 1
            if not diagnostics:
                return CheckResult(Status.INCOMPATIBLE)
            for index in failed:
                records.append(self._diagnostic(operators, (int(index),), np.ones(1)))
        states = np.concatenate((eigenvectors[:, :, 0], eigenvectors[:, :, -1]), axis=0)
        references = np.unique(
            np.einsum("si,aij,sj->sa", states.conj(), operators, states).real,
            axis=0,
        )
        unresolved = False
        pairs = self.pairs
        hulls = self._hulls
        if diagnostics and self.max_combination_size == 2:
            pairs = [
                _geometry(
                    self.data.values[[i, j]],
                    self.data.covariance[np.ix_([i, j], [i, j])],
                    (i, j),
                )
                for i, j in combinations(self._active, 2)
            ]
            hulls = _HullBatch(pairs)
        passed = hulls.witnesses(references, self.n_sigma, self.tolerance)
        for index, pair in enumerate(pairs):
            if passed[index] or _point_hull_witness(
                references[:, list(pair.indices)], pair, self.n_sigma, self.tolerance
            ):
                self.stats.pair_checks += 1
                self.stats.hull_passes += 1
                continue
            result = _check_pair(
                operators[list(pair.indices)],
                pair,
                self.n_sigma,
                self.tolerance,
                self.stats,
                self._solvers,
            )
            if result.status is Status.INCOMPATIBLE:
                incompatible = True
                if not diagnostics:
                    return CheckResult(Status.INCOMPATIBLE)
                assert result.normal is not None
                records.append(
                    self._diagnostic(
                        operators,
                        pair.indices,
                        np.asarray(result.normal),
                    )
                )
            elif result.status is Status.UNRESOLVED:
                unresolved = True
                if diagnostics:
                    records.append(
                        UnresolvedPair(
                            (
                                self.data.labels[pair.indices[0]],
                                self.data.labels[pair.indices[1]],
                            )
                        )
                    )
        status = (
            Status.INCOMPATIBLE
            if incompatible
            else (Status.UNRESOLVED if unresolved else Status.COMPATIBLE)
        )
        return CheckResult(status, tuple(records))
