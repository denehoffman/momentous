"""Compressed monotonic search and conservative minimal-set certification."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from heapq import merge
from itertools import combinations
from math import comb
from numbers import Integral
from threading import local
from typing import TYPE_CHECKING

from momentous._engine import Engine
from momentous.domain import Waveset
from momentous.results import CheckResult, SolverStats, Status

if TYPE_CHECKING:
    from momentous.analysis import AnalysisResult


@dataclass(frozen=True)
class Block:
    """Private bitmask interval containing only compatible candidates."""

    required: int
    allowed: int

    def count(self, minimum: int, maximum: int) -> int:
        """Count this branch's size-limited candidates without expanding subsets."""
        required, allowed = self.required.bit_count(), self.allowed.bit_count()
        return sum(
            comb(allowed - required, size - required)
            for size in range(max(minimum, required), min(maximum, allowed) + 1)
        )

    def restrict(self, include: int, exclude: int) -> Block | None:
        """Intersect a branch with required and forbidden wave masks."""
        required, allowed = self.required | include, self.allowed & ~exclude
        return None if required & ~allowed else Block(required, allowed)

    def members(
        self, minimum: int, maximum: int, *, smallest_only: bool = False
    ) -> Iterator[int]:
        """Yield branch members in cardinality and canonical-wave order."""
        required = self.required.bit_count()
        free = self.allowed & ~self.required
        bits = [1 << i for i in range(free.bit_length()) if free & (1 << i)]
        lower = max(minimum, required)
        upper = min(maximum, self.allowed.bit_count())
        if smallest_only:
            upper = min(upper, lower)
        for size in range(lower, upper + 1):
            for extra in combinations(bits, size - required):
                yield self.required | sum(extra)


@dataclass(frozen=True)
class Search:
    """Disjoint compatible blocks and unresolved leaves in a fixed size domain."""

    blocks: tuple[Block, ...]
    unresolved: frozenset[int]
    minimum: int
    maximum: int

    @property
    def count(self) -> int:
        """Count compatible subsets combinatorially without expanding blocks."""
        return sum(block.count(self.minimum, self.maximum) for block in self.blocks)

    def status(self, mask: int) -> Status:
        """Look up a candidate within the searched cardinality domain."""
        if not self.minimum <= mask.bit_count() <= self.maximum:
            raise ValueError(
                "Candidate is outside the searched size domain; use check()"
            )
        if any(
            block.required & mask == block.required
            and mask | block.allowed == block.allowed
            for block in self.blocks
        ):
            return Status.COMPATIBLE
        return Status.UNRESOLVED if mask in self.unresolved else Status.INCOMPATIBLE

    def passing(
        self, *, smallest_only: bool = False, include: int = 0, exclude: int = 0
    ) -> Iterator[int]:
        """Restrict compressed blocks before lazily expanding their members."""

        def order(mask: int) -> tuple[int, tuple[int, ...]]:
            """Order candidates independently of parallel branch completion."""
            return mask.bit_count(), tuple(
                i for i in range(mask.bit_length()) if mask & (1 << i)
            )

        restricted = (
            selected
            for block in self.blocks
            if (selected := block.restrict(include, exclude)) is not None
        )
        yield from merge(
            *(
                block.members(self.minimum, self.maximum, smallest_only=smallest_only)
                for block in restricted
            ),
            key=order,
        )

    def minimal(self, engine: Engine) -> tuple[int, ...]:
        """Certify minimal passes by excluding every admissible immediate subset."""
        seeds = sorted(
            set(self.passing(smallest_only=True)), key=lambda m: (m.bit_count(), m)
        )
        antichain: list[int] = []
        for mask in seeds:
            if not any(known & mask == known for known in antichain):
                antichain.append(mask)
        certified: list[int] = []
        for mask in antichain:
            if mask.bit_count() == self.minimum:
                certified.append(mask)
                continue
            smaller = [
                mask ^ (1 << i) for i in range(mask.bit_length()) if mask & (1 << i)
            ]
            if all(
                engine.check(subset).status is Status.INCOMPATIBLE for subset in smaller
            ):
                certified.append(mask)
        return tuple(certified)


@dataclass(frozen=True, repr=False)
class SearchResult:
    """Compressed subset-search results in an explicit cardinality domain.

    Notes
    -----
    Create with ``analysis.search(...)``. Counts and lookup do not expand passing
    supersets; enumeration remains potentially exponential. Minimality is relative
    to size_limits and unresolved subsets prevent certification. Passing is only
    a necessary moment-compatibility condition.

    Examples
    --------
    >>> from momentous import Covariance, MomentData, Waveset, analyze
    >>> data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
    >>> analysis = analyze(data, Waveset([(0, 0), (1, 0)]))
    >>> search = analysis.search(max_size=1)
    >>> search.count, search.candidate_count, search.size_limits
    (2, 2, (1, 1))
    """

    analysis: AnalysisResult
    _found: Search

    @property
    def pool(self) -> Waveset:
        """Return the searched basis, shared with the prepared analysis."""
        return self.analysis.pool

    @property
    def size_limits(self) -> tuple[int, int]:
        """Return inclusive searched minimum and maximum wave counts."""
        return self._found.minimum, self._found.maximum

    @property
    def complete(self) -> bool:
        """Report whether every searched candidate has a certified verdict."""
        return not self._found.unresolved

    @property
    def count(self) -> int:
        """Count compatible subsets without expanding compressed blocks."""
        return self._found.count

    @property
    def candidate_count(self) -> int:
        """Count all subsets in this search's selected size domain."""
        return sum(
            comb(len(self.pool), size)
            for size in range(self._found.minimum, self._found.maximum + 1)
        )

    @property
    def unresolved(self) -> tuple[Waveset, ...]:
        """Return unresolved candidates in deterministic order."""
        return tuple(
            self.analysis._engine.members(mask)
            for mask in sorted(self._found.unresolved)
        )

    @property
    def stats(self) -> SolverStats:
        """Return a detached snapshot of the shared analysis's cumulative work."""
        return self.analysis.stats

    def status(self, waves: Waveset) -> Status:
        """Look up a pool subset within this search's size limits.

        Parameters
        ----------
        waves : Waveset
            Candidate whose cardinality belongs to this search domain.

        Returns
        -------
        Status
            Compatible, incompatible, or unresolved.

        Raises
        ------
        ValueError
            If a wave is outside the pool or the candidate is outside size_limits.
            Use ``analysis.check(waves)`` for candidates outside search limits.
        """
        return self._found.status(self.analysis._engine.mask(waves))

    def wavesets(
        self,
        status: Status | str = Status.COMPATIBLE,
        *,
        include: Waveset | None = None,
        exclude: Waveset | None = None,
    ) -> Iterator[Waveset]:
        """Lazily enumerate searched candidates matching a status and wave filters.

        Parameters
        ----------
        status : Status or str, default 'compatible'
              Decision to enumerate; output can be exponential.
        include : Waveset, optional
            Require every wave in this set. Omission or an empty set imposes
            no inclusion constraint.
        exclude : Waveset, optional
            Omit every candidate containing any wave in this set. Omission or
            an empty set imposes no exclusion constraint.

        Yields
        ------
        Waveset
              Immutable matching candidates in deterministic order.

        Raises
        ------
        ValueError
            If a filter contains a wave outside the pool, has a different
            polarization mode, or includes and excludes the same wave.
        TypeError
            If a filter is not a Waveset.

        Notes
        -----
        Filters retain the original search's size limits and statuses. They
        do not rerun checks or change ``count``, ``complete``, or progress.
        Compatible blocks are restricted before expansion, avoiding enumeration
        of nonmatching supersets. Impossible cardinality constraints yield no
        candidates. Validation occurs when the iterator is advanced.

        Examples
        --------
        >>> from momentous import Covariance, MomentData, Waveset, analyze
        >>> pool = Waveset([(0, 0), (1, 0), (1, 1)])
        >>> data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
        >>> result = analyze(data, pool).search()
        >>> required = Waveset([(0, 0)])
        >>> forbidden = Waveset([(1, 1)])
        >>> list(result.wavesets(include=required, exclude=forbidden))
        [Waveset([(0, 0)]), Waveset([(0, 0), (1, 0)])]
        """
        requested = Status(status)
        included, excluded = self._filter_masks(include, exclude)
        if requested is Status.COMPATIBLE:
            for mask in self._found.passing(include=included, exclude=excluded):
                yield self.analysis._engine.members(mask)
        elif requested is Status.UNRESOLVED:
            for mask in sorted(self._found.unresolved):
                if mask & included == included and not mask & excluded:
                    yield self.analysis._engine.members(mask)
        else:
            domain = Block(0, (1 << len(self.pool)) - 1).restrict(included, excluded)
            if domain is not None:
                for mask in domain.members(*self.size_limits):
                    if self._found.status(mask) is requested:
                        yield self.analysis._engine.members(mask)

    def minimal_wavesets(
        self, *, include: Waveset | None = None, exclude: Waveset | None = None
    ) -> tuple[Waveset, ...]:
        """Return certified minimal passes matching optional wave filters.

        Parameters
        ----------
        include : Waveset, optional
            Require every listed wave in a returned minimum.
        exclude : Waveset, optional
            Require every listed wave to be absent from a returned minimum.

        Returns
        -------
        tuple of Waveset
              Certified minimal sets relative to size_limits. Unresolved smaller
              subsets prevent certification; incomplete searches may omit minima.

        Raises
        ------
        ValueError
            If a filter contains a wave outside the pool, has a different
            polarization mode, or includes and excludes the same wave.
        TypeError
            If a filter is not a Waveset.

        Notes
        -----
        These filters select among the original search's certified minima.
        They do not redefine minimality under forced wave inclusion. A matching
        compatible candidate can exist even when no original minimum matches.
        """
        included, excluded = self._filter_masks(include, exclude)
        return tuple(
            self.analysis._engine.members(mask)
            for mask in self._found.minimal(self.analysis._engine)
            if mask & included == included and not mask & excluded
        )

    def _filter_masks(
        self, include: Waveset | None, exclude: Waveset | None
    ) -> tuple[int, int]:
        """Validate pool membership and reject contradictory wave constraints."""
        engine = self.analysis._engine
        included = 0 if include is None else engine.mask(include)
        excluded = 0 if exclude is None else engine.mask(exclude)
        if included & excluded:
            raise ValueError("A wave cannot be both included and excluded")
        return included, excluded

    def __repr__(self) -> str:
        """Summarize searched sizes and verdict counts without enumerating subsets."""
        return f"SearchResult(sizes={self.size_limits}, compatible={self.count}/{self.candidate_count}, unresolved={len(self._found.unresolved)})"

    def __str__(self) -> str:
        """Return the compact search summary."""
        return repr(self)


class _Traversal:
    """Coordinate disjoint search branches and monotonic certificates."""

    def __init__(
        self,
        engine: Engine,
        minimum: int,
        maximum: int,
    ) -> None:
        """Initialize an exclusion-first traversal over the selected size domain."""
        self.engine, self.minimum, self.maximum = engine, minimum, maximum
        self.stack = [Block(0, (1 << len(engine.pool)) - 1)]
        self.total = self.stack[0].count(minimum, maximum)
        self.completed = 0
        self.blocks: list[Block] = []
        self.unresolved: set[int] = set()

    def advance(
        self, branch: Block, check: Callable[[int], CheckResult | None]
    ) -> int | None:
        """Resolve or split a branch, returning an uncached mask when work is needed."""
        required, allowed = branch.required, branch.allowed
        if required.bit_count() > self.maximum or allowed.bit_count() < self.minimum:
            return None
        if required:
            result = check(required)
            if result is None:
                return required
            if result.status is Status.COMPATIBLE:
                self.blocks.append(branch)
                self._complete(branch, required, result)
                return None
        result = check(allowed)
        if result is None:
            return allowed
        if result.status is Status.INCOMPATIBLE:
            self._complete(branch, allowed, result)
            return None
        elif required == allowed:
            self.unresolved.add(required)
            self._complete(branch, allowed, result)
        else:
            undecided = allowed & ~required
            bit = undecided & -undecided
            self.stack.append(Block(required | bit, allowed))
            self.stack.append(Block(required, allowed ^ bit))
        return None

    def _complete(self, branch: Block, mask: int, result: CheckResult) -> None:
        """Advance progress by a terminal branch's domain cardinality."""
        if self.engine.on_check is not None:
            self.completed += branch.count(self.minimum, self.maximum)
            self.engine.report_progress(mask, result, self.completed)

    def finish(self) -> Search:
        """Apply later certificates to unresolved leaves and freeze the search."""
        remaining = set()
        for mask in sorted(self.unresolved):
            status = self.engine.check(mask).status
            if status is Status.COMPATIBLE:
                self.blocks.append(Block(mask, mask))
            elif status is Status.UNRESOLVED:
                remaining.add(mask)
        return Search(
            tuple(self.blocks), frozenset(remaining), self.minimum, self.maximum
        )


@dataclass(frozen=True)
class _Evaluation:
    """A completed worker evaluation and its private work counters."""

    mask: int
    result: CheckResult
    stats: SolverStats


class _Workers:
    """Keep solver templates private to each search thread."""

    def __init__(self, engine: Engine) -> None:
        """Share immutable preparation while allocating thread-local evaluators."""
        self.engine = engine
        self.local = local()

    def evaluate(self, mask: int) -> _Evaluation:
        """Evaluate without reading or writing coordinating search certificates."""
        if not hasattr(self.local, "engine"):
            self.local.engine = self.engine.worker()
        engine = self.local.engine
        engine.stats = SolverStats()
        result = engine.evaluate(mask)
        return _Evaluation(mask, result, engine.stats)


def _parallel(traversal: _Traversal, workers: int) -> None:
    """Schedule bounded independent evaluations and merge them on the caller."""
    evaluators = _Workers(traversal.engine)
    pending: dict[Future[_Evaluation], list[Block]] = {}
    masks: dict[int, Future[_Evaluation]] = {}
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="momentous")
    try:
        while traversal.stack or pending:
            while traversal.stack and len(pending) < workers:
                branch = traversal.stack.pop()
                mask = traversal.advance(branch, traversal.engine.lookup)
                if mask is None:
                    continue
                future = masks.get(mask)
                if future is None:
                    future = executor.submit(evaluators.evaluate, mask)
                    masks[mask] = future
                    pending[future] = []
                pending[future].append(branch)
            if pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Merge simultaneous completions in a stable mask order.
                for future in sorted(finished, key=lambda f: f.result().mask):
                    evaluation = future.result()
                    traversal.engine.accept(
                        evaluation.mask, evaluation.result, evaluation.stats
                    )
                    del masks[evaluation.mask]
                    traversal.stack.extend(pending.pop(future))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def search(
    engine: Engine,
    minimum: int,
    maximum: int,
    workers: int | None = None,
) -> Search:
    """Prune monotonic branches with serial or private-thread candidate checks."""
    if workers is None:
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        workers = min(2 if gil_enabled else 8, os.cpu_count() or 1)
    if isinstance(workers, bool) or not isinstance(workers, Integral):
        raise TypeError("workers must be a positive integer or None")
    if workers < 1:
        raise ValueError("workers must be positive")
    traversal = _Traversal(engine, minimum, maximum)
    with engine.track_search(traversal.total):
        if workers == 1:
            while traversal.stack:
                traversal.advance(traversal.stack.pop(), engine.check)
        else:
            _parallel(traversal, int(workers))
        return traversal.finish()
