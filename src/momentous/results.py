"""Small result records shared by analysis and numerical geometry."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from momentous.domain import Component

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


class Status(StrEnum):
    """A verified necessary-condition pass, exclusion, or unresolved decision.

    Notes
    -----
    Compare explicitly with ``Status.COMPATIBLE`` or ``Status.INCOMPATIBLE``.
    Boolean coercion raises; use ``CheckResult.valid`` for Boolean validity.
    """

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNRESOLVED = "unresolved"

    def __str__(self) -> str:
        """Return the status value used by the public interface."""
        return self.value

    def __bool__(self) -> bool:
        """Reject truthiness; require explicit comparison or CheckResult.valid."""
        raise TypeError("Compare Status explicitly, or use CheckResult.valid")


class UnresolvedCompatibilityError(RuntimeError):
    """An optimizer returned neither a feasible nor a separating witness."""


class NormalizationWarning(UserWarning):
    """The normalizer is too uncertain for trustworthy first-order propagation."""


@dataclass(frozen=True)
class Bound:
    """An inclusive spectral interval in normalized moment coordinates.

    Parameters
    ----------
    lower, upper : float
        Minimum and maximum attainable values.
    """

    lower: float
    upper: float

    def __contains__(self, value: float) -> bool:
        """Test interval membership without adding numerical slack."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        """Format a compact inclusive interval."""
        return f"[{self.lower:.6g}, {self.upper:.6g}]"

    def __repr__(self) -> str:
        """Return an explicit interval representation."""
        return f"Bound(lower={self.lower!r}, upper={self.upper!r})"


@dataclass(frozen=True, repr=False)
class ProjectionFailure:
    """A verified candidate-specific scalar or pair exclusion.

    Parameters
    ----------
    components : tuple
        Participating scalar labels in projection order.
    coefficients : tuple of float
        Projection weights in the complete measurement's component order.
    bounds : Bound
        Candidate-specific attainable interval.
    value, uncertainty, sigma_distance : float
        Measured projection, standard deviation, and excursion in standard
        deviations. Exact exclusions have infinite sigma distance.
    normal : tuple of float
        Unit projection direction in participating component order.

    Notes
    -----
    All projection fields are present. Unresolved checks use UnresolvedPair.
    """

    components: tuple[Component, ...]
    coefficients: tuple[float, ...]
    bounds: Bound
    value: float
    uncertainty: float
    sigma_distance: float
    normal: tuple[float, ...]

    @property
    def status(self) -> Status:
        """Return the certified exclusion status."""
        return Status.INCOMPATIBLE

    def __str__(self) -> str:
        """Describe the excluded projection and its excursion."""
        return f"incompatible: {self.components} ({self.sigma_distance:.4g} sigma)"

    def __repr__(self) -> str:
        """Summarize a verified projection without dumping its coefficient vector."""
        return f"ProjectionFailure(components={self.components!r}, value={self.value!r}, bounds={self.bounds!r}, uncertainty={self.uncertainty!r})"


@dataclass(frozen=True)
class UnresolvedPair:
    """A continuous pair check for which no verdict was established.

    Parameters
    ----------
    components : tuple
        The two scalar observables involved.
    reason : str, optional
        Available explanation; it does not claim a separating witness exists.
    """

    components: tuple[Component, Component]
    reason: str = "Neither a feasible nor a separating witness was verified"

    @property
    def status(self) -> Status:
        """Return the unresolved status without treating it as a pass or exclusion."""
        return Status.UNRESOLVED

    def __str__(self) -> str:
        """Describe the unresolved observables and available reason."""
        return f"unresolved: {self.components}: {self.reason}"


type Diagnostic = ProjectionFailure | UnresolvedPair


@dataclass(frozen=True, repr=False)
class CheckResult:
    """Candidate compatibility and optional candidate-specific diagnostics.

    Parameters
    ----------
    status : Status
        A three-state compatibility decision.
    diagnostics : tuple of ProjectionFailure or UnresolvedPair, optional
        Collected exclusions and unresolved checks when diagnostics are enabled.

    Notes
    -----
    Passing these necessary conditions does not establish a simultaneous
    amplitude fit. ``valid`` raises instead of coercing an unresolved result.
    """

    status: Status
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def valid(self) -> bool:
        """Return Boolean validity without coercing unresolved decisions.

        Returns
        -------
        bool
            Whether the necessary compatibility conditions passed.

        Raises
        ------
        UnresolvedCompatibilityError
            If neither compatibility nor incompatibility has been verified.
        """
        if self.status is Status.UNRESOLVED:
            raise UnresolvedCompatibilityError("Compatibility remains unresolved")
        return self.status is Status.COMPATIBLE

    def __str__(self) -> str:
        """Summarize validity and diagnostic count."""
        return f"{self.status} ({len(self.diagnostics)} diagnostics)"

    def __bool__(self) -> bool:
        """Return validity, raising UnresolvedCompatibilityError if unresolved."""
        return self.valid

    def __repr__(self) -> str:
        """Summarize status and record count without listing every diagnostic."""
        return f"CheckResult(status={self.status.value!r}, diagnostics={len(self.diagnostics)})"


@dataclass(frozen=True)
class CheckProgress:
    """An immutable progress snapshot supplied to an analysis callback.

    Parameters
    ----------
    completed : int
        Search-domain wavesets processed so far, including cached decisions,
        compressed pruned branches, and unresolved leaves.
    total : int
        Number of wavesets in the selected search size domain.
    phase : {'check', 'search'}, default 'check'
        Search snapshots share one total per search. Numerical checks outside a
        search, including minimal-set certification, report one completed check
        with ``completed=total=1``.
    evaluated : bool, default True
        Whether this event reports a new numerical evaluation. Search also
        reports branch-completion events with ``evaluated=False`` so that
        progress includes candidates resolved without numerical work.

    Notes
    -----
    ``completed`` counts processed search candidates, not compatible candidates
    or solver calls. It reaches ``total`` even when results remain unresolved.
    An evaluation may support a whole branch or occur outside the search size
    limits, so evaluated events need not increment ``completed``. Use the
    absolute counts to update a progress bar rather than adding one per event.

    Examples
    --------
    >>> progress = CheckProgress(8, 15, phase='search', evaluated=False)
    >>> print(progress)
    search: 8/15 wavesets
    >>> progress.completed, progress.total
    (8, 15)
    """

    completed: int
    total: int
    phase: Literal["check", "search"] = "check"
    evaluated: bool = True

    def __str__(self) -> str:
        """Return the phase and processed fraction without inferring compatibility."""
        return f"{self.phase}: {self.completed}/{self.total} wavesets"


@dataclass(frozen=True)
class PairResult:
    """A numerical pair decision and optional verified separating normal."""

    status: Status
    normal: tuple[float, float] | None = None

    @property
    def valid(self) -> bool:
        """Return validity without silently accepting unresolved outcomes."""
        return CheckResult(self.status).valid

    def __str__(self) -> str:
        """Return the numerical decision."""
        return str(self.status)

    def __bool__(self) -> bool:
        """Return validity, raising UnresolvedCompatibilityError if unresolved."""
        return self.valid


@dataclass
class SolverStats:
    """Counters and solver wall time for a prepared analysis."""

    checks: int = 0
    cache_hits: int = 0
    inferred_checks: int = 0
    scalar_exclusions: int = 0
    pair_checks: int = 0
    wedge_passes: int = 0
    hull_passes: int = 0
    midpoint_exclusions: int = 0
    solver_calls: int = 0
    solver_seconds: float = 0.0

    def __str__(self) -> str:
        """Summarize evaluations and numerical solver work."""
        return f"{self.checks} checks, {self.solver_calls} conic solves"


@dataclass(frozen=True, repr=False)
class Region2D:
    """Numerical support and boundary data for a pair of scalar moments.

    Parameters
    ----------
    components : tuple
        Ordered x/y scalar labels.
    point : ndarray, shape (2,)
        Normalized measured point.
    covariance : ndarray, shape (2, 2)
        Corresponding marginal covariance.
    normals : ndarray, shape (n, 2)
        Unit outward support normals, sampled over a full circle.
    support : ndarray, shape (n,)
        Maximum attainable projection along each normal.
    boundary : ndarray, shape (k, 2)
        Counterclockwise vertices of an inner polygonal approximation, or
        one/two points for a degenerate point/segment. Not explicitly closed.
    n_sigma : float
        Radius multiplier for the measured covariance ellipse.

    Notes
    -----
    Support values come from eigenvalues; boundary vertices come from
    attainable extremizers. Joining vertices approximates curved boundaries
    from within and is never used to decide continuous compatibility.
    """

    components: tuple[Component, Component]
    point: RealArray
    covariance: RealArray
    normals: RealArray
    support: RealArray
    boundary: RealArray
    n_sigma: float

    def __repr__(self) -> str:
        """Summarize geometry without dumping numerical arrays."""
        return (
            f"Region2D(components={self.components!r}, "
            f"directions={len(self.normals)}, vertices={len(self.boundary)}, "
            f"n_sigma={self.n_sigma!r})"
        )

    def __str__(self) -> str:
        """Describe the two axes and numerical boundary size."""
        return f"{self.components[0]} × {self.components[1]}: {len(self.boundary)} vertices"
