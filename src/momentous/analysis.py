"""Prepared analyses, candidate views, and explicit subset searches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from momentous._engine import Engine
from momentous.covariance import readonly
from momentous.data import MomentData, prepare_data
from momentous.domain import (
    Component,
    ComponentInput,
    MomentKey,
    Observable,
    Waveset,
    as_component,
    size_limits,
)
from momentous.geometry import _settings, region_data
from momentous.results import (
    Bound,
    CheckProgress,
    CheckResult,
    ComplexArray,
    RealArray,
    Region2D,
    SolverStats,
    Status,
)
from momentous.search import SearchResult
from momentous.search import search as run_search


@dataclass(frozen=True, repr=False, eq=False)
class CandidateAnalysis:
    """A fixed waveset bound to a prepared analysis and its shared caches.

    Notes
    -----
    Construct with ``analysis.for_waves(waves)``. Membership is validated once.
    Geometry requires a nonempty candidate; the empty set checks incompatible.

    Examples
    --------
    >>> from momentous import Covariance
    >>> data = MomentData({(0, 0): 1, (1, 1): 0j}, covariance=Covariance.exact())
    >>> pool = Waveset([(0, 0), (1, 1)])
    >>> candidate = analyze(data, pool).for_waves(pool)
    >>> candidate.status().value
    'compatible'
    """

    _analysis: AnalysisResult
    waves: Waveset
    _mask: int

    def status(self) -> Status:
        """Return a cached or verified necessary-condition verdict."""
        return self.check().status

    def check(self, *, diagnostics: bool = False) -> CheckResult:
        """Check this candidate, optionally collecting full scientific diagnostics.

        Parameters
        ----------
        diagnostics : bool, default False
            Evaluate candidate-specific projections even if status can be inferred.

        Returns
        -------
        CheckResult
            Verdict with ProjectionFailure and UnresolvedPair records when requested.
        """
        if not isinstance(diagnostics, bool):
            raise ValueError("diagnostics must be a boolean")
        return self._analysis._engine.check(self._mask, diagnostics=diagnostics)

    def operators(self) -> ComplexArray:
        """Return immutable normalized operators in this candidate's wave order.

        Returns
        -------
        ndarray
            Hermitian operators aligned with analysis.component_labels. Polarized
            operators use a whitened normalization metric.
        """
        return readonly(self._analysis._engine.operators(self._mask))

    def bounds(self) -> Mapping[Component, Bound]:
        """Return scalar spectral bounds keyed by canonical observable labels."""
        spectra = np.linalg.eigvalsh(self.operators())
        return MappingProxyType(
            {
                label: Bound(float(spectra[index, 0]), float(spectra[index, -1]))
                for index, label in enumerate(self._analysis.component_labels)
            }
        )

    def region(
        self, first: ComponentInput, second: ComponentInput, *, n_directions: int = 360
    ) -> Region2D:
        """Return numerical geometry for two scalar observables of this candidate.

        Parameters
        ----------
        first, second : Observable, polarized Moment, or tuple
            Region axes; select complex moments with Moment.real or Moment.imag.
            Canonical tuple shorthand is also accepted.
        n_directions : int, default 360
            Full-circle display resolution, at least four; it never affects verdicts.

        Returns
        -------
        Region2D
            Support data, approximate boundary, measured point, and covariance.
        """
        labels = (as_component(first), as_component(second))
        if labels[0] == labels[1]:
            raise ValueError("Region axes must be distinct scalar observables")
        analysis = self._analysis
        try:
            indices = [analysis.component_labels.index(label) for label in labels]
        except ValueError as error:
            raise ValueError(
                "Region axes must be components supplied to this analysis"
            ) from error
        return region_data(
            self.operators()[indices],
            labels,
            analysis.values[indices],
            analysis.covariance[np.ix_(indices, indices)],
            analysis.n_sigma,
            n_directions,
        )

    def __repr__(self) -> str:
        """Describe the bound waveset without running checks or printing matrices."""
        return f"CandidateAnalysis(waves={self.waves!r})"

    def __str__(self) -> str:
        """Format the candidate's scientific wave labels."""
        return f"Candidate {self.waves}"


@dataclass(frozen=True, repr=False, eq=False)
class AnalysisResult:
    """Prepared normalized measurements, checks, and geometry for a wave pool.

    Notes
    -----
    Create with analyze. Input snapshots are immutable; search is a separate
    operation returning SearchResult. Candidate views and searches share numerical
    caches. Passing does not establish a simultaneous amplitude fit.
    ``data`` retains the raw measurement; values, moments, and covariance here
    describe its normalized counterpart.
    """

    _engine: Engine
    data: MomentData

    @property
    def pool(self) -> Waveset:
        """Return the complete immutable wave pool."""
        return self._engine.pool

    @property
    def moments(self) -> Mapping[MomentKey, complex]:
        """Return normalized moments with canonical tuple keys."""
        return MappingProxyType(
            {
                moment.key: value
                for moment, value in self._engine.data.normalized_values.items()
            }
        )

    @property
    def values(self) -> RealArray:
        """Return immutable normalized scalar values in component-label order."""
        return self._engine.data.values.view()

    @property
    def component_labels(self) -> tuple[Component, ...]:
        """Return canonical labels in normalized values and covariance order."""
        return self._engine.data.labels

    @property
    def observables(self) -> tuple[Observable, ...]:
        """Return scalar objects in normalized values and covariance order."""
        return self.data.observables

    @property
    def covariance(self) -> RealArray:
        """Return immutable normalized real covariance, including correlations."""
        return self._engine.data.covariance.view()

    @property
    def n_sigma(self) -> float:
        """Return the fixed covariance radius multiplier."""
        return self._engine.n_sigma

    @property
    def max_combination_size(self) -> int:
        """Return the largest checked scalar combination size."""
        return self._engine.max_combination_size

    @property
    def stats(self) -> SolverStats:
        """Return detached cumulative counters for all shared analysis work."""
        return replace(self._engine.stats)

    def for_waves(self, waves: Waveset) -> CandidateAnalysis:
        """Bind a pool subset to its reusable checks and geometry.

        Parameters
        ----------
        waves : Waveset
            Immutable candidate drawn from this pool.

        Returns
        -------
        CandidateAnalysis
            A lightweight view sharing this analysis and its numerical caches.

        Raises
        ------
        ValueError
            If a wave or polarization is outside the pool.
        """
        return CandidateAnalysis(self, waves, self._engine.mask(waves))

    def status(self, waves: Waveset) -> Status:
        """Check any pool subset regardless of any separately searched size domain."""
        return self.for_waves(waves).status()

    def check(
        self, waves: Waveset | None = None, *, diagnostics: bool = False
    ) -> CheckResult:
        """Check a candidate or the full pool, optionally collecting diagnostics.

        Parameters
        ----------
        waves : Waveset, optional
            Candidate, defaulting to the complete pool.
        diagnostics : bool, default False
            Collect ProjectionFailure and UnresolvedPair records for this candidate.

        Returns
        -------
        CheckResult
            Compatible, incompatible, or unresolved necessary-condition decision.
        """
        return self.for_waves(self.pool if waves is None else waves).check(
            diagnostics=diagnostics
        )

    def operators(self, waves: Waveset | None = None) -> ComplexArray:
        """Return normalized operators for a nonempty candidate or the full pool."""
        return self.for_waves(self.pool if waves is None else waves).operators()

    def bounds(self, waves: Waveset | None = None) -> Mapping[Component, Bound]:
        """Return scalar spectral bounds for a candidate or the full pool."""
        return self.for_waves(self.pool if waves is None else waves).bounds()

    def region(
        self,
        first: ComponentInput,
        second: ComponentInput,
        *,
        waves: Waveset | None = None,
        n_directions: int = 360,
    ) -> Region2D:
        """Return pair geometry for a candidate or the full pool.

        Parameters
        ----------
        first, second : Observable, polarized Moment, or tuple
            Ordered scalar axes. Tuple shorthand is accepted.
        waves : Waveset, optional
            Nonempty candidate, defaulting to the complete pool.
        n_directions : int, default 360
            Boundary display resolution, independent of compatibility checking.

        Returns
        -------
        Region2D
            Numerical support, boundary, measured point, and covariance.
        """
        return self.for_waves(self.pool if waves is None else waves).region(
            first, second, n_directions=n_directions
        )

    def search(
        self,
        *,
        min_size: int = 1,
        max_size: int | None = None,
        workers: int | None = None,
    ) -> SearchResult:
        """Search pool subsets in an explicit inclusive cardinality domain.

        Parameters
        ----------
        min_size : int, default 1
            Smallest searched nonempty waveset.
        max_size : int, optional
            Largest searched waveset, defaulting to pool size.
        workers : int, optional
            Number of search threads. One runs serially. The default uses up to two on
            ordinary Python and up to eight when the GIL is disabled. Each worker
            owns its numerical solver templates; ``on_check`` callbacks run on the
            calling thread. Native numerical libraries retain their own settings.

        Returns
        -------
        SearchResult
            Compressed passing sets, unresolved candidates, and certified minima.

        Raises
        ------
        ValueError
            If size limits are inadmissible or workers is not positive.
        TypeError
            If workers is not an integer.

        Notes
        -----
        Worst-case search and output are exponential. Repeated searches reuse
        established candidate certificates while returning separate size domains.
        Callback progress counts every candidate in these size limits, including
        cached or pruned candidates. Reaching the total means the traversal
        finished; inspect ``complete`` separately for unresolved compatibility.
        Default worker selection checks the runtime GIL state. A native extension
        such as CVXPY's current ``_cvxcore`` can re-enable the GIL when imported;
        numerical array operations may still run concurrently.
        """
        minimum, maximum = size_limits(min_size, max_size, len(self.pool))
        if minimum < 1:
            raise ValueError("Search min_size must be at least one")
        return SearchResult(self, run_search(self._engine, minimum, maximum, workers))

    def __repr__(self) -> str:
        """Summarize input dimensions without running search or expanding candidates."""
        return f"AnalysisResult(waves={len(self.pool)}, components={len(self.values)})"

    def __str__(self) -> str:
        """Return a compact prepared-analysis summary."""
        return repr(self)


def analyze(
    data: MomentData,
    waves: Waveset,
    *,
    max_combination_size: int = 2,
    n_sigma: float = 3.0,
    tolerance: float = 1e-8,
    on_check: Callable[[Waveset, CheckResult, CheckProgress], None] | None = None,
) -> AnalysisResult:
    r"""Normalize raw measurements and prepare necessary waveset compatibility checks.

    Parameters
    ----------
    data : MomentData
        Validated raw measurements with their covariance and positive real H00.
        Includes at least one higher or other polarized moment. Selected moments
        are allowed; complete multiplets are not required. Use the extraction
        definitions below; pre-normalized input is unsupported.
    waves : Waveset
        Nonempty wave pool, often built with Waveset.from_max_l.
    max_combination_size : {1, 2}, default 2
        One checks individuals; two additionally checks every continuous pair.
    n_sigma : float, default 3
        Covariance radius multiplier, not a joint confidence level or p-value.
    tolerance : float, default 1e-8
        Absolute numerical slack in normalized coordinates, separate from errors.
    on_check : callable, optional
        Called as ``on_check(waves, result, progress)`` after each numerical
        evaluation and when a search resolves a pruned or cached branch.
        Arguments are immutable ``Waveset``, ``CheckResult``, and
        ``CheckProgress`` objects. During searches, ``progress.completed`` and
        ``progress.total`` count wavesets in the selected size domain, including
        unresolved leaves. ``progress.evaluated`` distinguishes numerical checks
        from branch-completion events. Outside searches, numerical checks report
        ``phase='check'`` and ``completed=total=1``; cached and inferred direct
        checks remain silent. Callbacks run synchronously on the calling thread,
        including parallel searches. Exceptions propagate after retaining any
        completed numerical check.

    Returns
    -------
    AnalysisResult
        Normalized measurements, candidate checks, geometry, and explicit search.

    Raises
    ------
    ValueError
        If pool, polarization, or numerical settings are invalid.
    TypeError
        If data is not MomentData or on_check is not callable.
    NotImplementedError
        If a combination size larger than two is requested.

    Warns
    -----
    NormalizationWarning
        If the normalizer's relative deviation is at least one third. First-order
        ratio propagation may be poor; absence is not an accuracy guarantee.

    Notes
    -----
    Both modes use laddu's Wigner-D convention:

    .. math:: D^L_{M0}(\phi,\theta,0)=d^L_{M0}(\theta)e^{-iM\phi}.

    Unpolarized raw moments are ``H_LM = integral(I * D_LM)``; H00 is total
    intensity. All moments divide by H00 internally, with no rank-dependent
    normalization factor. For full acceptance, an empirical raw estimate is
    ``sum(w * D_LM)``. The old unit-normalized harmonic convention must be
    converted with ``sqrt(4*pi/(2*L+1))`` for each raw value and covariance row.
    This conversion leaves previously normalized predictions unchanged.

    For uniform beam orientations and full acceptance, the polarized raw
    estimators in the Mathieu convention are:

    * H^0(L,M): ``sum(w * D.real)``;
    * H^1(L,M): ``sum(w * 2*cos(2*Phi)/P * D.real)``;
    * Im H^2(L,M): ``sum(w * 2*sin(2*Phi)/P * D.imag)``.

    Here H^0(0,0) is total intensity and every variant divides by it. These
    simple formulas require P > 0 and uniform orientation coverage. Prefer
    ``Acceptance(...).extract(data).data``: it handles angular mixing,
    grouped weighted hypotheses, correlated covariance, and varying polarization
    without dividing events by P. Supply data, generated MC, and accepted MC,
    with truth angles for accepted MC and an explicit MC statistics policy.
    See Mathieu et al., Eq. (13), https://arxiv.org/abs/1906.04841, and
    ``docs/extraction.md`` for the response and first-order propagation.

    For independent physical events under a compound-Poisson model, raw
    covariance is ``sum(outer(g_e, g_e))``, where each event's feature contribution
    ``g_e`` sums its weighted hypotheses. It conditions on the supplied weights.
    Flatten unpolarized features as interleaved Re/Im columns. Never treat
    correlated hypotheses as independent or renormalize their weights per event.

    Raw covariance is propagated using the first-order analytic Jacobian,
    ``J @ V @ J.T``, including all normalizer correlations. The imaginary raw
    H00 component must be exact zero. The normalized denominator is exactly one
    with zero variance. Input snapshots prevent caller mutations invalidating
    cached decisions. A different harmonic convention, conjugation, or axes
    cannot be detected from numeric inputs alone. H00=1 is allowed but does
    not bypass normalization. A real angular intensity gives
    ``H(L,-M) = (-1)**M * H(L,M).conjugate()`` and Im H(L,0)=0. If both signed
    projections are supplied, their correlations must be retained.
    Passing scalar/pair conditions establishes neither
    a simultaneous amplitude solution nor a joint confidence level.

    Examples
    --------
    Observe actual evaluations without choosing a progress-display library:

    >>> from momentous import Covariance, Moment, NormalizationWarning
    >>> checks = []
    >>> data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
    >>> analysis = analyze(data, Waveset([(0, 0), (1, 0)]),
    ...     on_check=lambda waves, result, progress: checks.append((waves, result.status)))
    >>> checks[0][0] == analysis.pool, checks[0][1].value
    (True, 'compatible')

    Construct raw data, bind a candidate, and search explicitly:

    >>> data = MomentData({(0, 0): 10, (1, 1): 0j}, covariance=Covariance.exact())
    >>> analysis = analyze(data, Waveset([(0, 0), (1, 1)]))
    >>> candidate = analysis.for_waves(Waveset([(0, 0), (1, 1)]))
    >>> h11 = Moment(1, 1)
    >>> candidate.region(h11.real, h11.imag, n_directions=8).boundary.shape
    (8, 2)
    >>> search = analysis.search()
    >>> search.count, len(search.minimal_wavesets()), search.size_limits
    (3, 2, (1, 2))

    Obtain full-acceptance raw moments with laddu and a labeled covariance:

    >>> import laddu as ld
    >>> costheta = np.array([0.8, 0.4, -0.2, -0.6])
    >>> phi = np.array([0.0, 1.0, 2.0, 3.0])
    >>> dataset = ld.Dataset.from_arrays(p4s={}, scalars={"costheta": costheta, "phi": phi})
    >>> expression = ld.WignerD(1, 1, 0).D(
    ...     alpha=ld.scalar("phi"), beta=ld.scalar("costheta").acos())
    >>> d11 = dataset.evaluate(expression, execution=ld.Execution("jit", precision="f64"))
    >>> h00, h11 = Moment(0, 0), Moment(1, 1)
    >>> features = np.column_stack([np.ones(4), np.zeros(4), d11.real, d11.imag])
    >>> covariance = Covariance(features.T @ features,
    ...     components=[h00.real, h00.imag, h11.real, h11.imag])
    >>> data = MomentData({h00: 4.0, h11: complex(d11.sum())}, covariance=covariance)
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", NormalizationWarning)
    ...     analysis = analyze(data, Waveset.from_max_l(1))
    >>> analysis.moments[(0, 0)] == 1
    True

    Independent uncertainty and polarized scalar selectors:

    >>> h0, h1 = Moment(0, 0, variant=0), Moment(0, 0, variant=1)
    >>> data = MomentData({h0: 10.0, h1: 10.0},
    ...     covariance=Covariance.from_uncertainties({h0: 0.1, h1: 0.2}))
    >>> pool = Waveset.from_max_l(0, reflectivities=('+', '-'))
    >>> analysis = analyze(data, pool)
    >>> analysis.status(Waveset([(0, 0, '+')])).value
    'compatible'
    """
    _settings(n_sigma, tolerance)
    if on_check is not None and not callable(on_check):
        raise TypeError("on_check must be callable or None")
    if not isinstance(max_combination_size, Integral) or isinstance(
        max_combination_size, bool
    ):
        raise ValueError("max_combination_size must be an integer")
    if max_combination_size > 2:
        raise NotImplementedError("Combinations larger than two are not implemented")
    if max_combination_size < 1:
        raise ValueError("max_combination_size must be one or two")
    prepared = prepare_data(data, waves)
    engine = Engine(
        waves, prepared, n_sigma, tolerance, int(max_combination_size), on_check
    )
    engine.check((1 << len(waves)) - 1)
    return AnalysisResult(engine, data)
