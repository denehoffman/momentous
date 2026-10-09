"""Acceptance response estimation, physical-event statistics, and moment solving.

The derivation and covariance assumptions are documented in
``docs/extraction.md``. Every covariance is first-order and conditional on
supplied weights, generation density, analysis frame, and beam calibration.
"""

from dataclasses import dataclass
from typing import Literal, Self

import laddu as ld
import numpy as np
from numpy.typing import NDArray

from momentous._sources import execution_or_default
from momentous.basis import MomentBasis
from momentous.covariance import Covariance, covariance_factor, readonly
from momentous.data import MomentData
from momentous.domain import Component, Moment
from momentous.results import RealArray
from momentous.samples import EventSample


@dataclass(frozen=True, init=False, repr=False, eq=False)
class MCIntegration:
    """Declare how MC generation samples the reference angular measure.

    Notes
    -----
    The reference measure is uniform in decay solid angle and follows the data's
    beam exposure. Generated MC must reproduce that exposure, including beam
    polarization magnitude and orientation, or be reweighted appropriately.
    Generation and selection weights must have compatible exposure units.
    """

    relative_density: ld.Expr | None

    def __init__(self) -> None:
        """Require a constructor declaring the generation measure."""
        raise TypeError("Use MCIntegration.uniform or importance")

    @classmethod
    def uniform(cls) -> Self:
        """Declare uniform decay-angle generation with matching beam exposure.

        Returns
        -------
        MCIntegration
            Unit inverse-density weights. This is an explicit scientific
            assumption, not a distribution inferred from the sample.
        """
        instance = cls.__new__(cls)
        object.__setattr__(instance, "relative_density", None)
        return instance

    @classmethod
    def importance(cls, *, relative_density: ld.Expr) -> Self:
        """Correct nonuniform MC generation with a known positive density.

        Parameters
        ----------
        relative_density : laddu.Expr
            Generation density q relative to the reference measure. It may use
            sample scalars ``costheta``, ``phi`` and polarized ``P``, ``Phi``. It is
            evaluated at truth coordinates for both MC samples. An overall
            constant cancels. Do not also include 1/q in the sample weights.

        Returns
        -------
        MCIntegration
            A reusable importance-integration declaration.

        Examples
        --------
        >>> integration = MCIntegration.importance(
        ...     relative_density=1 + 0.2 * ld.scalar("costheta"))
        >>> str(integration)
        'Known generation density relative to the angular reference measure'
        """
        if not isinstance(relative_density, ld.Expr):
            raise TypeError("relative_density must be a laddu expression")
        instance = cls.__new__(cls)
        object.__setattr__(instance, "relative_density", relative_density)
        return instance

    def _weights(self, sample: EventSample, execution: ld.Execution) -> RealArray:
        """Compute nonnegative exposure weights with the declared density."""
        if np.any(sample.weights < 0):
            raise ValueError(
                "MC weights must be nonnegative exposure/selection weights"
            )
        density = np.ones(len(sample))
        if self.relative_density is not None and len(sample):
            raw = np.asarray(
                sample._dataset().evaluate(self.relative_density, execution=execution)
            )
            if np.any(raw.imag != 0):
                raise ValueError("Generation density must be real")
            density = np.asarray(raw.real, dtype=np.float64)
        if not np.all(np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("Generation density must be finite and strictly positive")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                return sample.weights / density
            except FloatingPointError as error:
                raise ValueError(
                    "MC importance weights exceed numerical range"
                ) from error

    def __repr__(self) -> str:
        """Describe the constructor without printing a density expression graph."""
        return (
            "MCIntegration.uniform()"
            if self.relative_density is None
            else "MCIntegration.importance(relative_density=expression)"
        )

    def __str__(self) -> str:
        """Describe the asserted generation measure."""
        return (
            "Uniform decay angles with matching beam exposure"
            if self.relative_density is None
            else "Known generation density relative to the angular reference measure"
        )


@dataclass(frozen=True, init=False, repr=False)
class MCStatistics:
    """An explicit policy for finite-MC uncertainty and sample correlation.

    Notes
    -----
    Missing matching IDs do not establish independent MC. Use ``data_only`` to
    condition on the response, ``independent`` for independent Poisson MC
    samples with comparable exposure, or ``linked`` for accepted hypotheses
    originating from a fixed-size generated physical-event sample. Data and MC
    are assumed independent in every policy. All weights are conditioned upon.
    """

    mode: Literal["data_only", "independent", "linked"]

    def __init__(self) -> None:
        """Require an explicit declaration of the MC covariance policy."""
        raise TypeError("Use MCStatistics.data_only, independent, or linked")

    @classmethod
    def data_only(cls) -> Self:
        """Condition on the estimated acceptance response.

        Returns
        -------
        MCStatistics
            Propagate physical-event data errors only. Finite MC is excluded,
            not asserted to be exact; the result reports this scope explicitly.
        """
        return cls._build("data_only")

    @classmethod
    def independent(cls) -> Self:
        """Declare independent generated and accepted Poisson MC samples.

        Returns
        -------
        MCStatistics
            Include both MC contributions without generated/accepted covariance.

        Notes
        -----
        Samples must be independent Poisson realizations with comparable known
        exposure. This policy is not valid for a selected subset of the same
        generated run, nor for MC counts fixed after separately normalizing each
        sample. Use linked IDs or condition on the response in those cases.
        """
        return cls._build("independent")

    @classmethod
    def linked(cls) -> Self:
        """Link accepted event IDs to a fixed-size generated MC sample.

        Returns
        -------
        MCStatistics
            Include generated/accepted covariance and within-event hypothesis
            correlations. At least two generated physical events are required.

        Notes
        -----
        Both samples must use ``EventGrouping.from_ids`` with a common ID
        namespace, such as (run, event). Each accepted event must be present in
        generated MC. Rejected generated events contribute zero accepted weight.
        """
        return cls._build("linked")

    @classmethod
    def _build(cls, mode: Literal["data_only", "independent", "linked"]) -> Self:
        """Freeze the declared statistical policy."""
        instance = cls.__new__(cls)
        object.__setattr__(instance, "mode", mode)
        return instance

    def __repr__(self) -> str:
        """Return a named-constructor representation."""
        return f"MCStatistics.{self.mode}()"

    def __str__(self) -> str:
        """Describe the scope and MC correlation assumptions."""
        return {
            "data_only": "Data uncertainty conditional on the MC response",
            "independent": "Data and independent Poisson MC uncertainty",
            "linked": "Data and linked fixed-size MC uncertainty",
        }[self.mode]

    def __reduce__(self) -> tuple:
        """Restore a policy using its named constructor."""
        return getattr(MCStatistics, self.mode), ()


@dataclass(frozen=True, repr=False, eq=False)
class ResponseDiagnostics:
    """Numerical identifiability and MC exposure summary.

    Parameters
    ----------
    singular_values : array_like
        Response singular values in decreasing order.
    generated_exposure : float
        Sum of generated importance/exposure weights; accepted sums never replace it.
    generated_events, accepted_events : int
        Physical-event counts, after grouping hypotheses.

    Notes
    -----
    The relative singular-value threshold is 1e-10. A response below that
    threshold is rejected, without regularization or dropped coordinates.
    """

    singular_values: RealArray
    generated_exposure: float
    generated_events: int
    accepted_events: int

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the numerical spectrum."""
        object.__setattr__(
            self,
            "singular_values",
            readonly(np.asarray(self.singular_values, dtype=np.float64)),
        )

    @property
    def rank(self) -> int:
        """Return the number of singular values above the relative threshold."""
        values = self.singular_values
        return int(np.count_nonzero(values > 1e-10 * values[0]))

    @property
    def condition_number(self) -> float:
        """Return the largest/smallest singular-value ratio, or infinity."""
        values = self.singular_values
        return float(values[0] / values[-1]) if values[-1] > 0 else float("inf")

    def __repr__(self) -> str:
        """Summarize rank, conditioning, and event counts without dumping arrays."""
        return f"ResponseDiagnostics(rank={self.rank}/{len(self.singular_values)}, condition_number={self.condition_number:.3g}, generated_events={self.generated_events}, accepted_events={self.accepted_events})"

    def __str__(self) -> str:
        """Describe numerical identifiability in scientific notation."""
        return f"Response rank {self.rank}/{len(self.singular_values)}, condition number {self.condition_number:.3g}"

    def __reduce__(self) -> tuple:
        """Restore an immutable spectrum through the record constructor."""
        return ResponseDiagnostics, (
            self.singular_values,
            self.generated_exposure,
            self.generated_events,
            self.accepted_events,
        )


class ExtractionError(ValueError):
    """An unidentifiable response or invalid corrected normalization.

    Parameters
    ----------
    message : str
        Actionable explanation of the failure.
    diagnostics : ResponseDiagnostics
        Response spectrum, conditioning, and exposure for investigation.
    """

    def __init__(self, message: str, diagnostics: ResponseDiagnostics) -> None:
        """Attach response diagnostics to the exception."""
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, repr=False, eq=False)
class ExtractionResult:
    """Corrected raw moments, first-order covariance, and response diagnostics.

    Parameters
    ----------
    data : MomentData
        Corrected raw Wigner-D moments. Pass this directly to ``analyze``.
    basis : MomentBasis
        Complete fitted expansion, including nuisance coordinates.
    measured : ndarray
        Weighted data projections in ``basis.observables`` order.
    response : ndarray
        Matrix mapping raw independent moments to measured projections.
    data_covariance, mc_covariance : Covariance
        Separate contributions in ``data.component_labels`` order. MC is zero
        under data-only policy because it was excluded, not because it is exact.
    statistics : MCStatistics
        Explicit policy underlying the reported MC uncertainty.
    diagnostics : ResponseDiagnostics
        Response identifiability and physical MC event counts.

    Notes
    -----
    Every data covariance uses grouped compound-Poisson event contributions.
    MC propagation is a first-order matrix-ratio approximation. Supplied weights,
    generation density, and beam calibration are treated as known. The covariance
    does not include fitted-weight or calibration nuisance uncertainty.
    """

    data: MomentData
    basis: MomentBasis
    measured: RealArray
    response: RealArray
    data_covariance: Covariance
    mc_covariance: Covariance
    statistics: MCStatistics
    diagnostics: ResponseDiagnostics

    def __post_init__(self) -> None:
        """Snapshot numerical arrays independently of the caller."""
        object.__setattr__(self, "measured", readonly(self.measured))
        object.__setattr__(self, "response", readonly(self.response))

    @property
    def covariance_scope(self) -> str:
        """Return an explicit description of included statistical uncertainty."""
        return str(self.statistics)

    def __repr__(self) -> str:
        """Summarize the expansion, uncertainty scope, and conditioning."""
        return f"ExtractionResult(basis={self.basis!r}, statistics={self.statistics!r}, condition_number={self.diagnostics.condition_number:.3g})"

    def __str__(self) -> str:
        """Describe extracted moments and their uncertainty scope."""
        return f"{self.data}; {self.covariance_scope}"

    def __reduce__(self) -> tuple:
        """Restore numerical snapshots through the record constructor."""
        return ExtractionResult, (
            self.data,
            self.basis,
            self.measured,
            self.response,
            self.data_covariance,
            self.mc_covariance,
            self.statistics,
            self.diagnostics,
        )


@dataclass(frozen=True, init=False, repr=False, eq=False)
class Acceptance:
    """A reusable MC response for acceptance-corrected Wigner-D moments.

    Notes
    -----
    Construct with separate generated and accepted samples, then reuse ``extract``
    for independent data samples sharing the frame, beam exposure, and detector
    response. Both MC samples are required. Accepted angles are truth angles. No resolution or
    incorrect-hypothesis migration is modeled. The full basis is solved even
    when only a few moments will be used in a later compatibility analysis.

    Examples
    --------
    Correct a constant efficiency using a tetrahedral angular quadrature. In
    real use, replace these columns with data and MC from your own source:

    >>> import momentous as mo
    >>> vertices = np.array([[1, 1, 1], [1, -1, -1],
    ...                      [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    >>> costheta = np.tile(vertices[:, 2], 5)
    >>> phi = np.tile(np.arctan2(vertices[:, 1], vertices[:, 0]), 5)
    >>> generated = mo.EventSample(costheta=costheta, phi=phi,
    ...     events=mo.EventGrouping.independent_rows())
    >>> accepted = mo.EventSample(costheta=costheta, phi=phi, weights=0.5,
    ...     events=mo.EventGrouping.independent_rows())
    >>> acceptance = mo.Acceptance(generated=generated, accepted=accepted,
    ...     basis=mo.MomentBasis(1),
    ...     statistics=mo.MCStatistics.data_only())
    >>> extraction = acceptance.extract(generated)
    >>> round(extraction.data.moments[(0, 0)].real)
    40
    >>> mo.analyze(extraction.data, mo.Waveset.from_max_l(1)).check().valid
    True
    """

    generated: EventSample
    accepted: EventSample
    basis: MomentBasis
    integration: MCIntegration
    statistics: MCStatistics
    diagnostics: ResponseDiagnostics
    _response: RealArray
    _accepted_features: RealArray
    _accepted_weights: RealArray
    _accepted_indices: NDArray[np.int64]
    _accepted_events: int
    _generated_exposure: RealArray
    _linked_indices: NDArray[np.int64] | None
    _execution: ld.Execution

    def __init__(
        self,
        generated: EventSample,
        accepted: EventSample,
        *,
        basis: MomentBasis | None = None,
        integration: MCIntegration | None = None,
        statistics: MCStatistics | None = None,
        execution: ld.Execution | None = None,
    ) -> None:
        """Estimate a moment response from two separately constructed MC samples.

        Parameters
        ----------
        generated, accepted : EventSample
            Generated exposure and selected MC with nonnegative weights and truth
            angles. Accepted hypothesis weights retain their supplied values. Both
            samples must use comparable exposure and weight units.
        basis : MomentBasis, optional
            Complete angular expansion, defaulting to L=4 with the generated
            sample's polarization mode. Set this explicitly for other truncations.
        integration : MCIntegration, optional
            Defaults to uniform decay-angle generation with matching beam exposure.
            Use ``importance`` for a known nonuniform generation density.
        statistics : MCStatistics, optional
            Defaults to ``linked()``: accepted MC is a selected subset of generated
            physical events with common IDs and matching truth coordinates.
            ``independent()`` asserts independent Poisson samples; ``data_only()``
            conditions on the response and excludes finite-MC uncertainty.
        execution : laddu.Execution, optional
            Defaults to double-precision JIT without thread limits.

        Raises
        ------
        TypeError
            If samples or configuration objects have the wrong types.
        ValueError
            If modes, weights, density, or the declared MC relationship are invalid.
        ExtractionError
            If the response is singular or has relative singular value <= 1e-10.

        Notes
        -----
        Statistical linking is declared here, never by sample construction. Both
        input samples remain unchanged. Linked validation checks IDs and supplied
        truth coordinates; it does not copy or replace accepted coordinates.
        Defaults assume a complete L=4 expansion, uniform generation, and linked
        MC. They are modeling assumptions, not facts inferred from numerical values.

        With test functions f and intensity functions c, the response is
        ``R = sum_accepted(w / q * outer(f, c)) / sum_generated(w / q)``.
        Dividing by generated exposure preserves the selection efficiency.
        """
        if not isinstance(generated, EventSample) or not isinstance(
            accepted, EventSample
        ):
            raise TypeError("generated and accepted must be EventSample objects")
        if basis is None:
            basis = MomentBasis(polarized=generated.polarized)
        if integration is None:
            integration = MCIntegration.uniform()
        if statistics is None:
            statistics = MCStatistics.linked()
        if not isinstance(basis, MomentBasis):
            raise TypeError("basis must be a MomentBasis")
        if not isinstance(integration, MCIntegration) or not isinstance(
            statistics, MCStatistics
        ):
            raise TypeError(
                "integration and statistics must be MCIntegration and MCStatistics objects"
            )
        if (
            generated.polarized != basis.polarized
            or accepted.polarized != basis.polarized
        ):
            raise ValueError("Both MC samples must match the MomentBasis polarization")
        _truth_by_event(generated)
        _truth_by_event(accepted)
        linked = (
            _link_events(generated, accepted) if statistics.mode == "linked" else None
        )
        runtime = execution_or_default(execution)
        generated_weights = integration._weights(generated, runtime)
        accepted_weights = integration._weights(accepted, runtime)
        exposure = float(np.sum(generated_weights))
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("Generated MC exposure must be positive and finite")
        features = basis._features(accepted, runtime)
        with np.errstate(over="raise", invalid="raise"):
            try:
                # Divide before the matrix product to avoid squaring raw weight scales.
                response = features.T @ (
                    features * (accepted_weights / exposure)[:, None]
                )
                response *= basis.expansion_scales[None, :]
            except FloatingPointError as error:
                raise ValueError(
                    "MC response exceeds numerical range; rescale weight units"
                ) from error
        singular = np.linalg.svd(response, compute_uv=False)
        diagnostics = ResponseDiagnostics(
            singular, exposure, generated.n_events, accepted.n_events
        )
        if diagnostics.rank != len(basis):
            raise ExtractionError(
                "Acceptance response is unidentifiable: "
                f"{diagnostics}; singular_values={singular.tolist()}. "
                "Check MC coverage, polarization sensitivity, and basis truncation. "
                "No regularization or moment dropping was performed.",
                diagnostics,
            )
        grouped_exposure = generated._aggregate(generated_weights)
        for name, value in (
            ("generated", generated),
            ("accepted", accepted),
            ("integration", integration),
            ("basis", basis),
            ("statistics", statistics),
            ("diagnostics", diagnostics),
            ("_response", readonly(response)),
            ("_accepted_features", readonly(features)),
            ("_accepted_weights", readonly(accepted_weights)),
            ("_accepted_indices", readonly(accepted._indices)),
            ("_accepted_events", accepted.n_events),
            ("_generated_exposure", readonly(grouped_exposure)),
            ("_linked_indices", None if linked is None else readonly(linked)),
            ("_execution", runtime),
        ):
            object.__setattr__(self, name, value)

    @property
    def response(self) -> RealArray:
        """Return an immutable response in ``basis.observables`` order."""
        return self._response.view()

    def extract(
        self,
        data: EventSample,
        *,
        execution: ld.Execution | None = None,
    ) -> ExtractionResult:
        """Solve raw moments and propagate grouped data and declared MC errors.

        Parameters
        ----------
        data : EventSample
            Observed angular sample, with signed weights allowed. Its physical
            events must be independent of MC and of each other. Group hypotheses
            by event ID; do not rescale their weight sums.
        execution : laddu.Execution, optional
            Override the response's execution object for data basis evaluation.

        Returns
        -------
        ExtractionResult
            Raw Wigner-D moments with full real covariance, ready for ``analyze``.
            Negative projections and exact-zero components are restored linearly.

        Raises
        ------
        ValueError
            If data is empty or polarization does not match the response.
        ExtractionError
            If corrected H00 is nonpositive or the solve exceeds numerical range.

        Notes
        -----
        Data covariance is ``sum_e g_e @ g_e.T``, with
        ``g_e = sum_h_in_e(w_h * f_h)`` under a compound-Poisson event model.
        The MC influence is ``-solve(R, delta_R @ H)``. It is computed per
        physical event, retaining linked generated/accepted correlations. The
        resulting covariance is first-order and excludes weight/calibration
        nuisance uncertainty. See ``docs/extraction.md`` for the derivation.
        """
        if not isinstance(data, EventSample):
            raise TypeError("extract requires an EventSample")
        if not len(data):
            raise ValueError("Data must contain at least one physical event")
        features = self.basis._features(
            data, self._execution if execution is None else execution
        )
        # Work in relative weight units before squaring. Raw deviations are
        # restored afterward, preserving covariance under extreme unit changes.
        weight_scale = float(np.max(np.abs(data.weights))) or 1.0
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                contributions = data._aggregate(
                    features * (data.weights / weight_scale)[:, None]
                )
                measured = contributions.sum(axis=0)
                moments = np.linalg.solve(self._response, measured)
                data_influence = np.linalg.solve(self._response, contributions.T).T
                data_cov = data_influence.T @ data_influence
                mc_cov = self._mc_covariance(moments)
                transform = self.basis._output_transform()
                values = (transform @ moments) * weight_scale
                measured = measured * weight_scale
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise ExtractionError(
                "Moment extraction exceeds numerical range; check response conditioning and rescale weights",
                self.diagnostics,
            ) from error
        if not np.all(np.isfinite(values)) or values[0] <= 0:
            raise ExtractionError(
                "Corrected H00 must be positive and finite; check data weights and acceptance modeling",
                self.diagnostics,
            )
        labels = [
            component
            for moment in self.basis.moments
            for component in moment.components
        ]
        raw: dict[Moment, complex] = {}
        stride = 1 if self.basis.polarized else 2
        for index, moment in enumerate(self.basis.moments):
            start = stride * index
            raw[moment] = (
                float(values[start])
                if self.basis.polarized
                else complex(values[start], values[start + 1])
            )
        return ExtractionResult(
            MomentData(
                raw,
                covariance=_scaled_covariance(
                    data_cov + mc_cov, transform, labels, weight_scale
                ),
            ),
            self.basis,
            measured,
            self.response,
            _scaled_covariance(data_cov, transform, labels, weight_scale),
            _scaled_covariance(mc_cov, transform, labels, weight_scale),
            self.statistics,
            self.diagnostics,
        )

    def _mc_covariance(self, moments: RealArray) -> RealArray:
        """Accumulate vector influences without a covariance over matrix entries."""
        if self.statistics.mode == "data_only":
            return np.zeros((len(self.basis), len(self.basis)))
        features = self._accepted_features
        projected = features @ (self.basis.expansion_scales * moments)
        rows = features * (self._accepted_weights * projected)[:, None]
        accepted = np.zeros((self._accepted_events, len(self.basis)))
        np.add.at(accepted, self._accepted_indices, rows)
        exposure = self.diagnostics.generated_exposure
        if self._linked_indices is not None:
            size = len(self._generated_exposure)
            numerator = np.zeros((size, len(self.basis)))
            numerator[self._linked_indices] = accepted
            numerator -= (
                self._generated_exposure[:, None] * (self._response @ moments)[None, :]
            )
            influence = -np.linalg.solve(self._response, (numerator / exposure).T).T
            return size / (size - 1) * (influence.T @ influence)
        accepted_influence = -np.linalg.solve(self._response, (accepted / exposure).T).T
        generated_influence = (
            self._generated_exposure[:, None] / exposure * moments[None, :]
        )
        return (
            accepted_influence.T @ accepted_influence
            + generated_influence.T @ generated_influence
        )

    def __repr__(self) -> str:
        """Summarize the response and uncertainty policy without dumping arrays."""
        return f"Acceptance(basis={self.basis!r}, statistics={self.statistics!r}, condition_number={self.diagnostics.condition_number:.3g})"

    def __str__(self) -> str:
        """Describe the corrected expansion and response identifiability."""
        return f"Acceptance for {self.basis}; {self.diagnostics}"


def _scaled_covariance(
    matrix: RealArray, transform: RealArray, labels: list[Component], scale: float
) -> Covariance:
    """Restore raw deviations without overflowing or underflowing their squares."""
    factor = transform @ covariance_factor(matrix, transform.shape[1])
    scaled = factor * scale
    if np.any(np.any(factor != 0, axis=1) & ~np.any(scaled != 0, axis=1)):
        raise ValueError(
            "Covariance deviations underflow numerical range; rescale raw units"
        )
    return Covariance._from_factor(labels, scaled)


def _truth_by_event(sample: EventSample) -> RealArray:
    """Require each physical MC event's hypotheses to share truth coordinates."""
    columns = [sample.costheta, sample.phi]
    if sample.polarization is not None:
        columns.extend(
            (
                np.asarray(sample.polarization.magnitude),
                np.asarray(sample.polarization.angle),
            )
        )
    truth = np.column_stack(columns)
    first = np.full(sample.n_events, len(sample), dtype=np.int64)
    np.minimum.at(first, sample._indices, np.arange(len(sample)))
    grouped = truth[first]
    if not np.allclose(truth, grouped[sample._indices], rtol=0, atol=1e-10):
        raise ValueError(
            "MC hypotheses in each physical event must share truth angles and polarization; reconstruction migration is unsupported"
        )
    return grouped


def _link_events(generated: EventSample, accepted: EventSample) -> NDArray[np.int64]:
    """Validate common event IDs and truth coordinates for linked MC statistics."""
    if generated.events.ids is None or accepted.events.ids is None:
        raise ValueError("Linked MC requires explicit event IDs in both samples")
    if generated.n_events < 2:
        raise ValueError("Linked MC needs at least two generated physical events")
    lookup = {identifier: index for index, identifier in enumerate(generated._ids)}
    if any(identifier not in lookup for identifier in accepted._ids):
        raise ValueError("Every accepted event ID must have a generated counterpart")
    indices = np.asarray(
        [lookup[identifier] for identifier in accepted._ids], dtype=np.int64
    )
    if not np.allclose(
        _truth_by_event(accepted),
        _truth_by_event(generated)[indices],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "Linked generated/accepted events must share truth coordinates"
        )
    return indices
