"""Real-component covariance validation and correlated moment normalization."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from warnings import warn

import numpy as np

from momentous.covariance import Covariance, readonly
from momentous.domain import (
    Component,
    Moment,
    MomentKey,
    Observable,
    Waveset,
    as_moment,
)
from momentous.operators import moment_operators, normalization_metric
from momentous.results import ComplexArray, NormalizationWarning, RealArray


def _canonical_mapping[Key: Moment | MomentKey](
    values: Mapping[Key, complex],
) -> dict[Moment, complex]:
    """Preserve mapping order while rejecting duplicate physical labels."""
    result: dict[Moment, complex] = {}
    for key, value in values.items():
        moment = as_moment(key)
        if moment in result:
            raise ValueError(f"Duplicate physical moment: {moment}")
        result[moment] = value
    return result


@dataclass(frozen=True, init=False, repr=False, eq=False)
class MomentData:
    """An immutable raw measurement and its uncertainty in the library convention.

    Parameters
    ----------
    moments : mapping
        Moment or tuple keys with raw Wigner-D values, including positive real
        H00. Polarized values must be real. Yield-only measurements are valid;
        compatibility analysis requires at least one other moment.
    covariance : Covariance
        Labeled matrix, independent uncertainties, or ``Covariance.exact()``.

    Raises
    ------
    ValueError
        If labels, uncertainty coverage, values, or normalization are invalid.

    Notes
    -----
    Construction fixes mapping order, validates polarization and covariance
    coverage, and snapshots values. See ``analyze`` for extraction formulas and
    the required convention. A different convention cannot be detected from
    numeric values. Normalization and first-order propagation occur in analyze.

    Examples
    --------
    >>> data = MomentData({(0, 0): 10, (1, 1): 1 + 2j}, covariance=Covariance.exact())
    >>> data.polarized, data.values.shape
    (False, (4,))
    """

    covariance: Covariance
    polarized: bool
    _moments: Mapping[Moment, complex]
    _values: RealArray
    _factor: RealArray
    component_labels: tuple[Component, ...]

    def __init__[Key: Moment | MomentKey](
        self, moments: Mapping[Key, complex], *, covariance: Covariance
    ) -> None:
        """Validate raw measurement semantics and own immutable scalar inputs."""
        if not isinstance(covariance, Covariance):
            raise TypeError(
                "Use a Covariance constructor to label every row or declare exact data"
            )
        items = _canonical_mapping(moments)
        modes = {moment.variant is not None for moment in items}
        if len(modes) > 1:
            raise ValueError("MomentData cannot mix polarized and unpolarized moments")
        polarized = True in modes
        denominator = Moment(0, 0, variant=0 if polarized else None)
        if denominator not in items:
            raise ValueError("Raw moments must include H(0,0) or H^0(0,0)")
        labels: list[Component] = []
        values: list[float] = []
        for moment, value in items.items():
            labels.extend(moment.components)
            if polarized:
                if np.iscomplexobj(value):
                    raise ValueError("Polarized measurements must be real")
                items[moment] = float(value.real)
                values.append(float(value.real))
            else:
                number = complex(value)
                items[moment] = number
                values.extend((number.real, number.imag))
        scalar = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(scalar)):
            raise ValueError("Moment values must be finite")
        factor = covariance._aligned_factor(labels)
        index = (1 if polarized else 2) * tuple(items).index(denominator)
        if scalar[index] <= 0 or (not polarized and scalar[index + 1] != 0):
            raise ValueError("The normalization moment must be positive and real")
        if not polarized and np.any(factor[index + 1] != 0):
            raise ValueError("The imaginary normalization component must be exact zero")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "polarized", polarized)
        object.__setattr__(self, "_moments", MappingProxyType(items))
        object.__setattr__(self, "_values", readonly(scalar))
        object.__setattr__(self, "_factor", readonly(factor))
        object.__setattr__(self, "component_labels", tuple(labels))

    @property
    def moments(self) -> Mapping[MomentKey, complex]:
        """Return raw values in insertion order with canonical tuple keys."""
        return MappingProxyType(
            {moment.key: value for moment, value in self._moments.items()}
        )

    @property
    def values(self) -> RealArray:
        """Return an immutable raw scalar vector in component-label order."""
        return self._values.view()

    @property
    def observables(self) -> tuple[Observable, ...]:
        """Return scalar objects in raw values and covariance order."""
        return tuple(Observable.from_key(label) for label in self.component_labels)

    def __repr__(self) -> str:
        """Summarize the measurement without dumping values or matrices."""
        return f"MomentData(moments={len(self._moments)}, polarized={self.polarized}, covariance={self.covariance!r})"

    def __str__(self) -> str:
        """Describe the number and mode of raw measured moments."""
        return f"{len(self._moments)} raw {'polarized' if self.polarized else 'unpolarized'} moments"

    def __reduce__(self) -> tuple:
        """Revalidate serialized values and restore immutable storage."""
        return _restore_moment_data, (dict(self.moments), self.covariance)


def _restore_moment_data(
    moments: Mapping[MomentKey, complex], covariance: Covariance
) -> MomentData:
    """Restore a measurement through its validated constructor."""
    return MomentData(moments, covariance=covariance)


@dataclass(frozen=True, repr=False)
class PreparedData:
    """Owned normalized scalar inputs and raw wave-basis operators."""

    moments: tuple[Moment, ...]
    labels: tuple[Component, ...]
    values: RealArray
    covariance: RealArray
    factor: RealArray
    operators: ComplexArray
    metric: ComplexArray
    normalized_values: Mapping[Moment, complex]

    def __repr__(self) -> str:
        """Summarize input dimensions without printing matrices."""
        return (
            f"PreparedData(moments={len(self.moments)}, components={len(self.labels)})"
        )


def prepare_data(measurements: MomentData, waves: Waveset) -> PreparedData:
    """Flatten measurements, propagate denominator uncertainty, and build operators."""
    if not isinstance(waves, Waveset) or not len(waves):
        raise ValueError("Analysis requires a nonempty Waveset pool")
    if not isinstance(measurements, MomentData):
        raise TypeError("analyze requires a MomentData object")
    if measurements.polarized != waves.polarized:
        raise ValueError("Moment polarization must match the Waveset pool")
    items = measurements._moments
    if len(items) < 2:
        raise ValueError("Supply at least one moment besides the normalization moment")
    denominator = Moment(0, 0, variant=0 if waves.polarized else None)
    labels = measurements.component_labels
    scalar = measurements.values
    factor = measurements._factor
    operators = [
        operator for moment in items for operator in moment_operators(moment, waves)
    ]

    stride = 1 if waves.polarized else 2
    index = stride * tuple(items).index(denominator)
    total = scalar[index]
    # Ratios avoid squaring the raw denominator or forming overflowing inverse
    # Jacobian entries. This is still the full first-order J @ F transformation.
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scalar = scalar / total
            relative_factor = factor / total
            denominator_error = float(np.linalg.norm(relative_factor[index]))
            factor = relative_factor - scalar[:, None] * relative_factor[index]
            # H00/H00 is an exact identity, not a noisy extra constraint.
            factor[index] = 0
            scalar[index] = 1
            if not waves.polarized:
                factor[index + 1] = 0
                scalar[index + 1] = 0
            propagated = np.asarray(factor @ factor.T, dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError(
            "Normalization exceeds numerical range; rescale all raw moments "
            "and uncertainties consistently"
        ) from error
    if denominator_error >= 1 / 3:
        warn(
            "The normalization moment has relative standard uncertainty >= 1/3; "
            "its three-standard-deviation interval reaches zero. First-order "
            "ratio covariance may be unreliable.",
            NormalizationWarning,
            stacklevel=3,
        )

    normalized_values: dict[Moment, complex] = {}
    stride = 1 if waves.polarized else 2
    for index, moment in enumerate(items):
        start = stride * index
        normalized_values[moment] = (
            float(scalar[start])
            if waves.polarized
            else complex(scalar[start], scalar[start + 1])
        )
    matrices = np.asarray(operators, dtype=np.complex128).reshape(
        len(scalar), len(waves), len(waves)
    )
    metric = normalization_metric(waves)
    if np.linalg.eigvalsh(metric)[0] <= 0:
        raise ValueError("Normalization must define a positive definite metric")
    return PreparedData(
        tuple(items),
        tuple(labels),
        readonly(scalar),
        readonly(propagated),
        readonly(factor),
        readonly(matrices),
        readonly(metric),
        MappingProxyType(normalized_values),
    )
