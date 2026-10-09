"""Labeled real covariance and immutable numerical input storage."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from momentous.domain import (
    Component,
    ComponentInput,
    Moment,
    MomentKey,
    as_component,
    as_moment,
)
from momentous.results import RealArray


def readonly[T: np.generic](array: NDArray[T]) -> NDArray[T]:
    """Copy an array onto immutable byte storage."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def covariance_factor(covariance: ArrayLike, size: int) -> RealArray:
    """Validate a real PSD covariance, preserving exact zero-variance rows."""
    raw = np.asarray(covariance)
    if np.iscomplexobj(raw):
        raise ValueError("Covariance must be real; complex covariance is insufficient")
    cov = np.array(raw, dtype=np.float64, copy=True)
    if cov.shape != (size, size):
        raise ValueError(f"Covariance must have shape ({size}, {size})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance must contain only finite values")
    if not size:
        return cov
    scale = float(np.max(np.abs(cov)))
    tolerance = 64 * np.finfo(float).eps * size * scale
    if np.max(np.abs(cov - cov.T)) > tolerance:
        raise ValueError("Covariance must be symmetric")
    cov = cov / 2 + cov.T / 2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Covariance has an excessive numerical scale; rescale units")
    if np.any(np.diag(cov) < 0) or eigenvalues[0] < -tolerance:
        raise ValueError("Covariance must be positive semidefinite")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    factor[np.diag(cov) == 0] = 0
    return np.asarray(factor, dtype=np.float64)


def _labels(components: Iterable[ComponentInput]) -> tuple[Component, ...]:
    """Require canonical, unique labels from a single polarization mode."""
    labels = tuple(as_component(component) for component in components)
    if len(set(labels)) != len(labels):
        raise ValueError("Covariance components must be unique")
    if len({isinstance(label[2], str) for label in labels}) > 1:
        raise ValueError("Covariance cannot mix polarized and unpolarized labels")
    return labels


@dataclass(frozen=True, init=False, repr=False, eq=False)
class Covariance:
    """Immutable uncertainty information for raw moment measurements.

    Parameters
    ----------
    matrix : array_like, shape (n, n)
        Finite symmetric positive semidefinite covariance of the raw supplied
        moments. Exact components have zero rows and columns.
    components : iterable of Observable, polarized Moment, or tuple
        One scalar per row. Use ``Moment.real``/``Moment.imag`` for unpolarized
        moments, or a polarized Moment directly. Tuple shorthand is
        ``(L, M, 'real'/'imag')`` or ``(L, M, variant)``. Labels must be unique and
        use one polarization mode. Quantum numbers accept laddu input forms.

    Notes
    -----
    Construct from a labeled matrix, use ``from_uncertainties`` for independent
    standard deviations, or ``exact()`` for no uncertainty. All three constructors
    produce the same object accepted by ``MomentData(covariance=...)``.
    Labels describe the order of this matrix. Analysis aligns it to measurement
    order and requires exactly the supplied components, including both Re/Im
    slots of real-valued unpolarized measurements. Labeling cannot detect a
    mislabeled matrix or a different moment convention.

    Examples
    --------
    >>> covariance = Covariance([[0.04, 0.01], [0.01, 0.09]],
    ...                         components=[(1, 1, 'imag'), (1, 1, 'real')])
    >>> covariance.aligned([(1, 1, 'real'), (1, 1, 'imag')]).tolist()
    [[0.09, 0.01], [0.01, 0.04]]
    >>> independent = Covariance.from_uncertainties({(1, 1): 0.1 + 0.2j})
    >>> independent.components
    ((1, 1, 'real'), (1, 1, 'imag'))
    >>> Covariance.exact().aligned(independent.components).tolist()
    [[0.0, 0.0], [0.0, 0.0]]
    """

    components: tuple[Component, ...] | None
    _matrix: RealArray | None
    _factor: RealArray

    def __init__(self, matrix: ArrayLike, components: Iterable[ComponentInput]) -> None:
        """Validate labeled covariance and take an immutable owned snapshot."""
        labels = _labels(components)
        snapshot = np.array(matrix, copy=True)
        factor = covariance_factor(snapshot, len(labels))
        raw = np.asarray(snapshot, dtype=np.float64)
        object.__setattr__(self, "components", labels)
        object.__setattr__(self, "_matrix", readonly(raw / 2 + raw.T / 2))
        object.__setattr__(self, "_factor", readonly(factor))

    @classmethod
    def from_uncertainties[Key: Moment | MomentKey](
        cls, uncertainties: Mapping[Key, complex]
    ) -> Self:
        """Construct covariance from independent raw standard deviations.

        Parameters
        ----------
        uncertainties : mapping
            One deviation per supplied moment, including the normalizer. Moment
            objects and tuple keys are accepted. For unpolarized moments a complex
            deviation specifies Re/Im separately; a real deviation gives exact Im.
            Polarized deviations must be real. Use explicit zeros for exact values.

        Returns
        -------
        Covariance
            Diagonal covariance with scalar labels derived from the moment keys.

        Raises
        ------
        ValueError
            If labels are duplicated or mixed, or deviations are negative,
            nonfinite, or complex for polarized moments.

        Notes
        -----
        All correlations are assumed zero. Analysis requires coverage of every
        supplied moment; omitted entries never implicitly mean exact values.
        Standard deviations are retained without squaring until a matrix is
        requested, preserving normalization under extreme common unit rescaling.

        Examples
        --------
        >>> covariance = Covariance.from_uncertainties({(0, 0): 1, (1, 1): 0.2 + 0.3j})
        >>> covariance.components
        ((0, 0, 'real'), (0, 0, 'imag'), (1, 1, 'real'), (1, 1, 'imag'))
        """
        labels: list[Component] = []
        deviations: list[float] = []
        seen: set[Moment] = set()
        for key, value in uncertainties.items():
            moment = as_moment(key)
            if moment in seen:
                raise ValueError(f"Duplicate physical moment: {moment}")
            seen.add(moment)
            labels.extend(moment.components)
            if moment.variant is not None:
                if np.iscomplexobj(value):
                    raise ValueError("Polarized uncertainties must be real")
                deviations.append(float(value.real))
            else:
                number = complex(value)
                deviations.extend((number.real, number.imag))
        return cls._from_deviations(labels, deviations)

    @classmethod
    def _from_deviations(
        cls, components: Iterable[ComponentInput], deviations: ArrayLike
    ) -> Self:
        """Validate and store independent deviations without squaring them."""
        labels = _labels(components)
        std = np.asarray(deviations, dtype=np.float64)
        if std.shape != (len(labels),) or not len(labels):
            raise ValueError("Supply one standard deviation per component")
        if not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError(
                "Uncertainties must be finite nonnegative standard deviations"
            )
        instance = cls.__new__(cls)
        object.__setattr__(instance, "components", labels)
        object.__setattr__(instance, "_matrix", None)
        object.__setattr__(instance, "_factor", readonly(np.diag(std)))
        return instance

    @classmethod
    def _from_factor(
        cls, components: Iterable[ComponentInput], factor: ArrayLike
    ) -> Self:
        """Store correlated deviations without squaring extreme raw unit scales."""
        labels = _labels(components)
        raw = np.asarray(factor)
        if np.iscomplexobj(raw):
            raise ValueError("Covariance factors must be real")
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(labels):
            raise ValueError("Covariance factor must have one row per component")
        if not np.all(np.isfinite(array)):
            raise ValueError("Covariance factors must be finite; rescale raw units")
        instance = cls.__new__(cls)
        object.__setattr__(instance, "components", labels)
        object.__setattr__(instance, "_matrix", None)
        object.__setattr__(instance, "_factor", readonly(array))
        return instance

    @classmethod
    def exact(cls) -> Self:
        """Declare that all moments have no uncertainty.

        Returns
        -------
        Covariance
            A reusable declaration with ``components=None``. Alignment creates
            a zero matrix for any requested polarized or unpolarized labels.

        Examples
        --------
        >>> Covariance.exact().aligned([(0, 0, 0), (1, 0, 1)]).tolist()
        [[0.0, 0.0], [0.0, 0.0]]
        """
        instance = cls.__new__(cls)
        object.__setattr__(instance, "components", None)
        object.__setattr__(instance, "_matrix", readonly(np.zeros((0, 0))))
        object.__setattr__(instance, "_factor", readonly(np.zeros((0, 0))))
        return instance

    @property
    def matrix(self) -> RealArray:
        """Return a read-only matrix in component order.

        Returns
        -------
        ndarray
            The labeled covariance. An unaligned ``exact()`` declaration returns
            shape (0, 0); use ``aligned(labels)`` to obtain its desired dimension.

        Raises
        ------
        ValueError
            If squaring stored deviations overflows or loses positive variances
            to underflow; rescale the raw units. Analysis can still normalize
            finite deviations without squaring them.
        """
        if self._matrix is not None:
            return self._matrix.view()
        try:
            with np.errstate(over="raise", invalid="raise"):
                matrix = self._factor @ self._factor.T
        except FloatingPointError as error:
            raise ValueError(
                "Covariance exceeds numerical range; rescale raw units"
            ) from error
        if np.any((np.diag(matrix) == 0) & np.any(self._factor != 0, axis=1)):
            raise ValueError("Covariance underflows numerical range; rescale raw units")
        return readonly(matrix)

    def _indices(self, components: Iterable[ComponentInput]) -> list[int]:
        """Require exactly matching labels and return their permutation."""
        labels = _labels(components)
        if self.components is None:
            return list(range(len(labels)))
        if set(labels) != set(self.components):
            missing = set(labels) - set(self.components)
            extra = set(self.components) - set(labels)
            raise ValueError(
                "Covariance labels must match every supplied component; "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        return [self.components.index(label) for label in labels]

    def aligned(self, components: Iterable[ComponentInput]) -> RealArray:
        """Return covariance aligned to an explicitly requested component order.

        Parameters
        ----------
        components : iterable of Observable, polarized Moment, or tuple
            The same physical labels in the desired order, or any valid labels
            for an ``exact()`` declaration.

        Returns
        -------
        ndarray
            An immutable covariance snapshot in the requested order.

        Raises
        ------
        ValueError
            If labels are missing, extra, or duplicated.
        """
        indices = self._indices(components)
        if self.components is None:
            return readonly(np.zeros((len(indices), len(indices))))
        return readonly(self.matrix[np.ix_(indices, indices)])

    def _aligned_factor(self, components: Iterable[ComponentInput]) -> RealArray:
        """Align factor rows while retaining the full underlying correlations."""
        indices = self._indices(components)
        if self.components is None:
            return np.zeros((len(indices), len(indices)))
        return self._factor[indices]

    def _is_independent(self) -> bool:
        """Identify diagonal deviation storage for concise representations."""
        size = self._factor.shape[0]
        return (
            self._matrix is None
            and self._factor.shape == (size, size)
            and np.array_equal(self._factor, np.diag(np.diag(self._factor)))
        )

    def __repr__(self) -> str:
        """Summarize covariance dimensions without dumping the matrix."""
        if self.components is None:
            return "Covariance.exact()"
        constructor = (
            "Covariance.from_uncertainties" if self._is_independent() else "Covariance"
        )
        size = len(self.components)
        return f"{constructor}(components={size}, shape=({size}, {size}))"

    def __str__(self) -> str:
        """Describe the labeled scalar covariance."""
        if self.components is None:
            return "Exact moments (zero covariance)"
        kind = "Independent" if self._is_independent() else "Real"
        return f"{kind} covariance of {len(self.components)} labeled components"

    def __reduce__(self) -> tuple:
        """Restore validation and immutable storage after process serialization."""
        if self.components is None:
            return Covariance.exact, ()
        if self._matrix is None:
            return Covariance._from_factor, (self.components, self._factor)
        return Covariance, (self.matrix, self.components)
