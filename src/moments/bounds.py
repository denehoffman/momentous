from moments.cg import c_matrix, polarized_c_matrix
from collections.abc import Sequence
from moments.waves import (
    Moment,
    PartialWave,
    LinearlyPolarizedMoment,
    ReflectivityPartialWave,
)
from dataclasses import dataclass
from numpy.typing import ArrayLike
from scipy.linalg import eigvalsh


@dataclass
class Bound:
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"[{self.lower:.3f}, {self.upper:.3f}]"

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def overlaps(self, other: Bound) -> bool:
        return self.lower <= other.upper and other.lower <= self.upper

    def is_compatible_with(self, value: float, uncertainty: float | None) -> bool:
        if uncertainty is None:
            return value in self
        return self.overlaps(Bound(value - uncertainty, value + uncertainty))


@dataclass
class ComplexBound:
    real_bound: Bound
    imag_bound: Bound

    def __str__(self) -> str:
        return f"{self.real_bound} + {self.imag_bound}i"

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, value: complex) -> bool:
        return value.real in self.real_bound and value.imag in self.imag_bound

    def overlaps(self, other: ComplexBound) -> bool:
        return self.real_bound.overlaps(other.real_bound) and self.imag_bound.overlaps(
            other.imag_bound
        )

    def is_compatible_with(self, value: complex, uncertainty: complex | None) -> bool:
        if uncertainty is None:
            return value in self
        other = ComplexBound(
            Bound(value.real - uncertainty.real, value.real + uncertainty.real),
            Bound(value.imag - uncertainty.imag, value.imag + uncertainty.imag),
        )
        return self.overlaps(other)


def get_bound(
    C: ArrayLike,
    N: ArrayLike | None = None,
) -> Bound:
    eigenvalues = eigvalsh(C, N)  # note that these are in ascending order
    return Bound(float(eigenvalues[0]), float(eigenvalues[-1]))


def get_bounds(moment: Moment, waves: Sequence[PartialWave]) -> ComplexBound:
    C_re, C_im = c_matrix(moment, waves)
    return ComplexBound(get_bound(C_re), get_bound(C_im))


def get_polarized_bound(
    moment: LinearlyPolarizedMoment, waves: Sequence[ReflectivityPartialWave]
) -> Bound:
    C = polarized_c_matrix(moment, waves)
    N = polarized_c_matrix(LinearlyPolarizedMoment(0, 0, 0), waves)
    return get_bound(C, N)
