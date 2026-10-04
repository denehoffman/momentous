from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal
import laddu as ld


class Moment:
    def __init__(self, L: int | ld.L, M: int | float | ld.M) -> None:
        self.L: ld.L = ld.L(L)
        self.M: ld.M = ld.M(M)
        if self.M not in self.L.projections():
            raise ValueError(f"Invalid moment L={self.L}, M={self.M}")

    def __lt__(self, other: Moment) -> bool:
        return (self.L.value, self.M.value) < (other.L.value, other.M.value)

    def __str__(self) -> str:
        return f"H({int(self.L.value)}, {int(self.M.value):+})"

    def __repr__(self) -> str:
        return str(self)


class LinearlyPolarizedMoment:
    def __init__(
        self, L: int | ld.L, M: int | float | ld.M, variant: Literal[0, 1, 2]
    ) -> None:
        if variant not in (0, 1, 2):
            raise ValueError("Polarized moment variant must be 0, 1, or 2")
        self.moment: Moment = Moment(L, M)
        self.variant: Literal[0, 1, 2] = variant

    @property
    def L(self) -> ld.L:
        return self.moment.L

    @property
    def M(self) -> ld.M:
        return self.moment.M

    def __lt__(self, other: LinearlyPolarizedMoment) -> bool:
        return (self.L.value, self.M.value, self.variant) < (
            other.L.value,
            other.M.value,
            other.variant,
        )

    def __str__(self) -> str:
        s = f"H^{{{self.variant}}}({int(self.moment.L.value)}, {int(self.moment.M.value):+})"
        if self.variant == 2:
            s = rf"\Im{s}"
        return s

    def __repr__(self) -> str:
        return str(self)


class PartialWave:
    def __init__(self, L: int | ld.L, M: int | float | ld.M) -> None:
        self.L: ld.L = ld.L(L)
        self.M: ld.M = ld.M(M)
        if self.M not in self.L.projections():
            raise ValueError(f"Invalid partial wave L={self.L}, M={self.M}")

    def __lt__(self, other: PartialWave) -> bool:
        return (self.L.value, self.M.value) < (other.L.value, other.M.value)

    def __str__(self) -> str:
        return f"{self.L!s}_{{{int(self.M.value):+}}}"

    def __repr__(self) -> str:
        return str(self)


class ReflectivityPartialWave:
    def __init__(
        self, L: int | ld.L, M: int | float | ld.M, r: Literal["+", "-"] | ld.Parity
    ) -> None:
        self.partial_wave: PartialWave = PartialWave(L, M)
        self.r: ld.Parity = ld.Parity(r)

    @property
    def L(self) -> ld.L:
        return self.partial_wave.L

    @property
    def M(self) -> ld.M:
        return self.partial_wave.M

    def __lt__(self, other: ReflectivityPartialWave) -> bool:
        return (self.L.value, self.M.value, self.r.value) < (
            other.L.value,
            other.M.value,
            other.r.value,
        )

    def __str__(self) -> str:
        sign = "+" if self.r == ld.Parity.POSITIVE else "-"
        return f"{self.partial_wave.L!s}^{{({sign})}}_{{{int(self.partial_wave.M.value):+}}}"

    def __repr__(self) -> str:
        return str(self)


def waveset(waves: Sequence[tuple[int, int]]) -> list[PartialWave]:
    return [PartialWave(v[0], v[1]) for v in waves]


def polarized_waveset(
    waves: Sequence[tuple[int, int, Literal["+", "-"]]],
) -> list[ReflectivityPartialWave]:
    return [ReflectivityPartialWave(v[0], v[1], v[2]) for v in waves]


def momentset(Ls: Sequence[int]) -> list[Moment]:
    return [Moment(L, M) for L in Ls for M in ld.L(L).projections()]


def polarized_momentset(Ls: Sequence[int]) -> list[LinearlyPolarizedMoment]:
    return [
        LinearlyPolarizedMoment(L, M, variant)
        for L in Ls
        for M in ld.L(L).projections()
        for variant in [0, 1, 2]
    ]


@dataclass
class Measurement:
    """A moment normalized by H(0, 0), with optional Re/Im standard deviations."""

    moment: Moment
    value: complex
    uncertainty: complex | None = None


@dataclass
class PolarizedMeasurement:
    """A real observable normalized by H^0(0, 0); variant 2 is Im H^2."""

    moment: LinearlyPolarizedMoment
    value: float
    uncertainty: float | None = None
