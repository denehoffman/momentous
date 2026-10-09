"""Immutable quantum numbers and iterable wave pools."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from numbers import Integral
from typing import Literal

import laddu as ld

type LInput = int | float | Fraction | ld.L
type MInput = int | float | Fraction | ld.M
type Reflectivity = Literal["+", "-"] | ld.Parity
type Variant = Literal[0, 1, 2]
type MomentKey = tuple[LInput, MInput] | tuple[LInput, MInput, int]
type WaveKey = tuple[LInput, MInput] | tuple[LInput, MInput, Reflectivity]
type Component = tuple[int, int, Literal["real", "imag"] | Variant]
type ComponentKey = tuple[LInput, MInput, str | int]


def _quantum_numbers(L: LInput, M: MInput) -> tuple[ld.L, ld.M]:
    """Convert laddu inputs and enforce orbital projection membership."""
    if isinstance(L, bool) or isinstance(M, bool):
        raise ValueError("Quantum numbers cannot be booleans")
    try:
        rank, projection = ld.L(L), ld.M(M)
    except (ld.LadduError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid quantum numbers L={L!r}, M={M!r}") from error
    if projection not in rank.projections():
        raise ValueError(f"M={projection} is not a projection of L={rank.value}")
    return rank, projection


@dataclass(frozen=True, init=False)
class Wave:
    """An orbital partial wave, optionally in a reflectivity sector.

    Parameters
    ----------
    L, M : int, float, Fraction, or laddu quantum number
        Orbital rank and projection. Numeric representations must describe
        integral orbital quantum numbers with ``abs(M) <= L``.
    reflectivity : {'+', '-'} or laddu.Parity, optional
        Reflectivity sector; omission denotes an unpolarized wave.

    Examples
    --------
    >>> from fractions import Fraction
    >>> Wave(Fraction(2), -1, reflectivity="+")
    Wave(2, -1, reflectivity='+')
    """

    L: ld.L
    M: ld.M
    reflectivity: ld.Parity | None

    def __init__(
        self, L: LInput, M: MInput, reflectivity: Reflectivity | None = None
    ) -> None:
        """Convert quantum numbers and freeze this wave's identity."""
        rank, projection = _quantum_numbers(L, M)
        if isinstance(reflectivity, bool):
            raise ValueError("Reflectivity must be '+' or '-'")
        try:
            parity = None if reflectivity is None else ld.Parity(reflectivity)
        except (ld.LadduError, TypeError, ValueError) as error:
            raise ValueError("Reflectivity must be '+' or '-'") from error
        object.__setattr__(self, "L", rank)
        object.__setattr__(self, "M", projection)
        object.__setattr__(self, "reflectivity", parity)

    @property
    def key(self) -> tuple[int, int, int]:
        """Return a sortable identity independent of numeric input representation."""
        return (
            self.L.value,
            int(self.M.value),
            (0 if self.reflectivity is None else self.reflectivity.value),
        )

    def __str__(self) -> str:
        """Return compact spectroscopic notation."""
        sign = (
            ""
            if self.reflectivity is None
            else ("+" if self.reflectivity.value == 1 else "-")
        )
        sector = "" if not sign else f"^{sign}"
        return f"{self.L}_{int(self.M.value)}{sector}"

    def __repr__(self) -> str:
        """Return an explicit constructor representation."""
        sector = ""
        if self.reflectivity is not None:
            sign = "+" if self.reflectivity.value == 1 else "-"
            sector = f", reflectivity={sign!r}"
        return f"Wave({self.L.value}, {int(self.M.value)}{sector})"

    def __reduce__(self) -> tuple:
        """Serialize canonical numbers rather than unpicklable native laddu objects."""
        sector = (
            None
            if self.reflectivity is None
            else ("+" if self.reflectivity.value == 1 else "-")
        )
        return Wave, (self.L.value, int(self.M.value), sector)


@dataclass(frozen=True, init=False)
class Moment:
    """An angular moment, optionally a real linearly polarized observable.

    Parameters
    ----------
    L, M : int, float, Fraction, or laddu quantum number
        Integral angular rank and its projection.
    variant : {0, 1, 2}, optional
        Polarized variant: ``H^0``, ``H^1``, or ``Im H^2``. Omission denotes
        a complex unpolarized moment.

    Examples
    --------
    >>> Moment(1.0, 0)
    Moment(1, 0)
    >>> print(Moment(2, 1, variant=2))
    Im H^2(2, +1)
    """

    L: ld.L
    M: ld.M
    variant: Variant | None

    def __init__(self, L: LInput, M: MInput, variant: int | None = None) -> None:
        """Convert quantum numbers and validate the polarized variant."""
        rank, projection = _quantum_numbers(L, M)
        if variant is not None and (
            isinstance(variant, bool)
            or not isinstance(variant, Integral)
            or variant not in (0, 1, 2)
        ):
            raise ValueError("Polarized variant must be 0, 1, or 2")
        object.__setattr__(self, "L", rank)
        object.__setattr__(self, "M", projection)
        object.__setattr__(self, "variant", variant)

    @property
    def key(self) -> tuple[int, int] | tuple[int, int, Variant]:
        """Return the canonical mapping key."""
        base = (self.L.value, int(self.M.value))
        return base if self.variant is None else (*base, self.variant)

    @property
    def components(self) -> tuple[Component, ...]:
        """Return covariance labels, with both Re/Im slots for complex moments."""
        L, M = self.L.value, int(self.M.value)
        return (
            ((L, M, "real"), (L, M, "imag"))
            if self.variant is None
            else ((L, M, self.variant),)
        )

    @property
    def real(self) -> Observable:
        """Return Re H, or the scalar polarized observable already specified.

        Examples
        --------
        >>> print(Moment(1, 1).real)
        Re H(1, +1)
        """
        return Observable(self, "real" if self.variant is None else None)

    @property
    def imag(self) -> Observable:
        """Return Im H for a complex unpolarized moment.

        Raises
        ------
        ValueError
            If this moment already denotes a scalar polarized observable.
        """
        return Observable(self, "imag")

    def __str__(self) -> str:
        """Return scientific observable notation."""
        name = "H" if self.variant is None else f"H^{self.variant}"
        if self.variant == 2:
            name = f"Im {name}"
        return f"{name}({self.L.value}, {int(self.M.value):+})"

    def __repr__(self) -> str:
        """Return an explicit constructor representation."""
        variant = "" if self.variant is None else f", variant={self.variant}"
        return f"Moment({self.L.value}, {int(self.M.value)}{variant})"

    def __reduce__(self) -> tuple:
        """Serialize the canonical orbital labels and polarized variant."""
        return Moment, (self.L.value, int(self.M.value), self.variant)


def as_moment(value: Moment | MomentKey) -> Moment:
    """Canonicalize an object or keyed moment label."""
    if isinstance(value, Moment):
        return value
    if not isinstance(value, tuple) or len(value) not in (2, 3):
        raise ValueError("Moment keys must be (L, M) or (L, M, variant)")
    return Moment(*value)


@dataclass(frozen=True)
class Observable:
    """One real scalar observable selected from an angular moment.

    Parameters
    ----------
    moment : Moment
        Unpolarized complex moment or scalar polarized observable.
    part : {'real', 'imag'}, optional
        Required for unpolarized moments; omitted for polarized observables.

    Examples
    --------
    >>> Moment(1, 1).imag == Observable(Moment(1, 1), 'imag')
    True
    >>> print(Observable(Moment(2, 1, variant=2)))
    Im H^2(2, +1)
    """

    moment: Moment
    part: Literal["real", "imag"] | None = None

    def __post_init__(self) -> None:
        """Require an unambiguous scalar selection in the moment's mode."""
        if not isinstance(self.moment, Moment):
            raise TypeError("An Observable requires a Moment")
        if self.moment.variant is None:
            if self.part not in ("real", "imag"):
                raise ValueError("Select real or imag for an unpolarized moment")
        elif self.part is not None:
            raise ValueError("Polarized moments already denote scalar observables")

    @classmethod
    def from_key(cls, key: ComponentInput) -> Observable:
        """Construct a scalar observable from canonical tuple shorthand.

        Parameters
        ----------
        key : tuple, Observable, or polarized Moment
            A scalar label or existing scalar selection.

        Returns
        -------
        Observable
            Canonical immutable scalar identity.
        """
        L, M, part = as_component(key)
        if isinstance(part, str):
            return cls(Moment(L, M), part)
        return cls(Moment(L, M, variant=part))

    @property
    def key(self) -> Component:
        """Return the canonical tuple used for matrix labels and serialization."""
        if self.part is None:
            return self.moment.components[0]
        return self.moment.L.value, int(self.moment.M.value), self.part

    def __str__(self) -> str:
        """Format the selected scientific observable."""
        if self.part is None:
            return str(self.moment)
        return f"{'Re' if self.part == 'real' else 'Im'} {self.moment}"

    def __repr__(self) -> str:
        """Return an unambiguous scalar constructor representation."""
        return f"Observable({self.moment!r}, part={self.part!r})"


type ComponentInput = Observable | Moment | ComponentKey


def as_component(key: ComponentInput) -> Component:
    """Canonicalize a real/imaginary or polarized scalar label."""
    if isinstance(key, Observable):
        return key.key
    if isinstance(key, Moment):
        if key.variant is None:
            raise ValueError("Select Moment.real or Moment.imag for a complex moment")
        return key.components[0]
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError("A component label must be (L, M, part or variant)")
    L, M, part = key
    if part in ("real", "imag"):
        moment = Moment(L, M)
        return moment.L.value, int(moment.M.value), "real" if part == "real" else "imag"
    if (
        isinstance(part, bool)
        or not isinstance(part, Integral)
        or part not in (0, 1, 2)
    ):
        raise ValueError("Select 'real', 'imag', or a polarized variant 0, 1, 2")
    moment = Moment(L, M, variant=int(part))
    assert moment.variant is not None
    return moment.L.value, int(moment.M.value), moment.variant


def size_limits(minimum: int, maximum: int | None, size: int) -> tuple[int, int]:
    """Validate cardinality bounds, including the optional empty subset."""
    upper = size if maximum is None else maximum
    if any(
        not isinstance(x, Integral) or isinstance(x, bool) for x in (minimum, upper)
    ):
        raise ValueError("Subset size limits must be integers")
    if not 0 <= minimum <= upper <= size:
        raise ValueError("Require 0 <= min_size <= max_size <= number of waves")
    return int(minimum), int(upper)


@dataclass(frozen=True, init=False)
class Waveset:
    """A fixed wave basis whose iterator yields nonempty subset wavesets.

    Parameters
    ----------
    waves : iterable of Wave or tuples
        Explicit ``(L, M)`` or ``(L, M, reflectivity)`` definitions.
    polarized : bool, optional
        Mode for an empty set, or an explicit consistency check for a nonempty
        set. Otherwise inferred from the supplied waves.

    Notes
    -----
    ``waves`` contains the basis in canonical order; ``len(pool)`` counts those
    waves. Iteration enumerates subsets, **not** individual waves. Enumeration
    is lazy but has exponential total size.

    Examples
    --------
    >>> pool = Waveset([(1, 0), (0, 0)])
    >>> len(pool), len(list(pool))
    (2, 3)
    >>> pool.waves
    (Wave(0, 0), Wave(1, 0))
    """

    waves: tuple[Wave, ...]
    polarized: bool

    def __init__(
        self, waves: Iterable[Wave | WaveKey], *, polarized: bool | None = None
    ) -> None:
        """Canonicalize and freeze a homogeneous, duplicate-free basis."""
        members = tuple(
            wave if isinstance(wave, Wave) else Wave(*wave) for wave in waves
        )
        modes = {wave.reflectivity is not None for wave in members}
        if len(modes) > 1:
            raise ValueError("A waveset cannot mix polarized and unpolarized waves")
        mode = bool(modes and True in modes) if polarized is None else polarized
        if not isinstance(mode, bool) or (modes and modes != {mode}):
            raise ValueError("Polarization must match the supplied waves")
        if len(set(members)) != len(members):
            raise ValueError("A waveset cannot contain duplicate waves")
        object.__setattr__(self, "waves", tuple(sorted(members, key=lambda w: w.key)))
        object.__setattr__(self, "polarized", mode)

    @classmethod
    def from_max_l(
        cls,
        max_l: LInput,
        *,
        reflectivities: Iterable[Reflectivity] | None = None,
    ) -> Waveset:
        """Construct every signed projection through an orbital rank.

        Parameters
        ----------
        max_l : int, float, Fraction, or laddu.L
            Largest integral orbital rank, inclusive.
        reflectivities : iterable of {'+', '-'} or laddu.Parity, optional
            Polarized sectors to include, such as ``('+',)`` or ``('+', '-')``.
            Omission constructs unpolarized waves. Empty or duplicate sectors
            are rejected.

        Returns
        -------
        Waveset
            The complete requested basis.

        Examples
        --------
        >>> len(Waveset.from_max_l(1))
        4
        >>> len(Waveset.from_max_l(1, reflectivities=('+', '-')))
        8
        """
        rank, _ = _quantum_numbers(max_l, 0)
        sectors: tuple[Reflectivity | None, ...] = (
            (None,) if reflectivities is None else tuple(reflectivities)
        )
        if not sectors:
            raise ValueError("Select at least one reflectivity sector")
        if reflectivities is not None and any(sector is None for sector in sectors):
            raise ValueError("Reflectivity sectors must be '+' or '-'")
        return cls(
            Wave(L, M, sector)
            for L in range(rank.value + 1)
            for M in range(-L, L + 1)
            for sector in sectors
        )

    def __len__(self) -> int:
        """Return the number of basis waves, not the powerset size."""
        return len(self.waves)

    def __iter__(self) -> Iterator[Waveset]:
        """Yield nonempty subsets in increasing size and canonical order."""
        return self.powerset()

    def powerset(
        self, min_size: int = 1, max_size: int | None = None
    ) -> Iterator[Waveset]:
        """Iterate subsets within inclusive cardinality limits.

        Parameters
        ----------
        min_size : int, default 1
            Smallest subset; use zero to include the empty set.
        max_size : int, optional
            Largest subset, defaulting to the complete basis size.

        Yields
        ------
        Waveset
            Independent immutable subsets preserving polarization.

        Notes
        -----
        Enumeration is lazy, but exhausting the iterator can take exponential
        time. Access ``.waves`` to iterate the individual basis waves instead.
        """
        if not self.waves and min_size == 1 and max_size is None:
            return
        lower, upper = size_limits(min_size, max_size, len(self))
        for size in range(lower, upper + 1):
            for members in combinations(self.waves, size):
                yield Waveset(members, polarized=self.polarized)

    def __str__(self) -> str:
        """Return a compact list of spectroscopic wave labels."""
        return "{" + ", ".join(str(wave) for wave in self.waves) + "}"

    def __repr__(self) -> str:
        """Describe the basis without expanding its powerset."""
        mode = ", polarized=True" if self.polarized else ""
        return f"Waveset({list(self.waves)!r}{mode})"
