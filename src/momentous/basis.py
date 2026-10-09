"""Independent Wigner-D moment coordinates and their intensity expansion."""

from dataclasses import dataclass
from typing import Self

import laddu as ld
import numpy as np

from momentous._sources import execution_or_default
from momentous.covariance import readonly
from momentous.domain import LInput, Moment, Observable, Waveset
from momentous.results import RealArray
from momentous.samples import EventSample


@dataclass(frozen=True, init=False, repr=False)
class MomentBasis:
    r"""A complete finite angular expansion with independent solve coordinates.

    Notes
    -----
    Construct with ``MomentBasis(max_L, polarized=False)`` or ``from_waves(pool)``.
    Rank means moment rank, not wave rank. Acceptance mixes
    moments; omitting higher intensity components can bias lower extracted ones.
    An intensity built from waves through l requires moments through 2*l.

    Use :math:`D^L_{M0}=d^L_{M0}(\theta)e^{-iM\phi}` in both modes. Unpolarized
    coordinates are Re H(L,M) for M >= 0 and Im H(L,M) for M > 0. Polarized
    coordinates are H^0 and H^1 for M >= 0 and Im H^2 for M > 0. The full output
    restores negative projections and exact zero components by symmetry.

    Examples
    --------
    >>> basis = MomentBasis(1)
    >>> len(basis), len(basis.moments)
    (4, 4)
    >>> MomentBasis.from_waves(Waveset.from_max_l(1)).max_L.value
    2
    """

    max_L: ld.L
    polarized: bool
    observables: tuple[Observable, ...]
    moments: tuple[Moment, ...]

    def __init__(self, max_L: LInput = 4, *, polarized: bool = False) -> None:
        """Construct a complete moment expansion through a maximum rank.

        Parameters
        ----------
        max_L : int, float, Fraction, or laddu.L, default=4
            Largest integral moment rank, including zero. This truncation is a
            modeling assumption; a wave expansion through l needs moments through 2*l.
        polarized : bool, default=False
            Include H^0, H^1, and Im H^2 for linearly polarized photons. Otherwise
            include complex unpolarized moments.

        Raises
        ------
        ValueError
            If the rank is not a nonnegative integer orbital angular momentum.
        TypeError
            If ``polarized`` is not a Boolean.

        Notes
        -----
        All signed moments are returned. Independent real solve coordinates omit
        symmetry-related duplicates and exact zeros; output covariance restores them.
        """
        if not isinstance(polarized, bool):
            raise TypeError("polarized must be a bool")
        rank = Moment(max_L, 0).L
        observables: list[Observable] = []
        moments: list[Moment] = []
        for L in range(rank.value + 1):
            for M in range(-L, L + 1):
                if polarized:
                    moments.extend(Moment(L, M, variant) for variant in (0, 1, 2))
                else:
                    moments.append(Moment(L, M))
            for M in range(L + 1):
                if polarized:
                    observables.extend(Moment(L, M, variant).real for variant in (0, 1))
                    if M:
                        observables.append(Moment(L, M, 2).real)
                else:
                    moment = Moment(L, M)
                    observables.append(moment.real)
                    if M:
                        observables.append(moment.imag)
        for name, value in (
            ("max_L", rank),
            ("polarized", polarized),
            ("observables", tuple(observables)),
            ("moments", tuple(moments)),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_waves(cls, waves: Waveset) -> Self:
        """Construct all moment ranks needed by a nonempty wave pool.

        Parameters
        ----------
        waves : Waveset
            Pool whose polarization and largest orbital rank define the basis.

        Returns
        -------
        MomentBasis
            Complete expansion through twice the largest wave rank. It does not
            drop coordinates based on the particular waves present in the pool.

        Raises
        ------
        ValueError
            If the pool is empty.
        """
        if not isinstance(waves, Waveset) or not len(waves):
            raise ValueError("MomentBasis.from_waves requires a nonempty Waveset")
        return cls(
            2 * max(wave.L.value for wave in waves.waves), polarized=waves.polarized
        )

    @property
    def expansion_scales(self) -> RealArray:
        """Return the intensity coefficient (2L+1)(2-delta_M0) per coordinate."""
        return readonly(
            np.array(
                [
                    (2 * observable.moment.L.value + 1)
                    * (1 if observable.moment.M.value == 0 else 2)
                    for observable in self.observables
                ],
                dtype=np.float64,
            )
        )

    def _features(
        self, sample: EventSample, execution: ld.Execution | None
    ) -> RealArray:
        """Evaluate independent test functions using laddu Wigner-D expressions."""
        if sample.polarized != self.polarized:
            raise ValueError("Sample polarization must match the MomentBasis")
        if not len(sample):
            return np.zeros((0, len(self)))
        dataset = sample._dataset()
        runtime = execution_or_default(execution)
        expressions: dict[str, ld.Expr] = {}
        for observable in self.observables:
            moment = observable.moment
            key = f"{moment.L.value},{int(moment.M.value)}"
            if key not in expressions:
                expressions[key] = ld.WignerD(moment.L, moment.M, ld.M(0)).D(
                    alpha=ld.scalar("phi"), beta=ld.scalar("costheta").acos()
                )
        angular = dataset.evaluate(expressions, execution=runtime)
        columns: list[RealArray] = []
        for observable in self.observables:
            moment = observable.moment
            key = f"{moment.L.value},{int(moment.M.value)}"
            D = np.asarray(angular[key])
            if moment.variant is None:
                column = D.real if observable.part == "real" else D.imag
            elif moment.variant == 0:
                column = D.real
            else:
                assert sample.polarization is not None
                magnitude = np.asarray(sample.polarization.magnitude)
                angle = np.asarray(sample.polarization.angle)
                column = (
                    magnitude * np.cos(2 * angle) * D.real
                    if moment.variant == 1
                    else magnitude * np.sin(2 * angle) * D.imag
                )
            columns.append(np.asarray(column, dtype=np.float64))
        return np.column_stack(columns)

    def _output_transform(self) -> RealArray:
        """Restore exact zeros and signed projections with all their correlations."""
        indices = {
            observable.key: index for index, observable in enumerate(self.observables)
        }
        rows: list[RealArray] = []
        for moment in self.moments:
            L, M = moment.L.value, int(moment.M.value)
            phase = (-1) ** abs(M) if M < 0 else 1
            for component in moment.components:
                row = np.zeros(len(self))
                part = component[2]
                if (part == "imag" or part == 2) and M == 0:
                    rows.append(row)
                    continue
                key = (L, abs(M), part)
                sign = -phase if M < 0 and (part == "imag" or part == 2) else phase
                row[indices[key]] = sign
                rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def __len__(self) -> int:
        """Return the number of independent real solve coordinates."""
        return len(self.observables)

    def __repr__(self) -> str:
        """Return an unambiguous constructor representation."""
        return f"MomentBasis({self.max_L.value}, polarized={self.polarized})"

    def __str__(self) -> str:
        """Describe the angular truncation and number of solve coordinates."""
        return f"{'Polarized' if self.polarized else 'Unpolarized'} moments through L={self.max_L.value} ({len(self)} independent components)"

    def __reduce__(self) -> tuple:
        """Restore the integral rank and polarization through the constructor."""
        return _restore_basis, (self.max_L.value, self.polarized)


def _restore_basis(max_L: LInput, polarized: bool) -> MomentBasis:
    """Restore a basis with its keyword-only polarization flag."""
    return MomentBasis(max_L, polarized=polarized)
