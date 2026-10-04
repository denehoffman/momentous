from collections.abc import Sequence
from moments.waves import (
    PartialWave,
    Moment,
    ReflectivityPartialWave,
    LinearlyPolarizedMoment,
)
import laddu as ld
import numpy as np
from numpy.typing import NDArray


def c_matrix_element(a: PartialWave, b: PartialWave, moment: Moment) -> float:
    # The all-zero-projection coefficient vanishes for odd total angular
    # momentum. Enforce the exact selection rule before numerical evaluation.
    if int(a.L.value + b.L.value + moment.L.value) % 2:
        return 0.0
    return float(
        np.sqrt(b.L.multiplicity / a.L.multiplicity)
        * ld.clebsch_gordan(j1=b.L, m1=0, j2=moment.L, m2=0, j=a.L, m=0)
        * ld.clebsch_gordan(j1=b.L, m1=b.M, j2=moment.L, m2=moment.M, j=a.L, m=a.M)
    )


def polarized_c_matrix_element(
    a: ReflectivityPartialWave,
    b: ReflectivityPartialWave,
    moment: LinearlyPolarizedMoment,
) -> complex:
    if a.r != b.r:
        return 0.0
    r = a.r.value
    a_bar = PartialWave(a.L, -a.M)
    b_bar = PartialWave(b.L, -b.M)
    a0 = a.partial_wave
    b0 = b.partial_wave

    if moment.variant == 0:
        return c_matrix_element(a0, b0, moment.moment) + (-1) ** (
            a.M.value + b.M.value
        ) * c_matrix_element(a_bar, b_bar, moment.moment)
    elif moment.variant == 1:
        # Mathieu et al., Phys. Rev. D 100, 054017, Eq. (13): H^1 is
        # extracted with +cos(2 Phi), hence the opposite sign to I^1.
        return r * (
            (-1) ** b.M.value * c_matrix_element(a0, b_bar, moment.moment)
            + (-1) ** a.M.value * c_matrix_element(a_bar, b0, moment.moment)
        )
    else:
        return (
            -1j
            * r
            * (
                (-1) ** b.M.value * c_matrix_element(a0, b_bar, moment.moment)
                - (-1) ** a.M.value * c_matrix_element(a_bar, b0, moment.moment)
            )
        )


def c_matrix(
    moment: Moment, waves: Sequence[PartialWave]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    waves = sorted(waves)
    C = np.array(
        [[c_matrix_element(a, b, moment) for b in waves] for a in waves],
        dtype=np.float64,
    )
    return (C + C.conj().T) / 2, (C - C.conj().T) / (2j)


def polarized_c_matrix(
    moment: LinearlyPolarizedMoment, waves: Sequence[ReflectivityPartialWave]
) -> NDArray[np.complex128]:
    waves = sorted(waves)
    C = np.array(
        [[polarized_c_matrix_element(a, b, moment) for b in waves] for a in waves],
        dtype=np.complex128,
    )
    if moment.variant == 2:
        return (C - C.conj().T) / (2j)
    return (C + C.conj().T) / 2
