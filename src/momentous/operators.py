"""Hermitian angular-moment operators in an explicitly ordered wave basis."""

import laddu as ld
import numpy as np

from momentous.domain import Moment, Wave, Waveset
from momentous.results import ComplexArray


def _element(a: Wave, b: Wave, moment: Moment) -> float:
    """Evaluate an unpolarized coefficient with exact odd-rank selection."""
    if (a.L.value + b.L.value + moment.L.value) % 2:
        return 0.0
    return float(
        np.sqrt(b.L.multiplicity / a.L.multiplicity)
        * ld.clebsch_gordan(j1=b.L, m1=0, j2=moment.L, m2=0, j=a.L, m=0)
        * ld.clebsch_gordan(j1=b.L, m1=b.M, j2=moment.L, m2=moment.M, j=a.L, m=a.M)
    )


def _polarized_element(a: Wave, b: Wave, moment: Moment) -> complex:
    """Evaluate reflectivity coefficients with Mathieu's positive H^1 sign."""
    if a.reflectivity != b.reflectivity:
        return 0j
    assert a.reflectivity is not None
    sign = a.reflectivity.value
    a_bar, b_bar = Wave(a.L, -a.M), Wave(b.L, -b.M)
    if moment.variant == 0:
        return complex(
            _element(a, b, moment)
            + (-1) ** (int(a.M.value) + int(b.M.value)) * _element(a_bar, b_bar, moment)
        )
    first = (-1) ** int(b.M.value) * _element(a, b_bar, moment)
    second = (-1) ** int(a.M.value) * _element(a_bar, b, moment)
    if moment.variant == 1:
        return complex(sign * (first + second))
    return -1j * sign * (first - second)


def moment_operators(moment: Moment, waves: Waveset) -> ComplexArray:
    """Construct operators in ``waves.waves`` order without implicit sorting.

    Parameters
    ----------
    moment : Moment
        Complex unpolarized or real polarized observable.
    waves : Waveset
        Nonempty wave basis with matching polarization.

    Returns
    -------
    ndarray
        Shape ``(2, n, n)`` for unpolarized real/imaginary parts, or
        ``(1, n, n)`` for the polarized observable. Polarized operators are
        raw: their metric is the ``H^0(0,0)`` operator.

    Raises
    ------
    ValueError
        If the wave basis is empty or has mismatched polarization.
    """
    if not len(waves) or waves.polarized != (moment.variant is not None):
        raise ValueError(
            "Moment and nonempty wave basis must have matching polarization"
        )
    basis = waves.waves
    if moment.variant is None:
        matrix = np.array(
            [[_element(b, a, moment) for b in basis] for a in basis],
            dtype=np.complex128,
        )
        return np.array(
            [(matrix + matrix.conj().T) / 2, (matrix - matrix.conj().T) / (2j)],
            dtype=np.complex128,
        )
    matrix = np.array(
        [[_polarized_element(a, b, moment) for b in basis] for a in basis],
        dtype=np.complex128,
    )
    operator = (
        (matrix - matrix.conj().T) / (2j)
        if moment.variant == 2
        else (matrix + matrix.conj().T) / 2
    )
    return np.asarray([operator], dtype=np.complex128)


def normalization_metric(waves: Waveset) -> ComplexArray:
    """Return the raw normalization metric in the supplied basis order."""
    if waves.polarized:
        return moment_operators(Moment(0, 0, variant=0), waves)[0]
    return np.eye(len(waves), dtype=np.complex128)
