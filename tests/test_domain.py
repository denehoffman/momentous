import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction

import laddu as ld
import numpy as np
import pytest

from momentous import Covariance, Moment, Observable, Wave, Waveset


@pytest.mark.parametrize(
    "rank,projection",
    [
        (2, -1),
        (2.0, -1.0),
        (Fraction(4, 2), Fraction(-2, 2)),
        (ld.L(2), ld.M(-1)),
    ],
    ids=["integers", "floats", "fractions", "laddu"],
)
def test_quantum_numbers_have_canonical_laddu_identity(
    rank: int | float | Fraction | ld.L, projection: int | float | Fraction | ld.M
) -> None:
    wave, moment = Wave(rank, projection), Moment(rank, projection)
    assert wave == Wave(2, -1)
    assert moment == Moment(2, -1)
    assert isinstance(wave.L, ld.L) and isinstance(moment.M, ld.M)
    assert hash(wave) == hash(Wave(2, -1))


@pytest.mark.parametrize(
    "rank,projection", [(1.5, 0), (1, 0.5), (1, 2), (-1, 0), (True, 0)]
)
def test_nonphysical_orbital_quantum_numbers_are_rejected(
    rank: float, projection: float
) -> None:
    with pytest.raises(ValueError):
        Wave(rank, projection)


def test_domain_objects_are_immutable() -> None:
    wave = Wave(1, 0)
    with pytest.raises(FrozenInstanceError):
        wave.L = ld.L(2)  # ty: ignore[invalid-assignment]


def test_domain_objects_can_cross_process_boundaries() -> None:
    pool = Waveset.from_max_l(1, reflectivities=("+", "-"))
    assert pickle.loads(pickle.dumps(pool)) == pool
    moment = Moment(2, 1, variant=2)
    assert pickle.loads(pickle.dumps(moment)) == moment


def test_generated_pools_select_signed_projections_and_reflectivity() -> None:
    unpolarized = Waveset.from_max_l(Fraction(1))
    both = Waveset.from_max_l(1, reflectivities=("+", "-"))
    positive = Waveset.from_max_l(1, reflectivities=(ld.Parity.POSITIVE,))
    assert [(wave.L.value, wave.M.value) for wave in unpolarized.waves] == [
        (0, 0),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    assert len(both) == 2 * len(positive) == 8
    assert positive.polarized and all(
        w.reflectivity == ld.Parity.POSITIVE for w in positive.waves
    )


def test_iteration_yields_nonempty_wavesets_and_explicit_powerset_can_include_empty() -> (
    None
):
    pool = Waveset([(1, 0), (0, 0)])
    assert pool.waves == (Wave(0, 0), Wave(1, 0))
    assert list(pool) == [Waveset([(0, 0)]), Waveset([(1, 0)]), pool]
    assert len(list(pool.powerset(min_size=0))) == 4
    assert list(pool.powerset(min_size=2, max_size=2)) == [pool]
    empty = Waveset([], polarized=True)
    assert list(empty) == [] and list(empty.powerset(min_size=0)) == [empty]


@pytest.mark.parametrize("definitions", [[(0, 0), (0.0, 0)], [(0, 0), (1, 0, "+")]])
def test_duplicate_or_mixed_wave_definitions_are_rejected(
    definitions: list[tuple],
) -> None:
    with pytest.raises(ValueError):
        Waveset(definitions)


@pytest.mark.parametrize(
    "value,label,constructor",
    [
        (Wave(0, 0), "S0", "Wave(0, 0)"),
        (Wave(2, 2, "+"), "D+2(+)", "Wave(2, 2, reflectivity='+')"),
        (Wave(2, -2, "-"), "D-2(-)", "Wave(2, -2, reflectivity='-')"),
        (Moment(2, -1), "H(2, -1)", "Moment(2, -1)"),
        (Moment(0, 0, 0), "H^0(0, 0)", "Moment(0, 0, variant=0)"),
        (Moment(2, 1, 1), "H^1(2, 1)", "Moment(2, 1, variant=1)"),
        (Moment(2, 1, 2), "Im H^2(2, 1)", "Moment(2, 1, variant=2)"),
        (
            Moment(2, 1).real,
            "Re H(2, 1)",
            "Observable(Moment(2, 1), part='real')",
        ),
        (
            Waveset([(1, 0), (0, 0)]),
            "{S0, P0}",
            "Waveset([(0, 0), (1, 0)])",
        ),
        (
            Waveset([(2, 2, "+"), (0, 0, "-")]),
            "{S0(-), D+2(+)}",
            "Waveset([(0, 0, '-'), (2, 2, '+')])",
        ),
        (Waveset([]), "{}", "Waveset([])"),
        (Waveset([], polarized=True), "{}", "Waveset([], polarized=True)"),
    ],
)
def test_scientific_strings_and_constructor_representations(
    value: Wave | Moment | Observable | Waveset, label: str, constructor: str
) -> None:
    assert str(value) == label
    assert repr(value) == constructor
    assert (
        eval(
            repr(value),
            {
                "__builtins__": {},
                "Wave": Wave,
                "Moment": Moment,
                "Observable": Observable,
                "Waveset": Waveset,
            },
        )
        == value
    )


def test_scalar_observables_preserve_physical_identity_and_label_covariance() -> None:
    h11 = Moment(Fraction(1), ld.M(1))
    assert h11.real == Observable.from_key((1.0, 1, "real"))
    assert h11.real != h11.imag
    assert hash(h11.imag) == hash(Observable.from_key((1, 1, "imag")))
    assert pickle.loads(pickle.dumps(h11.imag)) == h11.imag
    covariance = Covariance([[1, 0.5], [0.5, 2]], [h11.real, h11.imag])
    np.testing.assert_array_equal(
        covariance.aligned([h11.imag, h11.real]), [[2, 0.5], [0.5, 1]]
    )
    polarized = Moment(1, 1, variant=2)
    assert polarized.real == Observable(polarized)
    assert Covariance([[1]], [polarized]).components == ((1, 1, 2),)
    with pytest.raises(ValueError, match="scalar observables"):
        _ = polarized.imag
    with pytest.raises(ValueError, match="Select"):
        Observable(h11)
    with pytest.raises(ValueError, match="complex moment"):
        Covariance([[1]], [h11])


@pytest.mark.parametrize("sectors", [(), ("+", "+"), ("+", ld.Parity.POSITIVE)])
def test_sector_selection_cannot_be_empty_or_duplicate(sectors: tuple) -> None:
    with pytest.raises(ValueError):
        Waveset.from_max_l(1, reflectivities=sectors)
