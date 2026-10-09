import pickle
from fractions import Fraction

import laddu as ld
import numpy as np
import pytest

from momentous import Covariance, Moment, MomentData, Waveset, analyze
from momentous.domain import MomentKey


@pytest.mark.parametrize("polarized", [False, True], ids=["complex", "polarized"])
def test_independent_and_matrix_covariance_give_the_same_normalized_analysis(
    polarized: bool,
) -> None:
    if polarized:
        moments = {(0, 0, 0): 10, (1, 1, 2): 2}
        errors = {(1, 1, 2): 0.3, (0, 0, 0): 0.1}
        variances = [0.09, 0.01]
    else:
        moments = {(0, 0): 10, (1, 1): 2 + 3j}
        errors = {(1, 1): 0.2 + 0.3j, (0, 0): 0.1}
        variances = [0.04, 0.09, 0.01, 0]
    independent = Covariance.from_uncertainties(errors)
    assert independent.components == tuple(
        component for key in errors for component in Moment(*key).components
    )
    np.testing.assert_allclose(independent.matrix, np.diag(variances))
    matrix = Covariance(np.diag(variances), components=independent.components)
    pool = Waveset.from_max_l(1, reflectivities=("+", "-") if polarized else None)
    first = analyze(MomentData(moments, covariance=independent), pool)
    second = analyze(MomentData(moments, covariance=matrix), pool)
    np.testing.assert_allclose(first.covariance, second.covariance, atol=1e-18)
    if not polarized:
        assert first.covariance[2, 3] > 0  # Shared noisy denominator couples Re/Im.
    assert first.check().status is second.check().status


def test_exact_covariance_is_reusable_across_polarization_and_component_orders() -> (
    None
):
    exact = pickle.loads(pickle.dumps(Covariance.exact()))
    assert exact.components is None and repr(exact) == "Covariance.exact()"
    for labels in (
        [(1, 1, "imag"), (0, 0, "real"), (1, 1, "real"), (0, 0, "imag")],
        [(1, 1, 2), (0, 0, 0)],
    ):
        matrix = exact.aligned(labels)
        np.testing.assert_array_equal(matrix, np.zeros((len(labels), len(labels))))
        with pytest.raises(ValueError):
            matrix.flags.writeable = True


def test_independent_covariance_owns_its_input_snapshot_after_serialization() -> None:
    errors = {(ld.L(1), ld.M(1)): 0.2 + 0.3j}
    covariance = Covariance.from_uncertainties(errors)
    errors[(ld.L(1), ld.M(1))] = 100
    restored = pickle.loads(pickle.dumps(covariance))
    assert restored.components == ((1, 1, "real"), (1, 1, "imag"))
    np.testing.assert_allclose(restored.matrix, np.diag([0.04, 0.09]))
    with pytest.raises(ValueError):
        restored.matrix.flags.writeable = True


@pytest.mark.parametrize(
    "errors,message",
    [
        ({(1, 1): 0.1, Moment(Fraction(1), 1): 0.2}, "Duplicate physical moment"),
        ({(1, 1): 0.1, (1, 1, 0): 0.2}, "cannot mix"),
        ({(1, 1): -0.1}, "nonnegative"),
        ({(1, 1): 0.1 - 0.2j}, "nonnegative"),
        ({(1, 1): np.inf}, "finite"),
        ({(1, 1, 0): 0j}, "must be real"),
    ],
    ids=[
        "duplicate",
        "mixed",
        "negative-real",
        "negative-imag",
        "nonfinite",
        "complex-polarized",
    ],
)
def test_independent_uncertainty_constructor_rejects_ambiguous_or_invalid_inputs(
    errors: dict[Moment | MomentKey, complex], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Covariance.from_uncertainties(errors)
