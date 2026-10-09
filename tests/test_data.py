import pickle

import numpy as np
import pytest
from numpy.typing import ArrayLike

from momentous import (
    Covariance,
    Moment,
    MomentData,
    NormalizationWarning,
    Waveset,
    analyze,
)


def test_complex_covariance_labels_align_with_mapping_order() -> None:
    matrix = np.array(
        [
            [0.04, 0.01, 0.005, 0],
            [0.01, 0.09, 0, -0.01],
            [0.005, 0, 0.16, 0.02],
            [0, -0.01, 0.02, 0.25],
        ]
    )
    labels = [
        (1, 1, "real"),
        (1, 1, "imag"),
        (2, 0, "real"),
        (2, 0, "imag"),
        (0, 0, "real"),
        (0, 0, "imag"),
    ]
    covariance = Covariance(np.pad(matrix, (0, 2)), labels)
    result = analyze(
        MomentData({(1, 1): 0.1 + 0.2j, (0, 0): 1, (2, 0): 0.3}, covariance=covariance),
        Waveset.from_max_l(1),
    )
    assert result.component_labels == (
        (1, 1, "real"),
        (1, 1, "imag"),
        (0, 0, "real"),
        (0, 0, "imag"),
        (2, 0, "real"),
        (2, 0, "imag"),
    )
    np.testing.assert_allclose(result.values, [0.1, 0.2, 1, 0, 0.3, 0])
    np.testing.assert_allclose(
        result.covariance[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])],
        matrix,
        atol=1e-16,
    )


def test_raw_complex_normalization_propagates_denominator_cross_correlations() -> None:
    matrix = np.array(
        [[4, 0, 0.5, -0.25], [0, 0, 0, 0], [0.5, 0, 1, 0.2], [-0.25, 0, 0.2, 0.5]]
    )
    covariance = Covariance(
        matrix, [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")]
    )
    result = analyze(
        MomentData({(0, 0): 10, (1, 1): 2 + 3j}, covariance=covariance),
        Waveset([(0, 0)]),
    )
    jacobian = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-2 / (100), 0, 1 / (10), 0],
            [-3 / (100), 0, 0, 1 / (10)],
        ]
    )
    np.testing.assert_allclose(result.values, [1, 0, 2 / (10), 3 / (10)])
    np.testing.assert_allclose(
        result.covariance, jacobian @ matrix @ jacobian.T, atol=1e-17
    )
    np.testing.assert_array_equal(result.covariance[:2], np.zeros((2, 4)))


def test_normalizer_location_follows_mapping_order() -> None:
    result = analyze(
        MomentData(
            {(1, 0): 2, (0, 0): 10},
            covariance=Covariance.from_uncertainties({(0, 0): 1, (1, 0): 0}),
        ),
        Waveset([(0, 0)]),
    )
    assert result.values[0] == pytest.approx(2 / (10))
    assert result.values[2] == 1
    assert result.covariance[0, 0] == pytest.approx(4 / 10000)


def test_polarized_normalization_keeps_variant_correlations() -> None:
    covariance = Covariance([[4, 1], [1, 2]], [(0, 0, 0), (1, 0, 1)])
    result = analyze(
        MomentData({(0, 0, 0): 10, (1, 0, 1): 5}, covariance=covariance),
        Waveset.from_max_l(1, reflectivities=("+",)),
    )
    np.testing.assert_array_equal(result.values, [1, 0.5])
    assert result.covariance[1, 1] == pytest.approx(0.02)
    np.testing.assert_array_equal(result.covariance[0], [0, 0])


def test_unit_normalizer_uses_the_raw_wigner_d_convention() -> None:
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0.3}, covariance=Covariance.exact()),
        Waveset([(0, 0)]),
    )
    assert result.values[2] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "moments,message",
    [
        ({}, "must include"),
        ({(1, 0): 0}, "must include"),
        ({(0, 0): 1}, "besides"),
    ],
)
def test_analysis_requires_raw_normalization_and_nontrivial_data(
    moments: dict[tuple[int, int], complex],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze(MomentData(moments, covariance=Covariance.exact()), Waveset([(0, 0)]))


def test_independent_uncertainties_are_explicit_and_cover_every_moment() -> None:
    pool = Waveset([(0, 0)])
    moments = {(0, 0): 1, (1, 1): 0j}
    result = analyze(
        MomentData(
            moments,
            covariance=Covariance.from_uncertainties({(0, 0): 0, (1, 1): 0.1 + 0.2j}),
        ),
        pool,
    )
    np.testing.assert_allclose(result.covariance, np.diag([0, 0, 0.01, 0.04]))
    with pytest.raises(ValueError, match="missing"):
        analyze(
            MomentData(
                moments, covariance=Covariance.from_uncertainties({(1, 1): 0.1})
            ),
            pool,
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1]],
        [[1, 0.1], [0, 1]],
        [[1, 2], [2, 1]],
        [[-1e-30, 0], [0, 1]],
        [[np.nan, 0], [0, 1]],
        np.eye(2, dtype=complex),
        [[1e-30, 2e-30], [2e-30, 1e-30]],
    ],
    ids=[
        "shape",
        "asymmetry",
        "indefinite",
        "negative-variance",
        "nonfinite",
        "complex",
        "tiny-indefinite",
    ],
)
def test_invalid_real_component_covariance_is_rejected(matrix: ArrayLike) -> None:
    with pytest.raises(ValueError, match="Covariance"):
        Covariance(matrix, [(1, 0, "real"), (1, 0, "imag")])


@pytest.mark.parametrize(
    "components,message",
    [
        ([(1, 0, "real"), (1.0, 0, "real")], "unique"),
        ([(1, 0, "real"), (1, 0, 0)], "cannot mix"),
    ],
)
def test_ambiguous_covariance_labels_are_rejected(
    components: list[tuple[int | float, int, str | int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Covariance(np.eye(2), components)


def test_covariance_requires_every_real_and_imaginary_slot() -> None:
    covariance = Covariance(
        np.zeros((3, 3)), [(0, 0, "real"), (0, 0, "imag"), (1, 0, "real")]
    )
    with pytest.raises(ValueError, match="missing=.*imag"):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=covariance), Waveset([(0, 0)])
        )


def test_unlabeled_covariance_cannot_silently_depend_on_mapping_order() -> None:
    with pytest.raises(TypeError, match="label every row"):
        analyze(
            MomentData({(0, 0): 1, (1, 0): 0}, covariance=np.zeros((4, 4))),  # ty: ignore[invalid-argument-type]
            Waveset([(0, 0)]),
        )


@pytest.mark.parametrize("normalizer", [0, -1, 1 + 0.01j])
def test_raw_normalization_requires_a_positive_real_denominator(
    normalizer: complex,
) -> None:
    with pytest.raises(ValueError, match="positive and real"):
        analyze(
            MomentData({(0, 0): normalizer, (1, 0): 0}, covariance=Covariance.exact()),
            Waveset([(0, 0)]),
        )


def test_imaginary_normalizer_uncertainty_must_be_exact_zero() -> None:
    with pytest.raises(ValueError, match="imaginary normalization component"):
        analyze(
            MomentData(
                {(0, 0): 1, (1, 0): 0},
                covariance=Covariance.from_uncertainties({(0, 0): 0.1j, (1, 0): 0}),
            ),
            Waveset([(0, 0)]),
        )


def test_weak_normalizer_flags_the_first_order_approximation() -> None:
    with pytest.warns(NormalizationWarning, match="reaches zero"):
        result = analyze(
            MomentData(
                {(0, 0): 10, (1, 0): 1},
                covariance=Covariance.from_uncertainties({(0, 0): 5, (1, 0): 0}),
            ),
            Waveset([(0, 0)]),
        )
    assert np.isfinite(result.covariance).all()


@pytest.mark.parametrize("scale", [1e-200, 1e200], ids=["tiny-units", "large-units"])
def test_normalization_is_invariant_under_extreme_common_unit_rescaling(
    scale: float,
) -> None:
    pool = Waveset([(0, 0)])
    reference = analyze(
        MomentData(
            {(0, 0): 10, (1, 1): 2 + 3j},
            covariance=Covariance.from_uncertainties({(0, 0): 0.1, (1, 1): 0.2 + 0.3j}),
        ),
        pool,
    )
    covariance = Covariance.from_uncertainties(
        {(0, 0): 0.1 * scale, (1, 1): (0.2 + 0.3j) * scale}
    )
    covariance = pickle.loads(pickle.dumps(covariance))
    rescaled = analyze(
        MomentData(
            {(0, 0): 10 * scale, (1, 1): (2 + 3j) * scale}, covariance=covariance
        ),
        pool,
    )
    np.testing.assert_allclose(rescaled.values, reference.values, atol=1e-16)
    np.testing.assert_allclose(rescaled.covariance, reference.covariance, atol=1e-18)


def test_input_and_result_snapshots_cannot_invalidate_cached_decisions() -> None:
    values = {(0, 0, 0): 1.0, (0, 0, 1): 1.0}
    matrix = np.diag([0, 0.01])
    covariance = Covariance(matrix, [(0, 0, 0), (0, 0, 1)])
    result = analyze(
        MomentData(values, covariance=covariance),
        Waveset.from_max_l(0, reflectivities=("+", "-")),
    )
    values[(0, 0, 1)] = -100
    matrix[1, 1] = 100
    assert result.check(Waveset([(0, 0, "+")])).valid
    assert not result.check(Waveset([(0, 0, "-")])).valid
    with pytest.raises(ValueError):
        covariance.matrix.flags.writeable = True
    with pytest.raises(ValueError):
        result.covariance.flags.writeable = True
    restored = pickle.loads(pickle.dumps(covariance))
    np.testing.assert_array_equal(restored.matrix, covariance.matrix)
    with pytest.raises(ValueError):
        restored.matrix.flags.writeable = True


def test_reordering_raw_inputs_preserves_all_candidate_verdicts() -> None:
    pool = Waveset([(0, 0), (1, 1)])
    covariance = Covariance(
        [[4, 0, 0.5, -0.25], [0, 0, 0, 0], [0.5, 0, 1, 0.2], [-0.25, 0, 0.2, 0.5]],
        [(0, 0, "real"), (0, 0, "imag"), (1, 1, "real"), (1, 1, "imag")],
    )
    raw = analyze(
        MomentData({(0, 0): 10, (1, 1): 2 + 3j}, covariance=covariance), pool
    ).search()
    reordered = analyze(
        MomentData({(1, 1): 2 + 3j, (0, 0): 10}, covariance=covariance), pool
    ).search()
    order = [2, 3, 0, 1]
    np.testing.assert_allclose(reordered.analysis.values, raw.analysis.values[order])
    np.testing.assert_allclose(
        reordered.analysis.covariance,
        raw.analysis.covariance[np.ix_(order, order)],
        atol=1e-17,
    )
    for candidate in pool:
        assert raw.status(candidate) is reordered.status(candidate)


@pytest.mark.parametrize(
    "value,error,message",
    [
        (0j, 0, "must be real"),
        (np.nan, 0, "must be finite"),
        (0, -0.1, "nonnegative"),
    ],
    ids=["complex-polarized-value", "nonfinite-value", "negative-error"],
)
def test_invalid_measurements_have_meaningful_errors(
    value: complex, error: complex, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze(
            MomentData(
                {(0, 0, 0): 1, (1, 0, 0): value},
                covariance=Covariance.from_uncertainties(
                    {(0, 0, 0): 0, (1, 0, 0): error}
                ),
            ),
            Waveset.from_max_l(1, reflectivities=("+",)),
        )


def test_moment_objects_and_tuple_keys_cannot_duplicate_physical_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate physical moment"):
        analyze(
            MomentData(
                {(0, 0): 1, (1, 0): 0, Moment(1, 0): 0}, covariance=Covariance.exact()
            ),
            Waveset([(0, 0)]),
        )
