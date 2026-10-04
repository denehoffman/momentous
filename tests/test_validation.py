import unittest
from typing import Literal

import numpy as np

from moments import (
    LinearlyPolarizedMoment,
    Measurement,
    Moment,
    PolarizedMeasurement,
    polarized_waveset,
    validate,
    waveset,
)
from moments.bounds import Bound, ComplexBound, get_polarized_bound
from moments.cg import c_matrix, polarized_c_matrix


class ValidationTests(unittest.TestCase):
    def test_legacy_uncertainties_and_sigma_threshold(self) -> None:
        waves = waveset([(0, 0)])
        measurement = Measurement(Moment(2, 0), 0.2 + 0.3j, 0.1 + 0.1j)
        result = validate(waves, [measurement], pairwise=False)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.failures), 2)
        self.assertEqual(result.invalid_measurements, (measurement,))
        self.assertIs(result.measurement, measurement)
        self.assertIsInstance(result.bounds, ComplexBound)
        np.testing.assert_allclose([f.sigma_distance for f in result.failures], [2, 3])
        self.assertTrue(validate(waves, [measurement], n_sigma=3, pairwise=False).valid)
        # Mixed Re/Im projections are stricter than the marginal error box.
        self.assertFalse(validate(waves, [measurement], n_sigma=3).valid)
        self.assertTrue(validate(waves, [measurement], n_sigma=4).valid)

    def test_collects_all_individual_failures_in_input_order(self) -> None:
        measurements = [
            Measurement(Moment(2, 0), 0.2),
            Measurement(Moment(0, 0), 1),
            Measurement(Moment(4, 0), -0.3),
        ]
        result = validate(waveset([(0, 0)]), measurements, pairwise=False)
        self.assertEqual(
            result.invalid_measurements, (measurements[0], measurements[2])
        )
        self.assertEqual(
            [f.measurement for f in result.failures], list(result.invalid_measurements)
        )
        self.assertTrue(all(np.isinf(f.sigma_distance) for f in result.failures))

    def test_covariance_supersedes_uncertainties_without_mutation(self) -> None:
        measurement = Measurement(Moment(2, 0), 0.2, 100)
        covariance = np.diag([0.01, 0.0])
        original = covariance.copy()
        covariance.flags.writeable = False
        result = validate(
            waveset([(0, 0)]), [measurement], covariance=covariance, pairwise=False
        )
        self.assertFalse(result.valid)
        self.assertAlmostEqual(result.failures[0].uncertainty, 0.1)
        self.assertTrue(
            validate(
                waveset([(0, 0)]), [measurement], covariance=covariance, n_sigma=2
            ).valid
        )
        np.testing.assert_array_equal(covariance, original)

    def test_unpolarized_pairwise_excludes_individually_allowed_values(self) -> None:
        # S + P_0: x=2/sqrt(3) Re(S* P), y=(2/5)|P|^2.
        waves = waveset([(0, 0), (1, 0)])
        measurements = [
            Measurement(Moment(1, 0), 0.55),
            Measurement(Moment(2, 0), 0.02),
        ]
        self.assertTrue(validate(waves, measurements, pairwise=False).valid)
        result = validate(waves, measurements, n_angles=36)
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_measurements, tuple(measurements))
        self.assertGreater(len(result.failures), 1)
        self.assertTrue(all(f.measurement is None for f in result.failures))
        self.assertTrue(all(isinstance(f.bounds, Bound) for f in result.failures))
        self.assertTrue(
            all(f.coefficients[1] == f.coefficients[3] == 0 for f in result.failures)
        )

    def test_mixed_real_imaginary_components_of_one_moment(self) -> None:
        # For S + P_1, H(1,1) occupies a disk of radius 1/(2 sqrt(3)).
        waves = waveset([(0, 0), (1, 1)])
        measurements = [Measurement(Moment(1, 1), 0.25 + 0.25j)]
        self.assertTrue(validate(waves, measurements, pairwise=False).valid)
        result = validate(waves, measurements, n_angles=36)
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_measurements, tuple(measurements))
        self.assertTrue(
            all(f.measurements == tuple(measurements) for f in result.failures)
        )

    def test_unpolarized_covariance_correlations_change_pairwise_verdict(self) -> None:
        waves = waveset([(0, 0), (1, 1)])
        measurements = [Measurement(Moment(1, 1), 0.25 + 0.25j)]
        positive = np.array([[0.01, 0.009], [0.009, 0.01]])
        negative = np.array([[0.01, -0.009], [-0.009, 0.01]])
        self.assertTrue(validate(waves, measurements, covariance=positive).valid)
        result = validate(waves, measurements, covariance=negative)
        self.assertFalse(result.valid)
        values = np.array([0.25, 0.25])
        for failure in result.failures:
            coefficients = np.array(failure.coefficients)
            self.assertAlmostEqual(failure.value, coefficients @ values)
            self.assertAlmostEqual(
                failure.uncertainty, np.sqrt(coefficients @ negative @ coefficients)
            )
            self.assertGreater(failure.sigma_distance, 1)

    def test_collects_pairwise_failures_even_after_individual_failures(self) -> None:
        measurements = [Measurement(Moment(2, 0), 0.2), Measurement(Moment(4, 0), 0.3)]
        result = validate(waveset([(0, 0)]), measurements, n_angles=8)
        self.assertEqual(result.invalid_measurements, tuple(measurements))
        self.assertEqual(
            len([f for f in result.failures if f.measurement is not None]), 2
        )
        self.assertTrue(any(len(f.measurements) == 2 for f in result.failures))

    def test_exact_analytical_amplitudes_and_normalization_pass(self) -> None:
        for population in (0, 0.05, 0.5, 0.95, 1):
            for phase in (0, np.pi / 3, np.pi):
                with self.subTest(population=population, phase=phase):
                    x = (
                        2
                        / np.sqrt(3)
                        * np.sqrt(population * (1 - population))
                        * np.cos(phase)
                    )
                    measurements = [
                        Measurement(Moment(0, 0), 1),
                        Measurement(Moment(1, 0), x),
                        Measurement(Moment(2, 0), 0.4 * population),
                    ]
                    self.assertTrue(
                        validate(
                            waveset([(0, 0), (1, 0)]),
                            measurements,
                            n_sigma=0,
                            n_angles=36,
                        ).valid
                    )

    def test_polarized_cross_variant_checks_and_singular_covariance(self) -> None:
        waves = polarized_waveset([(0, 0, "+"), (1, 0, "+")])
        measurements = [
            PolarizedMeasurement(LinearlyPolarizedMoment(1, 0, v), value)
            for v, value in ((0, 0.4), (1, -0.4))
        ]
        self.assertTrue(validate(waves, measurements, pairwise=False).valid)
        result = validate(
            waves, measurements, covariance=[[0.25, 0.25], [0.25, 0.25]], n_angles=36
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_measurements, tuple(measurements))
        self.assertTrue(
            all(f.measurements == tuple(measurements) for f in result.failures)
        )
        self.assertTrue(
            validate(
                waves, measurements, covariance=[[0.25, -0.25], [-0.25, 0.25]]
            ).valid
        )
        self.assertTrue(
            validate(
                waves, measurements, covariance=np.diag([0.25, 0.25]), n_sigma=2
            ).valid
        )

    def test_polarized_all_variants_and_reflectivities(self) -> None:
        # Mathieu Eq. (E1b): pure S has H^1(0,0)/H^0(0,0)=r.
        cases: tuple[tuple[Literal["+", "-"], int], ...] = (("+", 1), ("-", -1))
        variants: tuple[Literal[0, 1, 2], ...] = (0, 1, 2)
        for reflectivity, sign in cases:
            waves = polarized_waveset([(0, 0, reflectivity)])
            measurements = [
                PolarizedMeasurement(LinearlyPolarizedMoment(0, 0, v), value)
                for v, value in zip(variants, (1, sign, 0), strict=True)
            ]
            self.assertTrue(
                validate(waves, measurements, covariance=np.zeros((3, 3))).valid
            )
            measurements[-1].value = 0.1
            result = validate(waves, measurements, pairwise=False)
            self.assertEqual(result.invalid_measurements, (measurements[-1],))
            self.assertIsInstance(result.bounds, Bound)

    def test_polarized_variant_two_has_nonzero_bounds(self) -> None:
        waves = polarized_waveset([(0, 0, "+"), (1, 1, "+")])
        moment = LinearlyPolarizedMoment(1, 1, 2)
        bound = get_polarized_bound(moment, waves)
        self.assertAlmostEqual(bound.lower, -1 / (2 * np.sqrt(3)))
        self.assertAlmostEqual(bound.upper, 1 / (2 * np.sqrt(3)))
        self.assertTrue(
            validate(waves, [PolarizedMeasurement(moment, bound.upper)]).valid
        )

    def test_amplitude_generated_polarized_moments_pass(self) -> None:
        waves = polarized_waveset(
            [(0, 0, "+"), (2, -1, "+"), (2, 1, "+"), (0, 0, "-"), (2, 0, "-")]
        )
        # Operators sort the waves, so amplitudes use that same order.
        amplitudes = np.array([1 + 0.2j, -0.3j, 0.5, 0.1 + 0.7j, -0.4])
        normalization = polarized_c_matrix(LinearlyPolarizedMoment(0, 0, 0), waves)
        total = (amplitudes.conj() @ normalization @ amplitudes).real
        measurements = []
        for L, M in ((0, 0), (2, 0), (2, 1), (2, 2), (4, 1)):
            for variant in (0, 1, 2):
                moment = LinearlyPolarizedMoment(L, M, variant)
                value = (
                    amplitudes.conj() @ polarized_c_matrix(moment, waves) @ amplitudes
                ).real / total
                measurements.append(PolarizedMeasurement(moment, value))
        self.assertTrue(validate(waves, measurements, n_angles=24, n_sigma=0).valid)

    def test_wave_and_measurement_reordering_preserves_verdict(self) -> None:
        waves = waveset([(0, 0), (1, 0)])
        measurements = [
            Measurement(Moment(1, 0), 0.55),
            Measurement(Moment(2, 0), 0.02),
        ]
        covariance = np.diag([0.001, 0, 0.002, 0])
        covariance[0, 2] = covariance[2, 0] = 0.0005
        result = validate(waves, measurements, covariance=covariance)
        permutation = [2, 3, 0, 1]
        reordered = validate(
            waves[::-1],
            measurements[::-1],
            covariance=covariance[np.ix_(permutation, permutation)],
        )
        self.assertEqual(result.valid, reordered.valid)
        self.assertEqual(reordered.invalid_measurements, tuple(measurements[::-1]))

    def test_invalid_covariance_is_rejected(self) -> None:
        measurements = [Measurement(Moment(2, 0), 0)]
        invalid = [
            [[1]],
            [[1, 0.1], [0, 1]],
            [[1, 2], [2, 1]],
            [[-1e-30, 0], [0, 1]],
            [[np.nan, 0], [0, 1]],
            [[np.inf, 0], [0, 1]],
            np.eye(2, dtype=complex),
            [[1e-30, 2e-30], [2e-30, 1e-30]],
        ]
        for covariance in invalid:
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                validate(waveset([(0, 0)]), measurements, covariance=covariance)

    def test_roundoff_in_singular_covariance_is_tolerated(self) -> None:
        covariance = np.array([[1, 1 + 1e-15], [1 + 1e-15, 1]])
        self.assertTrue(
            validate(
                waveset([(0, 0)]), [Measurement(Moment(2, 0), 0)], covariance=covariance
            ).valid
        )

    def test_invalid_parameters_and_data(self) -> None:
        waves = waveset([(0, 0)])
        for n_sigma in (-1, np.nan, np.inf):
            with self.subTest(n_sigma=n_sigma), self.assertRaises(ValueError):
                validate(waves, [], n_sigma=n_sigma)
        for n_angles in (0, 3, 4.5, True):
            with self.subTest(n_angles=n_angles), self.assertRaises(ValueError):
                # Intentionally pass a non-integer to test runtime rejection.
                validate(waves, [], n_angles=n_angles)  # ty: ignore[no-matching-overload]
        for value, uncertainty in (
            (np.nan, None),
            (np.inf, None),
            (0, -1),
            (0, -1j),
            (0, np.nan),
        ):
            with (
                self.subTest(value=value, uncertainty=uncertainty),
                self.assertRaises(ValueError),
            ):
                validate(waves, [Measurement(Moment(2, 0), value, uncertainty)])
        with self.assertRaises(ValueError):
            validate([], [])
        with self.assertRaises(ValueError):
            validate(waveset([(0, 0), (0, 0)]), [])
        with self.assertRaises(ValueError):
            # Deliberately mixed polarization to exercise runtime validation.
            validate(waves + polarized_waveset([(0, 0, "+")]), [])  # ty: ignore[no-matching-overload]
        with self.assertRaises(ValueError):
            # Deliberately mismatched measurement and wave types.
            validate(waves, [PolarizedMeasurement(LinearlyPolarizedMoment(0, 0, 0), 1)])  # ty: ignore[no-matching-overload]
        with self.assertRaises(ValueError):
            validate(polarized_waveset([(0, 0, "+")]), [Measurement(Moment(0, 0), 1)])  # ty: ignore[no-matching-overload]
        with self.assertRaises(ValueError):
            validate(
                polarized_waveset([(0, 0, "+")]),
                # Deliberately complex value for a real observable.
                [PolarizedMeasurement(LinearlyPolarizedMoment(0, 0, 0), 1j)],  # ty: ignore[invalid-argument-type]
            )

    def test_empty_measurements_pass(self) -> None:
        for waves in (waveset([(0, 0)]), polarized_waveset([(0, 0, "+")])):
            result = validate(waves, [], covariance=np.zeros((0, 0)))
            self.assertTrue(result.valid)
            self.assertEqual(result.failures, ())
            self.assertEqual(result.invalid_measurements, ())

    def test_exact_selection_rule_and_polarized_moment_ordering(self) -> None:
        real, imag = c_matrix(Moment(1, 0), waveset([(0, 0), (1, 0)]))
        np.testing.assert_allclose(real, [[0, 1 / np.sqrt(3)], [1 / np.sqrt(3), 0]])
        np.testing.assert_allclose(imag, 0, atol=1e-15)
        moments = [LinearlyPolarizedMoment(2, 1, v) for v in (2, 0, 1)]
        self.assertEqual([moment.variant for moment in sorted(moments)], [0, 1, 2])
        with self.assertRaises(ValueError):
            # Deliberately invalid variant.
            LinearlyPolarizedMoment(2, 1, 3)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    unittest.main()
