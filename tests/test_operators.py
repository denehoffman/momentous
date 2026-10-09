import numpy as np
import pytest
from scipy.special import sph_harm_y

from momentous import Covariance, Moment, MomentData, Waveset, analyze
from momentous.operators import moment_operators, normalization_metric


@pytest.mark.parametrize("L,M", [(1, 1), (2, 1), (2, 2), (3, 1), (4, 2)])
def test_unpolarized_operators_match_conjugated_harmonic_integrals(
    L: int, M: int
) -> None:
    pool = Waveset([(0, 0), (1, -1), (1, 0), (1, 1), (2, 1)])
    costheta, weights = np.polynomial.legendre.leggauss(12)
    phi = np.arange(24) * 2 * np.pi / 24
    theta, azimuth = np.meshgrid(np.arccos(costheta), phi, indexing="ij")
    basis = np.array(
        [sph_harm_y(w.L.value, int(w.M.value), theta, azimuth) for w in pool.waves]
    )
    feature = np.sqrt(4 * np.pi / (2 * L + 1)) * sph_harm_y(L, M, theta, azimuth).conj()
    integral = np.einsum(
        "t,atp,btp,tp->ab", weights * 2 * np.pi / 24, basis.conj(), basis, feature
    )
    real, imag = moment_operators(Moment(L, M), pool)
    np.testing.assert_allclose(real + 1j * imag, integral, atol=1e-14)


def test_exact_selection_rule_and_analytic_sp_bounds() -> None:
    pool = Waveset([(0, 0), (1, 0)])
    result = analyze(
        MomentData({(0, 0): 1, (1, 0): 0, (2, 0): 0}, covariance=Covariance.exact()),
        pool,
    )
    np.testing.assert_allclose(
        result.operators()[2], [[0, 1 / np.sqrt(3)], [1 / np.sqrt(3), 0]], atol=1e-15
    )
    bound = result.bounds()[(2, 0, "real")]
    assert (bound.lower, bound.upper) == pytest.approx((0, 0.4))


@pytest.mark.parametrize("sector,sign", [("+", 1), ("-", -1)])
def test_pure_s_polarized_variants_follow_mathieu_signs(sector: str, sign: int) -> None:
    pool = Waveset([(0, 0, sector)])  # ty: ignore[invalid-argument-type]
    result = analyze(
        MomentData(
            {(0, 0, 0): 1, (0, 0, 1): sign, (0, 0, 2): 0}, covariance=Covariance.exact()
        ),
        pool,
    )
    assert result.check().valid
    bounds = result.bounds()
    assert bounds[(0, 0, 0)].lower == bounds[(0, 0, 0)].upper == pytest.approx(1)
    assert bounds[(0, 0, 1)].lower == bounds[(0, 0, 1)].upper == pytest.approx(sign)
    assert bounds[(0, 0, 2)].lower == bounds[(0, 0, 2)].upper == 0


def test_polarized_variant_two_has_nonzero_analytic_bounds() -> None:
    pool = Waveset([(0, 0, "+"), (1, 1, "+")])
    result = analyze(
        MomentData({(0, 0, 0): 1, (1, 1, 2): 0}, covariance=Covariance.exact()), pool
    )
    bound = result.bounds()[(1, 1, 2)]
    assert (bound.lower, bound.upper) == pytest.approx(
        (-1 / (2 * np.sqrt(3)), 1 / (2 * np.sqrt(3)))
    )


def test_amplitude_generated_polarized_moments_pass_continuous_checks() -> None:
    pool = Waveset([(0, 0, "+"), (2, -1, "+"), (2, 1, "+"), (0, 0, "-"), (2, 0, "-")])
    amplitudes = np.array([1 + 0.2j, -0.3j, 0.5, 0.1 + 0.7j, -0.4])
    metric = normalization_metric(pool)
    assert (amplitudes.conj() @ metric @ amplitudes).real > 0
    measurements = {}
    for L, M in ((0, 0), (2, 0), (2, 1), (2, 2), (4, 1)):
        for variant in (0, 1, 2):
            moment = Moment(L, M, variant)
            measurements[moment] = float(
                (
                    amplitudes.conj() @ moment_operators(moment, pool)[0] @ amplitudes
                ).real
            )
    assert (
        analyze(
            MomentData(measurements, covariance=Covariance.exact()), pool, n_sigma=0
        )
        .check()
        .valid
    )
