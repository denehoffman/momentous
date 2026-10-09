import pickle
from fractions import Fraction

import laddu as ld
import numpy as np
import pytest
from numpy.typing import ArrayLike

from momentous import (
    Acceptance,
    EventGrouping,
    EventSample,
    ExtractionError,
    MCIntegration,
    MCStatistics,
    MomentBasis,
    Polarization,
    Waveset,
    analyze,
)
from momentous.results import RealArray


@pytest.fixture
def angular_grid() -> tuple[RealArray, RealArray, RealArray]:
    z, weight = np.polynomial.legendre.leggauss(8)
    azimuth = np.arange(16) * 2 * np.pi / 16
    theta, phi = np.meshgrid(np.arccos(z), azimuth, indexing="ij")
    return theta.ravel(), phi.ravel(), np.repeat(weight / 32, 16)


def angular_sample(
    grid: tuple[RealArray, RealArray, RealArray],
    weights: ArrayLike | None = None,
) -> EventSample:
    theta, phi, quadrature = grid
    return EventSample(
        costheta=np.cos(theta),
        phi=phi,
        weights=quadrature if weights is None else weights,
        events=EventGrouping.independent_rows(),
    )


def response(
    generated: EventSample,
    accepted: EventSample,
    basis: MomentBasis,
    statistics: MCStatistics | None = None,
) -> Acceptance:
    return Acceptance(
        generated=generated,
        accepted=accepted,
        basis=basis,
        integration=MCIntegration.uniform(),
        statistics=MCStatistics.data_only() if statistics is None else statistics,
    )


@pytest.mark.parametrize("rank", [1, 1.0, Fraction(1), ld.L(1)])
def test_basis_uses_laddu_quantum_numbers_and_complete_independent_coordinates(
    rank: int | float | Fraction | ld.L,
) -> None:
    basis = MomentBasis(rank)
    assert basis.max_L == ld.L(1)
    assert tuple(observable.key for observable in basis.observables) == (
        (0, 0, "real"),
        (1, 0, "real"),
        (1, 1, "real"),
        (1, 1, "imag"),
    )
    polarized = MomentBasis.from_waves(Waveset([(1, -1, "+")]))
    assert polarized.polarized and polarized.max_L == ld.L(2)
    assert all(observable.moment.M.value >= 0 for observable in polarized.observables)
    assert all(
        observable.moment.M.value != 0
        for observable in polarized.observables
        if observable.moment.variant == 2
    )


@pytest.mark.parametrize(
    "polarized", [False, True], ids=["unpolarized", "linear-photon"]
)
def test_acceptance_defaults_preserve_linked_mc_and_infer_basis_mode(
    angular_grid: tuple[RealArray, RealArray, RealArray], polarized: bool
) -> None:
    theta, phi, weights = (np.repeat(column, 8) for column in angular_grid)
    weights = weights / 8
    beam = (
        Polarization(
            magnitude=0.4, angle=np.tile(np.arange(8) * np.pi / 8, len(theta) // 8)
        )
        if polarized
        else None
    )
    events = EventGrouping.from_ids(np.arange(len(theta)))
    generated = EventSample(
        costheta=np.cos(theta),
        phi=phi,
        weights=weights,
        events=events,
        polarization=beam,
    )
    accepted = EventSample(
        costheta=np.cos(theta),
        phi=phi,
        weights=weights * 0.5,
        events=events,
        polarization=beam,
    )
    default = Acceptance(generated, accepted)
    explicit = Acceptance(
        generated,
        accepted,
        basis=MomentBasis(4, polarized=polarized),
        integration=MCIntegration.uniform(),
        statistics=MCStatistics.linked(),
    )
    assert default.generated is generated and default.accepted is accepted
    assert default.basis == explicit.basis
    assert default.basis.max_L == ld.L(4) and default.basis.polarized == polarized
    assert default.statistics.mode == "linked"
    if not polarized:
        np.testing.assert_allclose(
            default.response, 0.5 * np.eye(len(default.basis)), atol=1e-13
        )
    np.testing.assert_array_equal(default.response, explicit.response)
    extraction = default.extract(generated)
    normalizer = (0, 0, 0) if polarized else (0, 0)
    assert extraction.data.moments[normalizer] == pytest.approx(2.0)
    assert "linked" in extraction.covariance_scope


def test_mc_relationship_is_declared_by_acceptance() -> None:
    generated = EventSample(
        costheta=np.cos([0.2, 0.8]), phi=[0, 1], events=EventGrouping.from_ids([7, 9])
    )
    accepted = EventSample(
        costheta=np.cos([0.4, 1.0]), phi=[0, 1], events=EventGrouping.from_ids([7, 9])
    )
    with pytest.raises(ValueError, match="truth coordinates"):
        Acceptance(generated, accepted, basis=MomentBasis(0))
    independent = Acceptance(
        generated, accepted, basis=MomentBasis(0), statistics=MCStatistics.independent()
    )
    assert independent.extract(generated).mc_covariance.matrix[0, 0] > 0


@pytest.mark.parametrize("efficiency", [1.0, 0.4])
def test_full_acceptance_and_constant_efficiency_preserve_absolute_yield(
    angular_grid: tuple[RealArray, RealArray, RealArray],
    efficiency: float,
) -> None:
    theta, phi, weights = angular_grid
    z = np.cos(theta)
    density = 100 * (1 + 0.3 * z)
    generated = angular_sample(angular_grid)
    acceptance = response(
        generated,
        angular_sample(angular_grid, weights * efficiency),
        MomentBasis(2),
    )
    np.testing.assert_allclose(acceptance.response, efficiency * np.eye(9), atol=1e-14)
    result = acceptance.extract(
        angular_sample(angular_grid, weights * density * efficiency)
    )
    assert result.data.moments[(0, 0)] == pytest.approx(100)
    assert result.data.moments[(1, 0)] == pytest.approx(10)
    assert result.data.moments[(1, 1)] == pytest.approx(0, abs=1e-13)
    assert result.data.moments[(2, 0)] == pytest.approx(0, abs=1e-13)
    assert "conditional" in result.covariance_scope
    np.testing.assert_array_equal(
        result.mc_covariance.matrix, np.zeros_like(result.mc_covariance.matrix)
    )


@pytest.mark.parametrize("polarized", [False, True], ids=["unpolarized", "polarized"])
def test_laddu_extraction_matches_reordered_combos_to_exact_parent_ids(
    angular_grid: tuple[RealArray, RealArray, RealArray], polarized: bool
) -> None:
    theta, phi, weights = angular_grid
    angles = np.arange(8) * np.pi / 8
    theta, phi, weights = (np.repeat(column, 8) for column in (theta, phi, weights))
    weights = weights / 8
    orientation = np.tile(angles, len(theta) // 8)
    columns = {"event": np.arange(len(theta), dtype=np.uint64) + np.uint64(2**63)}
    scalars = {
        "costheta": np.cos(theta),
        "phi": phi,
        "P": np.full(len(theta), 0.4),
        "Phi": orientation,
    }
    events = EventGrouping.from_columns("event")
    beam = (
        Polarization(magnitude=ld.scalar("P"), angle=ld.scalar("Phi"))
        if polarized
        else None
    )
    generated = EventSample(
        ld.Dataset.from_arrays(
            p4s={}, scalars=scalars, columns=columns, weights=weights
        ),
        costheta=ld.scalar("costheta"),
        phi=ld.scalar("phi"),
        events=events,
        polarization=beam,
    )
    accepted = EventSample(
        ld.Dataset.from_arrays(
            p4s={},
            scalars={
                name: np.repeat(value[::-1], 2) for name, value in scalars.items()
            },
            columns={"event": np.repeat(columns["event"][::-1], 2)},
            weights=np.repeat(weights[::-1], 2) * np.tile([0.2, 0.3], len(weights)),
        ),
        events=events,
        polarization=beam,
    )
    density = 100 * (1 + 0.4 * np.cos(2 * orientation)) if polarized else 100
    data = EventSample(
        ld.Dataset.from_arrays(
            p4s={}, scalars=scalars, columns=columns, weights=0.5 * weights * density
        ),
        costheta=ld.scalar("costheta"),
        phi=ld.scalar("phi"),
        events=events,
        polarization=beam,
    )
    basis = MomentBasis(1, polarized=polarized)
    result = response(generated, accepted, basis, MCStatistics.linked()).extract(data)
    normalizer = (0, 0, 0) if polarized else (0, 0)
    assert result.data.moments[normalizer] == pytest.approx(100)
    assert accepted.n_events == generated.n_events
    assert result.statistics.mode == "linked"
    if polarized:
        assert result.data.moments[(0, 0, 1)] == pytest.approx(100)
    else:
        assert result.data.moments[(1, 1)] == pytest.approx(0, abs=1e-12)


def test_nonuniform_acceptance_unmixes_complex_moments_and_restores_signed_correlations(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    theta, phi, weights = angular_grid
    z, x, y = np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    efficiency = 0.7 + 0.15 * z + 0.1 * x
    density = 100 * (1 + 0.25 * z - 0.1 * x + 0.12 * y)
    acceptance = response(
        angular_sample(angular_grid),
        angular_sample(angular_grid, weights * efficiency),
        MomentBasis(1),
    )
    result = acceptance.extract(
        angular_sample(angular_grid, weights * density * efficiency)
    )
    assert abs(acceptance.response[0, 1]) > 0.1
    assert result.data.moments[(0, 0)] == pytest.approx(100)
    assert result.data.moments[(1, 0)] == pytest.approx(100 * 0.25 / 3)
    assert result.data.moments[(1, 1)] == pytest.approx(
        100 * (0.1 + 0.12j) / (3 * np.sqrt(2))
    )
    assert result.data.moments[(1, -1)] == pytest.approx(
        -result.data.moments[(1, 1)].conjugate()
    )
    labels = result.data.component_labels
    cov = result.data.covariance.matrix
    positive = labels.index((1, 1, "real"))
    negative = labels.index((1, -1, "real"))
    np.testing.assert_array_equal(cov[negative], -cov[positive])
    np.testing.assert_array_equal(cov[labels.index((1, 0, "imag"))], 0)
    analysis = analyze(result.data, Waveset.from_max_l(1), max_combination_size=1)
    assert analysis.moments[(1, 0)] == pytest.approx(0.25 / 3)


def test_known_generation_density_is_applied_to_both_mc_samples(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    theta, _, weights = angular_grid
    q = 1 + 0.3 * np.cos(theta)
    efficiency = 0.5 + 0.2 * np.cos(theta)
    acceptance = Acceptance(
        generated=angular_sample(angular_grid, weights * q),
        accepted=angular_sample(angular_grid, weights * q * efficiency),
        basis=MomentBasis(1),
        integration=MCIntegration.importance(
            relative_density=1 + 0.3 * ld.scalar("costheta")
        ),
        statistics=MCStatistics.data_only(),
    )
    result = acceptance.extract(angular_sample(angular_grid, weights * efficiency * 80))
    assert result.data.moments[(0, 0)] == pytest.approx(80)
    assert result.data.moments[(1, 0)] == pytest.approx(0, abs=1e-13)


def test_polar_cosine_endpoints_produce_exact_polar_moments(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    generated = angular_sample(angular_grid)
    acceptance = response(generated, generated, MomentBasis(1))
    data = EventSample(
        costheta=[-1, 1],
        phi=[0.7, -1.2],
        weights=[1, 3],
        events=EventGrouping.independent_rows(),
    )
    result = acceptance.extract(data)
    assert result.data.moments[(0, 0)] == pytest.approx(4)
    assert result.data.moments[(1, 0)] == pytest.approx(2)
    assert result.data.moments[(1, 1)] == pytest.approx(0, abs=1e-14)


def test_grouped_data_covariance_includes_hypothesis_cross_terms_without_rescaling(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    generated = angular_sample(angular_grid)
    acceptance = response(generated, generated, MomentBasis(1))
    theta, phi, weights = (
        np.array([0.4, 1.1, 2.0]),
        np.array([0.2, 1.3, -0.8]),
        np.array([0.2, 0.6, 1.2]),
    )
    sample = EventSample(
        costheta=np.cos(theta),
        phi=phi,
        weights=weights,
        events=EventGrouping.from_ids([7, 7, 8]),
    )
    result = acceptance.extract(sample)
    f = np.column_stack(
        (
            np.ones(3),
            np.cos(theta),
            -np.sin(theta) * np.cos(phi) / np.sqrt(2),
            np.sin(theta) * np.sin(phi) / np.sqrt(2),
        )
    )
    contributions = weights[:, None] * f
    groups = np.vstack((contributions[0] + contributions[1], contributions[2]))
    expected = groups.T @ groups
    labels = [observable.key for observable in acceptance.basis.observables]
    indices = [result.data.component_labels.index(label) for label in labels]
    np.testing.assert_allclose(
        result.data_covariance.matrix[np.ix_(indices, indices)], expected, atol=1e-14
    )
    assert result.data.moments[(0, 0)] == pytest.approx(2)
    independent = acceptance.extract(
        EventSample(
            costheta=np.cos(theta),
            phi=phi,
            weights=weights,
            events=EventGrouping.independent_rows(),
        )
    )
    assert not np.allclose(
        independent.data_covariance.matrix, result.data_covariance.matrix
    )


def test_polarized_response_handles_nonuniform_orientations_and_zero_polarization(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    theta, phi, weights = angular_grid
    orientations, magnitudes = (
        np.array([0, 0.3, 0.9, 1.7]),
        np.array([0, 0.3, 0.7, 0.5]),
    )
    theta, phi, weights = np.tile(theta, 4), np.tile(phi, 4), np.tile(weights / 4, 4)
    P, Phi = (
        np.repeat(magnitudes, len(theta) // 4),
        np.repeat(orientations, len(theta) // 4),
    )
    r0, r1, t1 = (
        np.cos(theta),
        -np.sin(theta) * np.cos(phi) / np.sqrt(2),
        np.sin(theta) * np.sin(phi) / np.sqrt(2),
    )
    cos, sin = P * np.cos(2 * Phi), P * np.sin(2 * Phi)
    f = np.column_stack(
        (np.ones(len(theta)), cos, r0, cos * r0, r1, cos * r1, sin * t1)
    )
    H = np.array([100, 25, 4, -3, 2, 1, -1.5])
    density = f @ (np.array([1, 1, 3, 3, 6, 6, 6]) * H)
    efficiency = 0.65 + 0.1 * np.cos(theta) + 0.07 * cos
    beam = Polarization(magnitude=P, angle=Phi)

    def sample(w: RealArray) -> EventSample:
        return EventSample(
            costheta=np.cos(theta),
            phi=phi,
            weights=w,
            events=EventGrouping.independent_rows(),
            polarization=beam,
        )

    acceptance = response(
        sample(weights), sample(weights * efficiency), MomentBasis(1, polarized=True)
    )
    result = acceptance.extract(sample(weights * efficiency * density))
    fitted = [
        result.data.moments[observable.moment.key]
        for observable in acceptance.basis.observables
    ]
    np.testing.assert_allclose(fitted, H, atol=2e-13)
    assert result.data.moments[(1, 1, 2)] == pytest.approx(-1.5)
    assert result.data.moments[(1, -1, 2)] == pytest.approx(-1.5)
    assert result.data.moments[(0, 0, 2)] == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_pure_s_polarization_has_the_existing_reflectivity_sign(sign: int) -> None:
    Phi = np.arange(24) * np.pi / 24
    beam = Polarization(magnitude=0.5, angle=Phi)

    def sample(weights: ArrayLike) -> EventSample:
        return EventSample(
            costheta=np.cos(np.full(24, 1.0)),
            phi=np.zeros(24),
            weights=weights,
            events=EventGrouping.independent_rows(),
            polarization=beam,
        )

    generated = sample(np.ones(24))
    result = response(generated, generated, MomentBasis(0, polarized=True)).extract(
        sample(1 + sign * 0.5 * np.cos(2 * Phi))
    )
    analysis = analyze(
        result.data, Waveset([(0, 0, "+" if sign == 1 else "-")]), n_sigma=0
    )
    assert analysis.moments[(0, 0, 1)] == pytest.approx(sign)
    assert analysis.check().valid


@pytest.mark.parametrize(
    "policy",
    [MCStatistics.linked(), MCStatistics.independent()],
    ids=["linked-fixed-size", "independent-poisson"],
)
def test_finite_mc_covariance_agrees_with_event_level_resampling(
    policy: MCStatistics,
) -> None:
    rng = np.random.default_rng(814)
    size = 1600
    theta, phi = np.arccos(rng.uniform(-1, 1, size)), rng.uniform(-np.pi, np.pi, size)
    z = np.cos(theta)
    accepted_weight = np.where(rng.random(size) < 0.65 + 0.15 * z, 0.9, 0)
    accepted_indices = np.flatnonzero(accepted_weight)
    generated = EventSample(
        costheta=np.cos(theta), phi=phi, events=EventGrouping.from_ids(range(size))
    )
    # Two hypotheses per selected event with total weight 0.9, not one.
    accepted = EventSample(
        costheta=np.cos(np.repeat(theta[accepted_indices], 2)),
        phi=np.repeat(phi[accepted_indices], 2),
        weights=np.tile([0.35, 0.55], len(accepted_indices)),
        events=EventGrouping.from_ids(np.repeat(accepted_indices, 2)),
    )
    acceptance = response(generated, accepted, MomentBasis(1), policy)
    data = EventSample(
        costheta=np.cos([0.4, 1.0, 2.0]),
        phi=[0.2, 1.5, -1.2],
        weights=[20, 30, 40],
        events=EventGrouping.independent_rows(),
    )
    result = acceptance.extract(data)
    f = np.column_stack(
        (
            np.ones(size),
            z,
            -np.sin(theta) * np.cos(phi) / np.sqrt(2),
            np.sin(theta) * np.sin(phi) / np.sqrt(2),
        )
    )
    matrices = (
        np.einsum("ni,nj->nij", f, f * [1, 3, 6, 6]) * accepted_weight[:, None, None]
    )
    draws = []
    for _ in range(1000):
        if policy.mode == "linked":
            counts = rng.multinomial(size, np.full(size, 1 / size))
            denominator = size
        else:
            counts = rng.poisson(1, size)
            denominator = rng.poisson(size)
        R = np.einsum("n,nij->ij", counts, matrices) / denominator
        draws.append(np.linalg.solve(R, result.measured))
    bootstrap = np.cov(np.asarray(draws), rowvar=False)
    indices = [
        result.data.component_labels.index(observable.key)
        for observable in acceptance.basis.observables
    ]
    predicted = result.mc_covariance.matrix[np.ix_(indices, indices)]
    np.testing.assert_allclose(np.diag(predicted), np.diag(bootstrap), rtol=0.15)
    np.testing.assert_allclose(
        result.data.covariance.matrix,
        result.data_covariance.matrix + result.mc_covariance.matrix,
        atol=1e-12,
    )


def test_yield_only_extraction_is_valid_but_has_no_nontrivial_compatibility_check() -> (
    None
):
    generated = EventSample(
        costheta=np.cos([0.5, 1.0]), phi=[0, 1], events=EventGrouping.independent_rows()
    )
    result = response(generated, generated, MomentBasis(0)).extract(generated)
    assert result.data.moments[(0, 0)] == 2
    with pytest.raises(ValueError, match="besides"):
        analyze(result.data, Waveset([(0, 0)]))


@pytest.mark.parametrize(
    "magnitude", [0.0, 1e-6], ids=["singular", "poorly-conditioned"]
)
def test_unidentifiable_response_reports_spectrum_without_silent_regularization(
    magnitude: float,
) -> None:
    beam = Polarization(magnitude=magnitude, angle=[0, 1])
    sample = EventSample(
        costheta=np.cos([0.5, 1.0]),
        phi=[0, 1],
        events=EventGrouping.independent_rows(),
        polarization=beam,
    )
    with pytest.raises(ExtractionError, match="singular_values") as error:
        response(sample, sample, MomentBasis(0, polarized=True))
    assert error.value.diagnostics.rank == 1
    assert error.value.diagnostics.condition_number > 1e10


def test_linked_statistics_require_matching_ids_and_true_coordinates() -> None:
    generated = EventSample(
        costheta=np.cos([0.5, 1.0]), phi=[0, 1], events=EventGrouping.from_ids([1, 2])
    )
    for theta, identifier, message in [
        (0.5, 3, "counterpart"),
        (0.8, 1, "truth coordinates"),
    ]:
        accepted = EventSample(
            costheta=np.cos([theta]),
            phi=[0],
            events=EventGrouping.from_ids([identifier]),
        )
        with pytest.raises(ValueError, match=message):
            response(generated, accepted, MomentBasis(0), MCStatistics.linked())
    with pytest.raises(ValueError, match="explicit event IDs"):
        response(
            generated,
            EventSample(
                costheta=np.cos([0.5]), phi=[0], events=EventGrouping.independent_rows()
            ),
            MomentBasis(0),
            MCStatistics.linked(),
        )


def test_mc_hypotheses_must_use_truth_angles_not_reconstructed_migrations() -> None:
    sample = EventSample(
        costheta=np.cos([0.5, 1.0]), phi=[0, 1], events=EventGrouping.from_ids([7, 7])
    )
    with pytest.raises(ValueError, match="share truth angles"):
        response(sample, sample, MomentBasis(0))


@pytest.mark.parametrize(
    "weight,density,message",
    [
        (-1.0, 1.0, "nonnegative"),
        (1.0, 0.0, "strictly positive"),
        (0.0, 1.0, "positive and finite"),
    ],
)
def test_invalid_mc_exposure_has_actionable_errors(
    weight: float, density: float, message: str
) -> None:
    sample = EventSample(
        costheta=np.cos([0.5]),
        phi=[0],
        weights=[weight],
        events=EventGrouping.independent_rows(),
    )
    with pytest.raises(ValueError, match=message):
        Acceptance(
            generated=sample,
            accepted=sample,
            basis=MomentBasis(0),
            integration=MCIntegration.importance(
                relative_density=ld.scalar("costheta") * 0 + density
            ),
            statistics=MCStatistics.data_only(),
        )


def test_response_and_extraction_arrays_are_owned_immutable_snapshots(
    angular_grid: tuple[RealArray, RealArray, RealArray],
) -> None:
    sample = angular_sample(angular_grid)
    acceptance = response(sample, sample, MomentBasis(1))
    result = pickle.loads(pickle.dumps(acceptance.extract(sample)))
    for array in (
        acceptance.response,
        result.response,
        result.measured,
        result.diagnostics.singular_values,
        result.data.covariance.matrix,
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    assert "statistics=MCStatistics.data_only()" in repr(result)


def test_signed_data_weights_cannot_produce_a_nonpositive_corrected_yield() -> None:
    sample = EventSample(
        costheta=np.cos([0.5, 1.0]), phi=[0, 1], events=EventGrouping.independent_rows()
    )
    acceptance = response(sample, sample, MomentBasis(0))
    negative = EventSample(
        costheta=np.cos([0.5, 1.0]),
        phi=[0, 1],
        weights=[1, -2],
        events=EventGrouping.independent_rows(),
    )
    with pytest.raises(ExtractionError, match="Corrected H00"):
        acceptance.extract(negative)


@pytest.mark.parametrize("scale", [1e-200, 1e200], ids=["tiny-units", "large-units"])
def test_normalized_extraction_preserves_correlated_uncertainty_under_extreme_unit_changes(
    angular_grid: tuple[RealArray, RealArray, RealArray],
    scale: float,
) -> None:
    sample = angular_sample(angular_grid)
    acceptance = response(sample, sample, MomentBasis(1))

    def data(unit: float) -> EventSample:
        return EventSample(
            costheta=np.cos(np.tile([0.4, 0.8, 1.1, 1.6, 2.0, 2.7], 2)),
            phi=np.tile([0.2, 1.1, -0.8, -1.7, 2.1, -0.3], 2),
            weights=np.tile([1, 0.8, 1.2, 0.6, 1.4, 1], 2) * unit,
            events=EventGrouping.independent_rows(),
        )

    reference = analyze(
        acceptance.extract(data(1)).data, Waveset.from_max_l(1), max_combination_size=1
    )
    extracted = acceptance.extract(data(scale))
    restored = pickle.loads(pickle.dumps(extracted.data))
    actual = analyze(restored, Waveset.from_max_l(1), max_combination_size=1)
    np.testing.assert_allclose(actual.values, reference.values, atol=1e-15)
    np.testing.assert_allclose(actual.covariance, reference.covariance, atol=1e-15)
    assert (
        actual.covariance[
            actual.component_labels.index((1, 1, "imag")),
            actual.component_labels.index((1, 1, "imag")),
        ]
        > 0
    )
    with pytest.raises(ValueError, match="rescale raw units"):
        _ = restored.covariance.matrix
