"""Regression tests for declared responses and grouped signed MC estimators."""

import pickle

import laddu as ld
import numpy as np
import pytest
from numpy.typing import ArrayLike

from momentous import (
    Acceptance,
    EventGrouping,
    EventSample,
    ExtractionError,
    ExtractionResult,
    MCIntegration,
    MCStatistics,
    MomentBasis,
    Polarization,
    ResponseModel,
    Waveset,
    analyze,
)
from momentous.results import RealArray


def sample(
    z: ArrayLike,
    phi: ArrayLike,
    weights: ArrayLike,
    ids: ArrayLike,
    *,
    run: int = 2,
    beam: Polarization | None = None,
) -> EventSample:
    return EventSample(
        costheta=z,
        phi=phi,
        weights=weights,
        events=EventGrouping.from_ids(
            [(run, int(i)) for i in np.asarray(ids, dtype=np.uint64)]
        ),
        polarization=beam,
    )


def features(events: EventSample) -> RealArray:
    z, phi = events.costheta, events.phi
    sine = np.sqrt(1 - z * z)
    return np.column_stack(
        (
            np.ones(len(events)),
            z,
            -sine * np.cos(phi) / np.sqrt(2),
            sine * np.sin(phi) / np.sqrt(2),
        )
    )


def signed_response(generated: EventSample, accepted: EventSample) -> Acceptance:
    return Acceptance(
        generated,
        accepted,
        basis=MomentBasis(1),
        response_model=ResponseModel.reconstructed_diagonal(),
        statistics=MCStatistics.independent(),
    )


@pytest.fixture
def tiny() -> tuple[EventSample, EventSample, EventSample]:
    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    z = np.repeat(vertices[:, 2], 2)
    phi = np.repeat(np.arctan2(vertices[:, 1], vertices[:, 0]), 2)
    phi[1::2] += 0.12
    ids = np.repeat(np.arange(4, dtype=np.uint64) + np.uint64(2**63), 2)
    accepted = sample(z, phi, np.tile([1.0, -1 / 6], 4), ids)
    generated = sample([0.2, 0.3, 0.4], [0, 1, 2], [2, 3, 5], [7, 8, 9], run=1)
    data = sample([0.2, -0.4, 0.6], [0.1, 1.3, -1.2], [2, -0.2, 3], [9, 9, 10], run=3)
    return generated, accepted, data


def independent_covariance(result: ExtractionResult, matrix: RealArray) -> RealArray:
    # Public coordinate labels select the independent raw moments.
    labels = result.data.component_labels
    indices = [
        labels.index(key)
        for key in [(0, 0, "real"), (1, 0, "real"), (1, 1, "real"), (1, 1, "imag")]
    ]
    return matrix[np.ix_(indices, indices)]


def test_explicit_policies_and_analytic_signed_influences(tiny: tuple) -> None:
    generated, accepted, data = tiny
    with pytest.raises(ValueError, match="share truth angles"):
        Acceptance(
            generated,
            accepted,
            basis=MomentBasis(1),
            statistics=MCStatistics.independent(),
        )
    acceptance = signed_response(generated, accepted)
    f, c = features(accepted), features(accepted) * [1, 3, 6, 6]
    rows = np.einsum("ni,nj,n->nij", f, c, accepted.weights)
    matrices = rows.reshape(4, 2, 4, 4).sum(axis=1)
    G = 10.0
    R = matrices.sum(axis=0) / G
    drows = features(data) * data.weights[:, None]
    d = np.vstack((drows[0] + drows[1], drows[2]))
    H = np.linalg.solve(R, d.sum(axis=0))
    udata = np.linalg.solve(R, d.T).T
    uacc = -np.linalg.solve(R, (matrices @ H / G).T).T
    ugen = np.array([2, 3, 5])[:, None] / G * H
    expected_mc = uacc.T @ uacc + ugen.T @ ugen
    result = acceptance.extract(data)
    np.testing.assert_allclose(result.response, R, atol=1e-14)
    actual = [
        result.data.moments[(0, 0)].real,
        result.data.moments[(1, 0)].real,
        result.data.moments[(1, 1)].real,
        result.data.moments[(1, 1)].imag,
    ]
    np.testing.assert_allclose(actual, H, atol=1e-13)
    np.testing.assert_allclose(
        independent_covariance(result, result.mc_covariance.matrix),
        expected_mc,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        independent_covariance(result, result.data_covariance.matrix),
        udata.T @ udata,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.data.covariance.matrix,
        result.data_covariance.matrix + result.mc_covariance.matrix,
        atol=1e-12,
    )
    ungrouped = -np.linalg.solve(R, (rows @ H / G).T).T
    assert not np.allclose(ungrouped.T @ ungrouped, uacc.T @ uacc)
    assert np.linalg.eigvalsh(result.data.covariance.matrix).min() > -1e-10
    assert len(set(accepted._ids)) == 4
    assert all(identifier[1] >= 2**63 for identifier in accepted._ids)
    analysis = analyze(result.data, Waveset.from_max_l(1), max_combination_size=1)
    assert np.all(np.isfinite(analysis.covariance))


def test_splitting_and_permutations_preserve_grouped_signed_covariance(
    tiny: tuple,
) -> None:
    generated, accepted, data = tiny
    original = signed_response(generated, accepted).extract(data)
    ids = np.asarray([identifier[1] for identifier in accepted._ids], dtype=np.uint64)
    row_ids = ids[accepted._indices]
    split = sample(
        np.repeat(accepted.costheta, 2),
        np.repeat(accepted.phi, 2),
        np.repeat(accepted.weights / 2, 2),
        np.repeat(row_ids, 2),
    )
    permutation = np.random.default_rng(9).permutation(len(split))
    split_ids = np.repeat(row_ids, 2)
    permuted = sample(
        split.costheta[permutation],
        split.phi[permutation],
        split.weights[permutation],
        split_ids[permutation],
    )
    for changed in (split, permuted):
        result = signed_response(generated, changed).extract(data)
        np.testing.assert_allclose(result.response, original.response, atol=1e-14)
        np.testing.assert_allclose(
            result.mc_covariance.matrix, original.mc_covariance.matrix, atol=1e-11
        )
    separate = sample(split.costheta, split.phi, split.weights, np.arange(len(split)))
    result = signed_response(generated, separate).extract(data)
    assert not np.allclose(result.mc_covariance.matrix, original.mc_covariance.matrix)

    # Generated exposure also clusters rows before squaring its denominator influence.
    gen_ids = np.repeat([7, 8, 9], 2)
    split_gen = sample(
        np.repeat(generated.costheta, 2),
        np.repeat(generated.phi, 2),
        np.repeat(generated.weights / 2, 2),
        gen_ids,
        run=1,
    )
    split_result = signed_response(split_gen, accepted).extract(data)
    np.testing.assert_allclose(split_result.response, original.response, atol=1e-14)
    np.testing.assert_allclose(
        split_result.mc_covariance.matrix, original.mc_covariance.matrix, atol=1e-11
    )
    independent_gen = sample(
        split_gen.costheta,
        split_gen.phi,
        split_gen.weights,
        np.arange(len(split_gen)),
        run=1,
    )
    independent_result = signed_response(independent_gen, accepted).extract(data)
    assert not np.allclose(
        independent_result.mc_covariance.matrix, original.mc_covariance.matrix
    )


def test_zero_net_event_cancels_only_identical_matrix_contributions(
    tiny: tuple,
) -> None:
    generated, accepted, data = tiny
    base = signed_response(generated, accepted).extract(data)
    ids = [identifier[1] for identifier in accepted._ids for _ in range(2)]
    for delta in (0.0, 0.3):
        extra = sample(
            np.r_[accepted.costheta, 0.4, 0.4],
            np.r_[accepted.phi, 0.1, 0.1 + delta],
            np.r_[accepted.weights, 0.2, -0.2],
            ids + [19, 19],
        )
        result = signed_response(generated, extra).extract(data)
        if delta == 0:
            np.testing.assert_allclose(result.response, base.response, atol=1e-14)
            np.testing.assert_allclose(
                result.mc_covariance.matrix, base.mc_covariance.matrix, atol=1e-11
            )
        else:
            assert not np.allclose(result.response, base.response)
            assert not np.allclose(
                result.mc_covariance.matrix, base.mc_covariance.matrix
            )


@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("migration", [False, True])
def test_paired_truth_closure_and_diagonal_bias(
    polarized: bool, migration: bool
) -> None:
    z, quadrature = np.polynomial.legendre.leggauss(6)
    phi = np.arange(12) * 2 * np.pi / 12
    z, phi = (column.ravel() for column in np.meshgrid(z, phi, indexing="ij"))
    weights = np.repeat(quadrature / 24, 12)
    orientations = np.arange(8) * np.pi / 8
    z, phi, weights = (np.repeat(column, 8) for column in (z, phi, weights / 8))
    Phi = np.tile(orientations, len(z) // 8)
    beam = Polarization(magnitude=0.4, angle=Phi) if polarized else None
    row_ids = np.repeat(np.arange(len(z)), 2)
    truth_beam = (
        Polarization(magnitude=0.4, angle=np.repeat(Phi, 2)) if polarized else None
    )
    truth = sample(np.repeat(z, 2), np.repeat(phi, 2), 1, row_ids, beam=truth_beam)
    shift = np.tile([0.17, 0.37], len(z)) if migration else 0
    reco_beam = (
        Polarization(
            magnitude=0.4, angle=np.repeat(Phi, 2) + (0.12 if migration else 0)
        )
        if polarized
        else None
    )
    efficiency = 0.6 + 0.1 * truth.costheta
    q = 1 + 0.2 * truth.costheta
    estimator = np.repeat(weights, 2) * np.tile([1, -1 / 6], len(z)) * efficiency * q
    accepted = sample(
        truth.costheta, truth.phi + shift, estimator, row_ids, beam=reco_beam
    )
    generation = sample(
        z, phi, weights * (1 + 0.2 * z), np.arange(len(z)), run=1, beam=beam
    )
    basis = MomentBasis(1, polarized=polarized)
    H = (
        np.array([100, 12, 4, -2, 1, 0.5, -0.8])
        if polarized
        else np.array([100, 4, 1, -0.8])
    )
    # Independent analytic intensity, including polarized expansion conventions.
    f = features(truth)
    if polarized:
        cos = 0.4 * np.cos(2 * np.repeat(Phi, 2))
        sin = 0.4 * np.sin(2 * np.repeat(Phi, 2))
        c = np.column_stack(
            (
                f[:, 0],
                cos,
                f[:, 1],
                cos * f[:, 1],
                f[:, 2],
                cos * f[:, 2],
                sin * f[:, 3],
            )
        ) * [1, 1, 3, 3, 6, 6, 6]
    else:
        c = f * [1, 3, 6, 6]
    data = sample(
        accepted.costheta,
        accepted.phi,
        estimator / q * (c @ H),
        row_ids,
        run=3,
        beam=reco_beam,
    )
    acceptance = Acceptance(
        generation,
        accepted,
        basis=basis,
        integration=MCIntegration.importance(
            relative_density=1 + 0.2 * ld.scalar("costheta")
        ),
        response_model=ResponseModel.truth_to_reconstruction(truth=truth),
        statistics=MCStatistics.independent(),
    )
    result = acceptance.extract(data)
    actual = [
        result.data.moments[o.moment.key].real
        if polarized or o.key[-1] == "real"
        else result.data.moments[o.moment.key].imag
        for o in basis.observables
    ]
    np.testing.assert_allclose(actual, H, atol=1e-12)
    if migration:
        # Both the angular response and density evaluation must use paired truth.
        if not polarized:
            assert not np.allclose(
                acceptance.response / basis.expansion_scales,
                (acceptance.response / basis.expansion_scales).T,
            )
        diagonal = Acceptance(
            generation,
            accepted,
            basis=basis,
            integration=acceptance.integration,
            response_model=ResponseModel.reconstructed_diagonal(),
            statistics=MCStatistics.independent(),
        ).extract(data)
        assert not np.allclose(diagonal.response, result.response)
        diagonal_values = [
            diagonal.data.moments[o.moment.key].real
            if polarized or o.key[-1] == "real"
            else diagonal.data.moments[o.moment.key].imag
            for o in basis.observables
        ]
        assert not np.allclose(diagonal_values, H, rtol=1e-6, atol=1e-12)
    for scale in (1e-80, 1e80):
        gen = sample(
            generation.costheta,
            generation.phi,
            generation.weights * scale,
            np.arange(len(generation)),
            run=1,
            beam=beam,
        )
        acc = sample(
            accepted.costheta,
            accepted.phi,
            accepted.weights * scale,
            row_ids,
            beam=reco_beam,
        )
        changed = Acceptance(
            gen,
            acc,
            basis=basis,
            integration=acceptance.integration,
            response_model=acceptance.response_model,
            statistics=MCStatistics.independent(),
        ).extract(data)
        np.testing.assert_allclose(changed.response, result.response, atol=1e-14)
        np.testing.assert_allclose(
            changed.mc_covariance.matrix, result.mc_covariance.matrix, atol=1e-10
        )


def test_paired_truth_density_not_reconstruction(tiny: tuple) -> None:
    generated, accepted, _ = tiny
    ids = [identifier[1] for identifier in accepted._ids for _ in range(2)]
    truth = sample(
        np.repeat([0.1, 0.2, -0.3, -0.6], 2),
        np.repeat([0.2, 1.4, -1.3, 2.5], 2),
        99,
        ids,
    )
    acceptance = Acceptance(
        generated,
        accepted,
        basis=MomentBasis(1),
        integration=MCIntegration.importance(
            relative_density=1 + 0.3 * ld.scalar("costheta")
        ),
        response_model=ResponseModel.truth_to_reconstruction(truth=truth),
        statistics=MCStatistics.independent(),
    )
    exposure = np.sum(generated.weights / (1 + 0.3 * generated.costheta))
    expected = features(accepted).T @ (
        features(truth)
        * [1, 3, 6, 6]
        * (accepted.weights / (1 + 0.3 * truth.costheta) / exposure)[:, None]
    )
    np.testing.assert_allclose(acceptance.response, expected, atol=1e-14)
    # Independently check influences for distinct observed/intensity coordinates.
    H = np.array([100, 2, 1, -0.5])
    a = accepted.weights / (1 + 0.3 * truth.costheta)
    c = features(truth) * [1, 3, 6, 6]
    data = sample(accepted.costheta, accepted.phi, a * (c @ H) / exposure, ids, run=3)
    result = acceptance.extract(data)
    matrices = (
        np.einsum("ni,nj,n->nij", features(accepted), c, a)
        .reshape(4, 2, 4, 4)
        .sum(axis=1)
    )
    uacc = -np.linalg.solve(expected, (matrices @ H / exposure).T).T
    B = generated.weights / (1 + 0.3 * generated.costheta)
    ugen = B[:, None] / exposure * H
    np.testing.assert_allclose(
        independent_covariance(result, result.mc_covariance.matrix),
        uacc.T @ uacc + ugen.T @ ugen,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        Acceptance(
            generated,
            accepted,
            basis=MomentBasis(1),
            integration=MCIntegration.importance(
                relative_density=ld.scalar("costheta") + 0.5
            ),
            response_model=acceptance.response_model,
            statistics=MCStatistics.independent(),
        )


def test_declared_truth_and_generated_validation(tiny: tuple) -> None:
    generated, accepted, _ = tiny
    for truth, message in [(accepted, "share truth angles"), (generated, "row order")]:
        with pytest.raises(ValueError, match=message):
            Acceptance(
                generated,
                accepted,
                basis=MomentBasis(1),
                response_model=ResponseModel.truth_to_reconstruction(truth=truth),
                statistics=MCStatistics.independent(),
            )
    negative = sample(
        generated.costheta, generated.phi, -generated.weights, [7, 8, 9], run=1
    )
    with pytest.raises(ValueError, match="nonnegative"):
        signed_response(negative, accepted)
    with pytest.raises(ValueError, match="positive and finite"):
        signed_response(
            sample(generated.costheta, generated.phi, 0, [7, 8, 9], run=1), accepted
        )
    invalid_ids = sample(
        accepted.costheta, accepted.phi, 1, np.arange(len(accepted)), run=4
    )
    with pytest.raises(ValueError, match="row order"):
        Acceptance(
            generated,
            accepted,
            basis=MomentBasis(1),
            response_model=ResponseModel.truth_to_reconstruction(truth=invalid_ids),
            statistics=MCStatistics.independent(),
        )


def test_signed_linked_response_pairs_genuine_ids_and_declared_truth(
    tiny: tuple,
) -> None:
    _, accepted, data = tiny
    ids = [identifier[1] for identifier in accepted._ids]
    generated = sample(accepted.costheta[::2], accepted.phi[::2], 1, ids)
    truth = sample(
        np.repeat(generated.costheta, 2),
        np.repeat(generated.phi, 2),
        1,
        [i for i in ids for _ in range(2)],
    )
    acceptance = Acceptance(
        generated,
        accepted,
        basis=MomentBasis(1),
        response_model=ResponseModel.truth_to_reconstruction(truth=truth),
        statistics=MCStatistics.linked(),
    )
    result = acceptance.extract(data)
    matrices = (
        np.einsum(
            "ni,nj,n->nij",
            features(accepted),
            features(truth) * [1, 3, 6, 6],
            accepted.weights,
        )
        .reshape(4, 2, 4, 4)
        .sum(axis=1)
    )
    H = np.linalg.solve(acceptance.response, result.measured)
    numerator = matrices @ H - (acceptance.response @ H)[None, :]
    u = -np.linalg.solve(acceptance.response, (numerator / 4).T).T
    np.testing.assert_allclose(
        independent_covariance(result, result.mc_covariance.matrix),
        4 / 3 * u.T @ u,
        atol=1e-12,
    )
    other_namespace = sample(generated.costheta, generated.phi, 1, ids, run=1)
    with pytest.raises(ValueError, match="counterpart"):
        Acceptance(
            other_namespace,
            accepted,
            basis=MomentBasis(1),
            response_model=acceptance.response_model,
        )
    shifted_truth = sample(
        truth.costheta, truth.phi + 0.1, 1, [i for i in ids for _ in range(2)]
    )
    with pytest.raises(ValueError, match="share truth coordinates"):
        Acceptance(
            generated,
            accepted,
            basis=MomentBasis(1),
            response_model=ResponseModel.truth_to_reconstruction(truth=shifted_truth),
        )


def test_signed_cancellation_fails_diagnostically() -> None:
    generated = sample([0, 0], [0, 0], 1, [1, 2], run=1)
    for weights, message in [([1, -1], "unidentifiable"), ([1, -2], "Corrected H00")]:
        accepted = sample([0, 0], [0, 0], weights, [3, 3])
        with pytest.raises(ExtractionError, match=message):
            acceptance = Acceptance(
                generated,
                accepted,
                basis=MomentBasis(0),
                statistics=MCStatistics.independent(),
            )
            acceptance.extract(generated)


def test_policy_and_diagnostic_serialization(tiny: tuple) -> None:
    generated, accepted, data = tiny
    acceptance = signed_response(generated, accepted)
    result = pickle.loads(pickle.dumps(acceptance.extract(data)))
    diag = result.diagnostics
    assert diag.response_model == "reconstructed_diagonal"
    assert diag.accepted_weights == "signed"
    assert (
        diag.accepted_positive_rows,
        diag.accepted_negative_rows,
        diag.accepted_zero_rows,
    ) == (4, 4, 0)
    assert diag.accepted_positive_sum == 4
    assert diag.accepted_negative_sum == pytest.approx(-2 / 3)
    assert diag.accepted_contribution_sum == pytest.approx(10 / 3)
    assert (diag.generated_events, diag.accepted_events, diag.generated_exposure) == (
        3,
        4,
        10,
    )
    for policy in (acceptance.response_model,):
        restored = pickle.loads(pickle.dumps(policy))
        assert restored.mode == policy.mode
    truth_model = ResponseModel.truth_to_reconstruction(truth=generated)
    restored = pickle.loads(pickle.dumps(truth_model))
    assert restored.mode == truth_model.mode
    np.testing.assert_array_equal(restored.truth_sample.costheta, generated.costheta)
    with pytest.raises(ValueError):
        restored.truth_sample.costheta.flags.writeable = True


def test_signed_grouped_independent_poisson_pseudoexperiments() -> None:
    rng = np.random.default_rng(9241)
    size = 1800
    z, phi = rng.uniform(-1, 1, size), rng.uniform(-np.pi, np.pi, size)
    accepted = sample(
        np.repeat(z, 2),
        np.repeat(phi, 2) + np.tile([0, 0.2], size),
        np.tile([1, -1 / 6], size),
        np.repeat(np.arange(size), 2),
    )
    generated = sample(z, phi, 1, np.arange(size), run=1)
    data = sample([0.2, -0.4, 0.7], [0, 1, -1], [20, 30, 40], [1, 2, 3], run=3)
    result = signed_response(generated, accepted).extract(data)
    f = features(accepted)
    matrices = (
        np.einsum("ni,nj,n->nij", f, f * [1, 3, 6, 6], accepted.weights)
        .reshape(size, 2, 4, 4)
        .sum(axis=1)
    )
    draws = []
    for _ in range(1600):
        # Resample independent physical events and exposure, never hypotheses.
        counts = rng.poisson(1, size)
        G = rng.poisson(size)
        R = np.einsum("n,nij->ij", counts, matrices) / G
        draws.append(np.linalg.solve(R, result.measured))
    empirical = np.cov(np.asarray(draws), rowvar=False)
    predicted = independent_covariance(result, result.mc_covariance.matrix)
    np.testing.assert_allclose(np.diag(predicted), np.diag(empirical), rtol=0.15)
    assert np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted) < 0.12


def test_signed_response_may_be_indefinite_with_positive_corrected_yield() -> None:
    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    z, phi = vertices[:, 2], np.arctan2(vertices[:, 1], vertices[:, 0])
    generated = sample(z, phi, 1, [1, 2, 3, 4], run=1)
    accepted = sample(z, phi, [1, 1, 1, -0.25], [1, 2, 3, 4])
    acceptance = Acceptance(
        generated,
        accepted,
        basis=MomentBasis(1),
        statistics=MCStatistics.independent(),
    )
    gram = acceptance.response / np.array([1, 3, 6, 6])
    assert np.linalg.eigvalsh(gram).min() < 0
    H = np.array([10, 1, 1, 0.1])
    data = sample(
        z,
        phi,
        accepted.weights * ((features(accepted) * [1, 3, 6, 6]) @ H) / 4,
        [1, 2, 3, 4],
        run=3,
    )
    result = acceptance.extract(data)
    assert result.data.moments[(0, 0)].real == pytest.approx(10)
    assert np.linalg.eigvalsh(result.data.covariance.matrix).min() > -1e-10


def test_signed_accepted_weights_work_automatically_with_default_policies() -> None:
    z = np.array([0.1, -0.4, 0.8])
    phi = np.array([0.2, -1.0, 2.4])
    generated = sample(z, phi, 1, [7, 8, 9])
    accepted = sample(
        np.repeat(z, 2),
        np.repeat(phi, 2),
        np.tile([1, -1 / 6], 3),
        np.repeat([7, 8, 9], 2),
    )
    acceptance = Acceptance(generated, accepted, basis=MomentBasis(0))
    result = acceptance.extract(generated)
    assert acceptance.statistics.mode == "linked"
    assert acceptance.response_model.mode == "truth"
    assert result.data.moments[(0, 0)] == pytest.approx(3 / (5 / 6))
    assert result.diagnostics.accepted_negative_rows == 3
    assert result.diagnostics.accepted_negative_sum == pytest.approx(-0.5)
    assert result.diagnostics.accepted_weights == "signed"
