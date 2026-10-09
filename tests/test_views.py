import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from momentous import (
    Covariance,
    Moment,
    MomentData,
    Status,
    UnresolvedCompatibilityError,
    UnresolvedPair,
    Waveset,
    analyze,
)
from momentous.geometry import _ConicSolver


def test_raw_measurement_snapshot_can_be_reused_after_caller_mutation() -> None:
    values = {(0, 0): 10.0, (1, 1): 0j}
    data = MomentData(values, covariance=Covariance.exact())
    values[(1, 1)] = 100
    restored = pickle.loads(pickle.dumps(data))
    first = analyze(data, Waveset([(0, 0)]))
    second = analyze(restored, Waveset.from_max_l(1))
    assert first.check().valid and second.check().valid
    assert first.moments == second.moments
    assert data.moments[(0, 0)] == restored.moments[(0, 0)] == 10
    assert first.observables == (
        Moment(0, 0).real,
        Moment(0, 0).imag,
        Moment(1, 1).real,
        Moment(1, 1).imag,
    )
    with pytest.raises(ValueError):
        restored.values.flags.writeable = True


def test_candidate_view_matches_direct_geometry_and_checks() -> None:
    data = MomentData({(0, 0): 1, (1, 1): 0j}, covariance=Covariance.exact())
    analysis = analyze(data, Waveset.from_max_l(1))
    waves = Waveset([(0, 0), (1, 1)])
    candidate = analysis.for_waves(waves)
    assert candidate.waves is waves
    assert candidate.status() is analysis.status(waves)
    assert candidate.bounds() == analysis.bounds(waves)
    np.testing.assert_array_equal(candidate.operators(), analysis.operators(waves))
    h11 = Moment(1, 1)
    selected = candidate.region(h11.real, h11.imag, n_directions=8)
    shorthand = analysis.region(
        (1, 1, "real"), (1, 1, "imag"), waves=waves, n_directions=8
    )
    np.testing.assert_array_equal(selected.boundary, shorthand.boundary)
    assert candidate.check(diagnostics=True).valid
    with pytest.raises(ValueError, match="outside the pool"):
        analysis.for_waves(Waveset([(2, 0)]))
    with pytest.raises(ValueError, match="polarization"):
        analysis.for_waves(Waveset([(0, 0, "+")]))
    with pytest.raises(ValueError, match="distinct"):
        candidate.region(h11.real, h11.real)
    empty = analysis.for_waves(Waveset([]))
    assert empty.status() is Status.INCOMPATIBLE
    with pytest.raises(ValueError, match="Empty candidates"):
        empty.bounds()


def test_unresolved_pair_diagnostics_do_not_expose_fictitious_projection_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_ConicSolver, "solve", Mock(return_value=False))
    data = MomentData(
        {(0, 0): 1, (1, 1): (0.95 + 0.28j) / (2 * np.sqrt(3))},
        covariance=Covariance.exact(),
    )
    candidate = analyze(data, Waveset([(0, 0), (1, 1)])).for_waves(
        Waveset([(0, 0), (1, 1)])
    )
    checked = candidate.check(diagnostics=True)
    assert checked.status is Status.UNRESOLVED
    assert len(checked.diagnostics) == 1
    diagnostic = checked.diagnostics[0]
    assert isinstance(diagnostic, UnresolvedPair)
    assert diagnostic.components == ((1, 1, "real"), (1, 1, "imag"))
    assert "witness" in diagnostic.reason
    with pytest.raises(UnresolvedCompatibilityError):
        bool(checked)
