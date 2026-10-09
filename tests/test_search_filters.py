from unittest.mock import Mock

import pytest

from momentous import (
    CheckResult,
    Covariance,
    MomentData,
    SearchResult,
    Status,
    Waveset,
    analyze,
)
from momentous._engine import Engine


@pytest.fixture
def searched() -> SearchResult:
    data = MomentData({(0, 0): 1, (1, 0): 0.2}, covariance=Covariance.exact())
    return analyze(data, Waveset.from_max_l(1)).search(max_size=3)


@pytest.mark.parametrize("status", list(Status))
@pytest.mark.parametrize(
    "include,exclude",
    [
        (None, None),
        (Waveset([]), Waveset([])),
        (Waveset([(0, 0)]), None),
        (None, Waveset([(1, -1)])),
        (Waveset([(0, 0), (1, 0)]), Waveset([(1, -1)])),
        (Waveset.from_max_l(1), None),
    ],
    ids=["unfiltered", "empty", "required", "forbidden", "combined", "too-large"],
)
def test_wave_filters_select_matching_results_without_changing_status_or_order(
    searched: SearchResult,
    status: Status,
    include: Waveset | None,
    exclude: Waveset | None,
) -> None:
    required = set(() if include is None else include.waves)
    forbidden = set(() if exclude is None else exclude.waves)
    expected = [
        candidate
        for candidate in searched.wavesets(status)
        if required <= set(candidate.waves)
        and not forbidden.intersection(candidate.waves)
    ]
    assert list(searched.wavesets(status, include=include, exclude=exclude)) == expected
    assert all(searched.status(candidate) is status for candidate in expected)


def test_unresolved_candidates_are_filtered_without_becoming_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Engine, "evaluate", Mock(return_value=CheckResult(Status.UNRESOLVED))
    )
    pool = Waveset([(0, 0), (1, 0), (2, 0)])
    data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
    result = analyze(data, pool).search()
    required, forbidden = Waveset([(0, 0)]), Waveset([(2, 0)])
    assert list(result.wavesets("unresolved", include=required, exclude=forbidden)) == [
        required,
        Waveset([(0, 0), (1, 0)]),
    ]
    assert list(result.wavesets(include=required, exclude=forbidden)) == []
    assert result.minimal_wavesets(include=required, exclude=forbidden) == ()
    assert not result.complete


def test_filtering_minima_preserves_the_original_minimality() -> None:
    data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
    pool = Waveset([(0, 0), (1, 0), (1, 1)])
    result = analyze(data, pool).search()
    required = Waveset([(0, 0)])
    forbidden = Waveset([(1, 1)])
    assert result.minimal_wavesets(include=required, exclude=forbidden) == (required,)
    forced_pair = Waveset([(0, 0), (1, 0)])
    assert list(result.wavesets(include=forced_pair))
    assert result.minimal_wavesets(include=forced_pair) == ()


def test_reflectivity_sectors_are_distinct_filter_identities() -> None:
    pool = Waveset([(0, 0, "+"), (0, 0, "-"), (1, 0, "+"), (1, 0, "-")])
    data = MomentData({(0, 0, 0): 1, (0, 0, 1): 0}, covariance=Covariance.exact())
    result = analyze(data, pool).search()
    required, forbidden = Waveset([(0, 0, "+")]), Waveset([(0, 0, "-")])
    assert list(result.wavesets(include=required, exclude=forbidden)) == [
        Waveset([(0, 0, "+"), (1, 0, "-")]),
        Waveset([(0, 0, "+"), (1, 0, "-"), (1, 0, "+")]),
    ]


def test_filters_restrict_compressed_blocks_before_expanding_supersets() -> None:
    # This pool has over 68 billion subsets, but exactly one matches the filters.
    pool = Waveset.from_max_l(5)
    data = MomentData({(0, 0): 1, (1, 0): 0}, covariance=Covariance.exact())
    result = analyze(data, pool).search(workers=1)
    required, forbidden = Waveset(pool.waves[:-1]), Waveset(pool.waves[-1:])
    assert list(result.wavesets(include=required, exclude=forbidden)) == [required]


@pytest.mark.parametrize("accessor", ["wavesets", "minimal_wavesets"])
@pytest.mark.parametrize("argument", ["include", "exclude"])
@pytest.mark.parametrize(
    "waves,message",
    [
        (Waveset([(2, 0)]), "outside the pool"),
        (Waveset([(0, 0, "+")]), "polarization"),
    ],
    ids=["unknown-wave", "wrong-mode"],
)
def test_filter_waves_must_belong_to_the_search_pool(
    searched: SearchResult, accessor: str, argument: str, waves: Waveset, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        list(getattr(searched, accessor)(**{argument: waves}))


@pytest.mark.parametrize("accessor", ["wavesets", "minimal_wavesets"])
def test_the_same_wave_cannot_be_required_and_forbidden(
    searched: SearchResult, accessor: str
) -> None:
    wave = Waveset([(0, 0)])
    with pytest.raises(ValueError, match="both included and excluded"):
        list(getattr(searched, accessor)(include=wave, exclude=wave))


def test_filters_use_waveset_objects(searched: SearchResult) -> None:
    with pytest.raises(TypeError, match="Waveset"):
        list(searched.wavesets(include=[(0, 0)]))  # ty: ignore[invalid-argument-type]
