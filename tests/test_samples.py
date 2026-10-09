import pickle
from collections.abc import Iterator
from pathlib import Path

import laddu as ld
import numpy as np
import pytest

from momentous import EventGrouping, EventSample, Polarization


def test_samples_preserve_hypothesis_weights_and_physical_event_order() -> None:
    sample = EventSample(
        costheta=[0.2, 0.7, -0.2],
        phi=[0, 2 * np.pi, 1],
        weights=[0.2, 0.3, -0.1],
        events=EventGrouping.from_ids([("run", 7), ("run", 7), ("run", 3)]),
    )
    assert len(sample) == 3 and sample.n_events == 2
    np.testing.assert_array_equal(sample.weights, [0.2, 0.3, -0.1])
    np.testing.assert_array_equal(sample.phi, [0, 0, 1])
    assert "events=2" in repr(sample)


def test_laddu_adapter_evaluates_angles_polarization_and_preserves_weights() -> None:
    dataset = ld.Dataset.from_arrays(
        p4s={},
        scalars={"z": np.array([0.2, -0.3]), "azimuth": np.array([0.7, 1.2])},
        weights=np.array([0.2, 0.6]),
    )
    sample = EventSample(
        dataset,
        costheta=ld.scalar("z"),
        phi=ld.scalar("azimuth"),
        events=EventGrouping.independent_rows(),
        polarization=Polarization(magnitude=0.4, angle=ld.scalar("azimuth") / 2),
    )
    assert sample.n_events == 2 and sample.polarized
    assert sample.polarization is not None
    np.testing.assert_allclose(sample.costheta, [0.2, -0.3])
    np.testing.assert_allclose(np.asarray(sample.polarization.angle), [0.35, 0.6])
    np.testing.assert_array_equal(sample.weights, [0.2, 0.6])


def test_sample_snapshots_stay_immutable_after_caller_mutation_and_pickling() -> None:
    costheta, magnitude = np.array([0.4, 0.8]), np.array([0.2, 0.6])
    sample = EventSample(
        costheta=costheta,
        phi=[0, 1],
        events=EventGrouping.from_ids([3, 4]),
        polarization=Polarization(magnitude=magnitude, angle=0),
    )
    costheta[:] = 10
    magnitude[:] = 1
    restored = pickle.loads(pickle.dumps(sample))
    assert restored.polarization is not None
    np.testing.assert_array_equal(restored.costheta, [0.4, 0.8])
    np.testing.assert_array_equal(restored.polarization.magnitude, [0.2, 0.6])
    for array in (
        restored.costheta,
        restored.phi,
        restored.weights,
        restored.polarization.magnitude,
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize(
    "costheta,phi,weights,message",
    [
        ([-1.01], [0], [1], r"\[-1, 1\]"),
        ([1.01], [0], [1], r"\[-1, 1\]"),
        ([0.5], [np.nan], [1], "finite"),
        ([0.5], [0], [1j], "real"),
        ([0.5, 1], [0], [1, 1], "shape"),
    ],
    ids=[
        "below-minus-one",
        "above-one",
        "nonfinite-angle",
        "complex-weight",
        "length-mismatch",
    ],
)
def test_invalid_angular_columns_have_meaningful_errors(
    costheta: list[float],
    phi: list[float],
    weights: list[complex],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EventSample(
            costheta=costheta,
            phi=phi,
            weights=weights,
            events=EventGrouping.independent_rows(),
        )


def test_grouping_rejects_missing_ids_and_lengths_that_do_not_match_rows() -> None:
    with pytest.raises(ValueError, match="nonmissing"):
        EventGrouping.from_ids([(7, np.nan)])
    with pytest.raises(ValueError, match="one entry"):
        EventSample(costheta=[0.5], phi=[0], events=EventGrouping.from_ids([1, 2]))


def test_beam_magnitude_is_a_fraction_and_zero_polarization_is_allowed() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EventSample(
            costheta=[0.5],
            phi=[0],
            events=EventGrouping.independent_rows(),
            polarization=Polarization(magnitude=40, angle=0),
        )
    sample = EventSample(
        costheta=[0.5],
        phi=[0],
        events=EventGrouping.independent_rows(),
        polarization=Polarization(magnitude=0, angle=np.pi),
    )
    assert sample.polarization is not None
    np.testing.assert_array_equal(sample.polarization.angle, [0])


def test_complex_laddu_angles_are_not_silently_projected_to_real_parts() -> None:
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={"x": np.array([0.5])})
    with pytest.raises(ValueError, match="real values"):
        EventSample(
            dataset,
            costheta=ld.scalar("x") + 1j,
            phi=ld.scalar("x"),
            events=EventGrouping.independent_rows(),
        )


@pytest.mark.parametrize("storage", ["arrays", "streaming", "parquet"])
def test_laddu_event_columns_preserve_large_compound_ids_and_filtered_row_alignment(
    storage: str, tmp_path: Path
) -> None:
    scalars = {"costheta": np.array([0.2, 0.6, 0.9]), "phi": np.array([0.1, 0.4, 0.7])}
    columns = {
        "run": np.array([-7, -7, -6], dtype=np.int32),
        "event": np.array([2**63 + 1, 2**63 + 2, 2**63 + 1], dtype=np.uint64),
    }
    weights = np.array([0.2, 0.3, 0.7])
    if storage == "streaming":

        def batches(**_: object) -> Iterator[dict[str, object]]:
            for rows in (slice(0, 1), slice(1, 3)):
                yield {
                    "p4s": {},
                    "scalars": {name: value[rows] for name, value in scalars.items()},
                    "columns": {name: value[rows] for name, value in columns.items()},
                    "weights": weights[rows],
                }

        dataset = ld.Dataset.from_batches(
            batches,
            schema={
                "p4s": [],
                "scalars": list(scalars),
                "columns": {"run": "int32", "event": "uint64"},
                "weights": True,
            },
            length=3,
            cache="streaming",
        )
    else:
        dataset = ld.Dataset.from_arrays(
            p4s={}, scalars=scalars, columns=columns, weights=weights
        )
        if storage == "parquet":
            path = tmp_path / "events.parquet"
            dataset.write_to(ld.ParquetSink(path))
            dataset = ld.read_parquet(path, cache="streaming")
    sample = EventSample(
        dataset.select(ld.scalar("costheta") > 0.3),
        costheta=ld.scalar("costheta"),
        phi=ld.scalar("phi"),
        events=EventGrouping.from_columns("run", "event"),
    )
    assert sample.events.ids == ((-7, 2**63 + 2), (-6, 2**63 + 1))
    np.testing.assert_array_equal(sample.costheta, [0.6, 0.9])
    np.testing.assert_array_equal(sample.weights, [0.3, 0.7])


def test_angles_and_beam_expressions_share_one_streaming_evaluation() -> None:
    traversals = []

    def batches(**_: object) -> Iterator[dict[str, object]]:
        traversals.append(True)
        yield {
            "p4s": {},
            "scalars": {"costheta": np.array([0.5]), "phi": np.array([0.8])},
            "weights": np.array([0.3]),
        }

    dataset = ld.Dataset.from_batches(
        batches,
        schema={"p4s": [], "scalars": ["costheta", "phi"], "weights": True},
        length=1,
        cache="streaming",
    )
    traversals.clear()
    sample = EventSample(
        dataset,
        costheta=ld.scalar("costheta"),
        phi=ld.scalar("phi"),
        events=EventGrouping.independent_rows(),
        polarization=Polarization(
            magnitude=ld.scalar("costheta"), angle=ld.scalar("phi")
        ),
    )
    # One traversal for all four expressions and one for the stored weights.
    assert len(traversals) == 2
    assert sample.polarization is not None
    np.testing.assert_array_equal(sample.polarization.magnitude, [0.5])


def test_separate_laddu_samples_keep_their_own_coordinates_beam_and_weights() -> None:
    generated = EventSample(
        ld.Dataset.from_arrays(
            p4s={},
            scalars={"costheta": np.array([0.2, 0.8]), "phi": np.array([0.1, 0.7])},
            columns={"identifier": np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)},
            weights=np.array([2.0, 3.0]),
        ),
        events="identifier",
        polarization=Polarization(magnitude=0.4, angle=0.3),
    )
    accepted = EventSample(
        ld.Dataset.from_arrays(
            p4s={},
            scalars={
                "costheta": np.array([0.9, 0.9, 0.3]),
                "phi": np.array([0.8, 0.8, 0.2]),
                "P": np.full(3, 0.5),
                "Phi": np.full(3, 0.4),
            },
            columns={
                "identifier": np.array(
                    [2**63 + 2, 2**63 + 2, 2**63 + 1], dtype=np.uint64
                )
            },
            weights=np.array([0.2, 0.3, 0.7]),
        ),
        events="identifier",
        polarization=Polarization(),
    )
    assert accepted.n_events == 2
    assert accepted.events.ids == (2**63 + 2, 2**63 + 2, 2**63 + 1)
    np.testing.assert_array_equal(accepted.costheta, [0.9, 0.9, 0.3])
    np.testing.assert_array_equal(accepted.weights, [0.2, 0.3, 0.7])
    np.testing.assert_array_equal(generated.costheta, [0.2, 0.8])
    assert accepted.polarization is not None
    np.testing.assert_array_equal(accepted.polarization.magnitude, [0.5] * 3)


def test_column_declarations_are_explicit_and_cannot_treat_float_scalars_as_ids() -> (
    None
):
    grouping = EventGrouping.from_columns("event")
    assert pickle.loads(pickle.dumps(grouping)) == grouping
    assert "event" in str(grouping)
    for names in ((), ("",), ("run", "run")):
        with pytest.raises(ValueError):
            EventGrouping.from_columns(*names)
    with pytest.raises(ValueError, match="require a source"):
        EventSample(costheta=[0.5], phi=[0], events=grouping)
    dataset = ld.Dataset.from_arrays(p4s={}, scalars={"event": np.array([7.0])})
    with pytest.raises(ValueError, match="exact integers"):
        EventSample(
            dataset,
            costheta=ld.scalar("event") / 10,
            phi=ld.scalar("event") / 10,
            events=grouping,
        )


@pytest.mark.parametrize("storage", ["mapping", "structured-array"])
def test_named_columns_group_events_and_snapshot_ordinary_sources(
    storage: str,
) -> None:
    columns = {
        "costheta": np.array([0.2, 0.8]),
        "phi": np.array([0.1, 0.7]),
        "run": np.array(["a", "b"]),
        "event": np.array([2**63 + 1, 2**63 + 1], dtype=np.uint64),
        "weight": np.array([2.0, 3.0]),
        "P": np.array([0.4, 0.5]),
        "Phi": np.array([0.3, 0.6]),
    }
    if storage == "mapping":
        source = columns
    else:
        source = np.empty(
            2, dtype=[(name, values.dtype) for name, values in columns.items()]
        )
        for name, values in columns.items():
            source[name] = values
    events = ("run", "event")
    generated = EventSample(
        source,
        weights="weight",
        events=events,
        polarization=Polarization(magnitude="P", angle="Phi"),
    )
    accepted = EventSample(
        {
            "run": ["b", "b", "a"],
            "event": np.array([2**63 + 1] * 3, dtype=np.uint64),
            "weight": [0.2, 0.3, 0.7],
            "costheta": [0.8, 0.8, 0.2],
            "phi": [0.7, 0.7, 0.1],
            "P": [0.5, 0.5, 0.4],
            "Phi": [0.6, 0.6, 0.3],
        },
        events=events,
        weights="weight",
        polarization=Polarization(),
    )
    assert generated.events.ids == (("a", 2**63 + 1), ("b", 2**63 + 1))
    assert accepted.n_events == 2
    np.testing.assert_array_equal(accepted.costheta, [0.8, 0.8, 0.2])
    np.testing.assert_array_equal(accepted.weights, [0.2, 0.3, 0.7])
    assert accepted.polarization is not None
    np.testing.assert_array_equal(accepted.polarization.magnitude, [0.5, 0.5, 0.4])
    source["costheta"][:] = 1.5
    np.testing.assert_array_equal(generated.costheta, [0.2, 0.8])
    restored = pickle.loads(pickle.dumps(accepted))
    assert restored.events == accepted.events
    np.testing.assert_array_equal(restored.costheta, accepted.costheta)


def test_laddu_column_names_batch_angles_and_beam_and_preserve_native_weights() -> None:
    dataset = ld.Dataset.from_arrays(
        p4s={},
        scalars={
            "costheta": np.array([0.5]),
            "phi": np.array([0.8]),
            "P": np.array([0.4]),
            "Phi": np.array([0.3]),
        },
        columns={"event": np.array([7])},
        weights=np.array([0.2]),
    )
    sample = EventSample(
        dataset,
        events=EventGrouping.from_columns("event"),
        polarization=Polarization(magnitude="P", angle="Phi"),
    )
    np.testing.assert_array_equal(sample.costheta, [0.5])
    np.testing.assert_array_equal(sample.weights, [0.2])
    assert sample.polarization is not None
    np.testing.assert_array_equal(sample.polarization.angle, [0.3])


def test_named_weight_columns_are_explicit_for_ordinary_sources() -> None:
    rows = {"costheta": [0.5], "phi": [0.8], "weight": [-0.2]}
    events = EventGrouping.independent_rows()
    unit = EventSample(rows, events=events)
    weighted = EventSample(rows, weights="weight", events=events)
    np.testing.assert_array_equal(unit.weights, [1])
    np.testing.assert_array_equal(weighted.weights, [-0.2])


@pytest.mark.parametrize("bad_input", ["missing-column", "expression-without-dataset"])
def test_source_errors_explain_missing_columns_and_unavailable_expression_evaluation(
    bad_input: str,
) -> None:
    rows = {"costheta": [0.5], "phi": [0.8]}
    if bad_input == "missing-column":
        with pytest.raises(ValueError, match="missing"):
            EventSample(
                rows, costheta="missing", events=EventGrouping.independent_rows()
            )
    else:
        with pytest.raises(TypeError, match="expressions require a laddu dataset"):
            EventSample(
                rows,
                costheta=ld.scalar("costheta"),
                events=EventGrouping.independent_rows(),
            )


def test_event_grouping_requires_an_explicit_declaration() -> None:
    with pytest.raises(TypeError, match="events"):
        EventSample(costheta=[0.5], phi=[0])  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match="my_event_key"):
        EventSample({"costheta": [0.5], "phi": [0]}, events="my_event_key")
