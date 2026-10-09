"""Format-independent angular samples and explicit physical-event grouping."""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import laddu as ld
import numpy as np
from numpy.typing import ArrayLike, NDArray

from momentous._sources import (
    ColumnInput,
    DataSource,
    evaluate_columns,
    read_column,
    source_weights,
)
from momentous.covariance import readonly
from momentous.results import RealArray


def real_array(value: ArrayLike, name: str) -> RealArray:
    """Validate real finite numerical input and own an immutable snapshot."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return readonly(array)


def _column(value: ArrayLike, size: int, name: str) -> RealArray:
    """Broadcast constants and require one value per hypothesis."""
    array = real_array(value, name)
    if array.ndim == 0:
        array = readonly(np.full(size, float(array)))
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},) or be a scalar")
    return array


@dataclass(frozen=True, init=False, repr=False)
class EventGrouping:
    """Declare which hypotheses share a physical event.

    Notes
    -----
    Use ``from_ids`` for supplied identifiers, ``from_columns`` for source
    columns, or ``independent_rows`` when every row is an independent
    physical event. IDs are local to each sample except when
    ``MCStatistics.linked()`` explicitly links the two MC samples.

    Examples
    --------
    >>> grouping = EventGrouping.from_ids([10, 10, 20])
    >>> print(grouping)
    2 physical events in 3 hypothesis rows
    """

    ids: tuple[Hashable, ...] | None
    columns: tuple[str, ...]

    def __init__(self) -> None:
        """Require an explicit constructor declaring the statistical units."""
        raise TypeError("Use EventGrouping.from_ids, from_columns, or independent_rows")

    @classmethod
    def from_ids(cls, ids: Iterable[Hashable]) -> Self:
        """Group rows by a supplied physical-event identifier.

        Parameters
        ----------
        ids : iterable of hashable
            One nonmissing identifier per row. Repeated IDs join hypotheses.
            Compound IDs, such as ``(run, event)``, are supported.

        Returns
        -------
        EventGrouping
            An immutable snapshot preserving first-seen event order.

        Raises
        ------
        ValueError
            If an identifier is missing, nonfinite, or unhashable.
        """
        values = tuple(ids)
        for value in values:
            _validate_id(value)
        instance = cls.__new__(cls)
        object.__setattr__(instance, "ids", values)
        object.__setattr__(instance, "columns", ())
        return instance

    @classmethod
    def from_columns(cls, *names: str) -> Self:
        """Declare source columns identifying each physical event.

        Parameters
        ----------
        *names : str
            One column name, or several names forming a compound key. Columns
            contain exact identifiers, such as integers or strings. Floating-point
            identifiers are rejected to avoid loss of precision.

        Returns
        -------
        EventGrouping
            A declaration reusable across column sources. Each sample
            resolves it in its own row order without converting IDs to floats.

        Raises
        ------
        ValueError
            If names are empty, repeated, or not nonempty strings.

        Examples
        --------
        >>> EventGrouping.from_columns("run", "event")
        EventGrouping.from_columns('run', 'event')
        """
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Supply at least one nonempty event column name")
        if len(set(names)) != len(names):
            raise ValueError("Event column names must be distinct")
        instance = cls.__new__(cls)
        object.__setattr__(instance, "ids", None)
        object.__setattr__(instance, "columns", names)
        return instance

    @classmethod
    def independent_rows(cls) -> Self:
        """Declare that every row is an independent physical event.

        Returns
        -------
        EventGrouping
            A declaration reusable with samples of any length. Do not use it
            for multiple hypotheses of the same physical event.
        """
        instance = cls.__new__(cls)
        object.__setattr__(instance, "ids", None)
        object.__setattr__(instance, "columns", ())
        return instance

    def _resolve(self, source: DataSource | None) -> Self:
        """Snapshot identifiers from the current source without float conversion."""
        if not self.columns:
            return self
        values = []
        for name in self.columns:
            column = read_column(source, name)
            entries = column.tolist()
            if column.dtype.kind in "fc" or any(
                isinstance(value, float | complex | np.floating | np.complexfloating)
                for value in entries
            ):
                raise ValueError(
                    f"Event column {name!r} must contain exact integers or strings"
                )
            values.append(entries)
        ids = values[0] if len(values) == 1 else zip(*values, strict=True)
        return type(self).from_ids(ids)

    def _indices(self, size: int) -> tuple[tuple[Hashable, ...], NDArray[np.int64]]:
        """Map rows to first-seen physical events without sorting IDs."""
        if self.columns:
            raise ValueError("Resolve event column names against a source first")
        if self.ids is None:
            return tuple(range(size)), np.arange(size, dtype=np.int64)
        if len(self.ids) != size:
            raise ValueError("Event IDs must have one entry per hypothesis row")
        unique = tuple(dict.fromkeys(self.ids))
        lookup = {key: index for index, key in enumerate(unique)}
        return unique, np.asarray([lookup[key] for key in self.ids], dtype=np.int64)

    def __repr__(self) -> str:
        """Summarize grouping without listing event identifiers."""
        if self.columns:
            names = ", ".join(repr(name) for name in self.columns)
            return f"EventGrouping.from_columns({names})"
        if self.ids is None:
            return "EventGrouping.independent_rows()"
        return (
            f"EventGrouping.from_ids(rows={len(self.ids)}, events={len(set(self.ids))})"
        )

    def __str__(self) -> str:
        """Describe the physical-event declaration."""
        if self.columns:
            return f"Physical events keyed by {', '.join(self.columns)}"
        if self.ids is None:
            return "Every row is an independent physical event"
        return (
            f"{len(set(self.ids))} physical events in {len(self.ids)} hypothesis rows"
        )

    def __reduce__(self) -> tuple:
        """Restore grouping through its validating constructor."""
        if self.columns:
            return EventGrouping.from_columns, self.columns
        return (
            (EventGrouping.independent_rows, ())
            if self.ids is None
            else (EventGrouping.from_ids, (self.ids,))
        )


def _validate_id(value: Hashable) -> None:
    """Reject missing IDs, including missing entries in compound keys."""
    if isinstance(value, tuple):
        for part in value:
            _validate_id(part)
    if value is None or (
        isinstance(value, float | np.floating) and not np.isfinite(value)
    ):
        raise ValueError("Event IDs must be nonmissing and finite")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("Event IDs must be hashable") from error


@dataclass(frozen=True, init=False, repr=False, eq=False)
class Polarization:
    """Linear beam polarization magnitude and orientation.

    Notes
    -----
    Use column names, arrays/constants, or laddu expressions
    resolved by ``EventSample`` against its source. Orientation is
    in radians relative to the production plane, modulo pi. Magnitude is in
    [0, 1]; zero is allowed. Omit polarization for an unpolarized sample.
    """

    magnitude: RealArray | ld.Expr | str
    angle: RealArray | ld.Expr | str

    def __init__(
        self, *, magnitude: ColumnInput = "P", angle: ColumnInput = "Phi"
    ) -> None:
        """Describe event-wise linear photon polarization.

        Parameters
        ----------
        magnitude : str, array_like, or laddu.Expr, default='P'
            Polarization fraction, or its source column. Scalars broadcast.
        angle : str, array_like, or laddu.Expr, default='Phi'
            Orientation in radians relative to the production plane, or its source
            column. Scalars broadcast; samples canonicalize orientation modulo pi.

        Notes
        -----
        Column names and expressions are resolved by ``EventSample``. The current
        model is linear photon polarization; other polarization types are outside
        this interface.

        Examples
        --------
        >>> Polarization(magnitude=0.4, angle=0)
        Polarization(magnitude=scalar, angle=scalar)
        >>> Polarization()
        Polarization(magnitude='P', angle='Phi')
        """
        for name, value in (("magnitude", magnitude), ("angle", angle)):
            object.__setattr__(
                self,
                name,
                value if isinstance(value, str | ld.Expr) else real_array(value, name),
            )

    def __repr__(self) -> str:
        """Summarize inputs without dumping arrays or expression graphs."""

        def describe(value: RealArray | ld.Expr | str) -> str:
            """Describe the input's storage form."""
            if isinstance(value, ld.Expr):
                return "expression"
            if isinstance(value, str):
                return repr(value)
            return "scalar" if value.ndim == 0 else f"array{value.shape}"

        return f"Polarization(magnitude={describe(self.magnitude)}, angle={describe(self.angle)})"

    def __str__(self) -> str:
        """Describe the polarization convention."""
        return "Linear polarization relative to the production plane (radians)"


def _beam_columns(beam: Polarization | None) -> dict[str, ColumnInput]:
    """Name optional beam inputs for joint source resolution."""
    if beam is not None and not isinstance(beam, Polarization):
        raise TypeError("Use a Polarization object")
    return {} if beam is None else {"P": beam.magnitude, "Phi": beam.angle}


def _evaluated_beam(columns: Mapping[str, ArrayLike]) -> Polarization | None:
    """Restore an optional beam from evaluated numerical columns."""
    return (
        Polarization(magnitude=columns["P"], angle=columns["Phi"])
        if "P" in columns
        else None
    )


@dataclass(frozen=True, init=False, repr=False, eq=False)
class EventSample:
    """An immutable angular sample, weights, and physical-event grouping.

    Notes
    -----
    Construction snapshots all numerical columns. Both data and accepted MC may
    contain multiple weighted hypotheses. Weights are never rescaled per event.
    Data may have negative subtraction weights; MC weights must be nonnegative.
    The statistical model conditions on the supplied weights and calibration.
    All samples must use the same analysis frame. Accepted MC angles must be
    truth angles: this API corrects acceptance, not reconstruction migration.
    """

    costheta: RealArray
    phi: RealArray
    weights: RealArray
    events: EventGrouping
    polarization: Polarization | None
    _ids: tuple[Hashable, ...]
    _indices: NDArray[np.int64]

    def __init__(
        self,
        source: DataSource | None = None,
        *,
        costheta: ColumnInput = "costheta",
        phi: ColumnInput = "phi",
        events: EventGrouping | str | tuple[str, ...],
        weights: ColumnInput | None = None,
        polarization: Polarization | None = None,
        execution: ld.Execution | None = None,
    ) -> None:
        """Snapshot angular measurements from a column source or supplied arrays.

        Parameters
        ----------
        source : laddu.Dataset or column source, optional
            A mapping, structured NumPy array, or table supporting ``source[name]``.
            Omit it when supplying arrays directly. No file loading occurs here.
        costheta, phi : str, array_like, or laddu.Expr
            Column names (defaulting to ``"costheta"`` and ``"phi"``), numerical
            arrays, or expressions evaluated on a laddu dataset. ``costheta`` is the
            polar-angle cosine in [-1, 1]; ``phi`` is radians, modulo 2*pi.
        events : str, tuple of str, or EventGrouping
            Physical-event column name, or names forming a compound key. Explicit
            IDs use ``EventGrouping.from_ids``; ``independent_rows`` declares each
            row independent. Grouping is required; no column name is assumed.
        weights : str, array_like, or laddu.Expr, optional
            Row weights or their source column. Scalars broadcast. Omission preserves
            laddu's stored weights; other sources default to unit weights, so name
            their weight column explicitly when present.
        polarization : Polarization, optional
            Beam column names, numerical values, or laddu expressions. Omission
            denotes unpolarized measurements.
        execution : laddu.Execution, optional
            Expression evaluation defaults to double-precision JIT with no thread
            limit. Numerical arrays and ordinary column access require no execution.

        Raises
        ------
        ValueError
            If columns, grouping, angles, weights, or beam values are invalid.
        TypeError
            If a laddu expression is supplied without a laddu dataset.
        laddu.LadduError
            If laddu cannot evaluate expressions or read source columns.

        Notes
        -----
        laddu expressions and named floating-point columns evaluate together in
        one source traversal. IDs and stored weights may require additional reads.
        All input forms produce owned immutable snapshots in the supplied row order.

        Examples
        --------
        >>> rows = {"costheta": [0.3, 0.8], "phi": [0, 1],
        ...         "event": [7, 7], "weight": [0.2, 0.3]}
        >>> sample = EventSample(rows, weights="weight",
        ...     events="event")
        >>> sample.n_events, sample.weights.sum()
        (1, np.float64(0.5))
        >>> dataset = ld.Dataset.from_arrays(p4s={},
        ...     scalars={"costheta": np.array([0.5]), "phi": np.array([1.0])},
        ...     columns={"event": np.array([2**63 + 1], dtype=np.uint64)})
        >>> EventSample(dataset, events="event").n_events
        1
        """
        if isinstance(events, str):
            events = EventGrouping.from_columns(events)
        elif isinstance(events, tuple):
            events = EventGrouping.from_columns(*events)
        if not isinstance(events, EventGrouping):
            raise TypeError("events must be column names or an EventGrouping")
        inputs = {"costheta": costheta, "phi": phi, **_beam_columns(polarization)}
        if weights is not None:
            inputs["weights"] = weights
        values = evaluate_columns(source, inputs, execution)
        self._initialize(
            costheta=values["costheta"],
            phi=values["phi"],
            events=events._resolve(source),
            weights=values["weights"]
            if weights is not None
            else source_weights(source),
            polarization=_evaluated_beam(values),
        )

    def _initialize(
        self,
        *,
        costheta: ArrayLike,
        phi: ArrayLike,
        events: EventGrouping,
        weights: ArrayLike | None,
        polarization: Polarization | None,
    ) -> None:
        """Validate resolved numerical inputs and store immutable sample columns."""
        if not isinstance(events, EventGrouping):
            raise TypeError("Declare events with an EventGrouping constructor")
        cosine = real_array(costheta, "costheta")
        if cosine.ndim != 1:
            raise ValueError("costheta must be a one-dimensional array")
        size = len(cosine)
        azimuth = real_array(phi, "phi")
        if azimuth.shape != (size,):
            raise ValueError(f"phi must have shape ({size},)")
        if np.any((cosine < -1) | (cosine > 1)):
            raise ValueError("costheta must be in [-1, 1]")
        identifiers, indices = events._indices(size)
        beam = None
        if polarization is not None:
            if not isinstance(polarization, Polarization):
                raise TypeError("Use a Polarization object")
            if isinstance(polarization.magnitude, str | ld.Expr) or isinstance(
                polarization.angle, str | ld.Expr
            ):
                raise TypeError("Polarization inputs must resolve to numerical values")
            magnitude = _column(polarization.magnitude, size, "polarization magnitude")
            angle = _column(polarization.angle, size, "polarization angle")
            if np.any((magnitude < 0) | (magnitude > 1)):
                raise ValueError("Polarization magnitude must be in [0, 1]")
            beam = Polarization(magnitude=magnitude, angle=angle % np.pi)
        for name, value in (
            ("costheta", cosine),
            ("phi", readonly((azimuth + np.pi) % (2 * np.pi) - np.pi)),
            ("weights", _column(1 if weights is None else weights, size, "weights")),
            ("events", events),
            ("polarization", beam),
            ("_ids", identifiers),
            ("_indices", readonly(indices)),
        ):
            object.__setattr__(self, name, value)

    @property
    def polarized(self) -> bool:
        """Return whether the sample carries linear-polarization information."""
        return self.polarization is not None

    @property
    def n_events(self) -> int:
        """Return the number of independent physical-event groups."""
        return len(self._ids)

    def _dataset(self) -> ld.Dataset:
        """Create a scalar-only laddu dataset for angular and density evaluation."""
        scalars = {"costheta": self.costheta, "phi": self.phi}
        if self.polarization is not None:
            scalars["P"] = np.asarray(self.polarization.magnitude, dtype=np.float64)
            scalars["Phi"] = np.asarray(self.polarization.angle, dtype=np.float64)
        return ld.Dataset.from_arrays(p4s={}, scalars=scalars)

    def _aggregate(self, rows: RealArray) -> RealArray:
        """Sum hypothesis contributions before constructing event covariance."""
        grouped = np.zeros((self.n_events, *rows.shape[1:]), dtype=np.float64)
        np.add.at(grouped, self._indices, rows)
        return grouped

    def __len__(self) -> int:
        """Return the number of hypothesis rows, not physical events."""
        return len(self.costheta)

    def __repr__(self) -> str:
        """Summarize sample dimensions without printing columns."""
        return f"EventSample(rows={len(self)}, events={self.n_events}, polarized={self.polarized})"

    def __str__(self) -> str:
        """Describe sample mode and physical-event count."""
        return f"{self.n_events} {'polarized' if self.polarized else 'unpolarized'} events ({len(self)} hypothesis rows)"

    def __reduce__(self) -> tuple:
        """Revalidate and restore immutable snapshots when unpickling."""
        beam = (
            None
            if self.polarization is None
            else (
                self.polarization.magnitude,
                self.polarization.angle,
            )
        )
        return _restore_sample, (
            self.costheta,
            self.phi,
            self.weights,
            self.events,
            beam,
        )


def _restore_sample(
    costheta: RealArray,
    phi: RealArray,
    weights: RealArray,
    events: EventGrouping,
    beam: tuple[RealArray, RealArray] | None,
) -> EventSample:
    """Restore an angular sample through its public validating constructor."""
    return EventSample(
        costheta=costheta,
        phi=phi,
        weights=weights,
        events=events,
        polarization=None
        if beam is None
        else Polarization(magnitude=beam[0], angle=beam[1]),
    )
