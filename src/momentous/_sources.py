"""Resolve numerical columns and expressions behind a common source interface."""

from collections.abc import Mapping
from typing import Protocol

import laddu as ld
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ColumnSource(Protocol):
    """Describe mappings, structured arrays, and tables with named column access."""

    def __getitem__(self, name: str, /) -> ArrayLike:
        """Return one array-like column in the source's current row order."""
        ...


type DataSource = ld.Dataset | ColumnSource
type ColumnInput = str | ld.Expr | ArrayLike


def execution_or_default(execution: ld.Execution | None) -> ld.Execution:
    """Use double-precision JIT evaluation without imposing a thread count."""
    return ld.Execution("jit", precision="f64") if execution is None else execution


def read_column(source: DataSource | None, name: str) -> NDArray[np.generic]:
    """Read a named one-dimensional column without changing its dtype."""
    if source is None:
        raise ValueError("Named columns require a source; otherwise supply arrays")
    if isinstance(source, ld.Dataset):
        values = source.column(name)
    else:
        try:
            values = source[name]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Cannot read source column {name!r}") from error
    column = np.asarray(values)
    if column.ndim != 1:
        raise ValueError(f"Source column {name!r} must be one-dimensional")
    return column


def evaluate_columns(
    source: DataSource | None,
    columns: Mapping[str, ColumnInput],
    execution: ld.Execution | None,
) -> dict[str, ArrayLike]:
    """Resolve names and arrays, evaluating laddu expressions together when present."""
    expressions: dict[str, ld.Expr] = {}
    result: dict[str, ArrayLike] = {}
    scalar_names = (
        set(source.scalar_names()) if isinstance(source, ld.Dataset) else set()
    )
    for name, value in columns.items():
        if isinstance(value, ld.Expr):
            if not isinstance(source, ld.Dataset):
                raise TypeError(
                    "laddu expressions require a laddu dataset; "
                    "use column names or arrays with other sources"
                )
            expressions[name] = value
        elif isinstance(value, str):
            if value in scalar_names:
                expressions[name] = ld.scalar(value)
            else:
                result[name] = read_column(source, value)
        else:
            result[name] = value
    if expressions:
        assert isinstance(source, ld.Dataset)
        evaluated = source.evaluate(
            expressions, execution=execution_or_default(execution)
        )
        for name, value in evaluated.items():
            values = np.asarray(value)
            if np.any(values.imag != 0):
                raise ValueError(f"{name} expressions must evaluate to real values")
            result[name] = np.asarray(values.real, dtype=np.float64)
    return result


def source_weights(source: DataSource | None) -> ArrayLike | None:
    """Preserve laddu's stored weights; other sources use explicit weights or units."""
    return source.weights() if isinstance(source, ld.Dataset) else None
