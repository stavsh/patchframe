"""Deterministic dataset factories for manual operator benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from patchframe.data.accessor import DataAccessor
from patchframe.data.dimensions import Dimension, IndexDimension, TemporalDimension
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import DataField, DimensionField, IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState


@dataclass(frozen=True, slots=True)
class BenchmarkPair:
    """Left/right datasets prepared for composition benchmarks."""

    left: Dataset
    right: Dataset


def benchmark_dimensions() -> dict[str, Dimension]:
    """Return the standard multidimensional layout used by benchmarks."""
    return {
        "time": TemporalDimension(name="time", sample_rate=16_000),
        "x": IndexDimension(name="x"),
        "y": IndexDimension(name="y"),
    }


def dimension_bindings(prefix: str = "") -> dict[str, tuple[str, str]]:
    """Return BindDimensions bindings matching ``make_multidim_dataset`` fields."""
    return {
        "time": (f"{prefix}time_start", f"{prefix}time_stop"),
        "x": (f"{prefix}x_start", f"{prefix}x_stop"),
        "y": (f"{prefix}y_start", f"{prefix}y_stop"),
    }


def make_multidim_dataset(
    rows: int,
    *,
    field_prefix: str = "",
    index_start: int = 0,
    index_field: str = "item_id",
    value_cols: int = 8,
    string_cols: int = 1,
    group_mod: int = 1024,
    include_group: bool = True,
    include_dimensions: bool = True,
    include_data: bool = True,
    null_every: int = 0,
    source_desc_id: int = 0,
) -> Dataset:
    """Create a table-heavy multidimensional dataset without materializing arrays."""
    if rows < 0:
        raise ValueError("rows must be non-negative.")
    if value_cols < 0:
        raise ValueError("value_cols must be non-negative.")

    index = pd.RangeIndex(index_start, index_start + rows, name=index_field)
    table = pd.DataFrame(index=index)
    fields: list[Any] = [IndexField(name=index_field)]

    if include_group:
        table["group"] = _nullable_int(np.arange(rows) % max(group_mod, 1), index)
        fields.append(ValueField(name="group", dtype=int))

    _add_value_columns(
        table,
        fields,
        rows=rows,
        index=index,
        field_prefix=field_prefix,
        value_cols=value_cols,
        string_cols=string_cols,
        null_every=null_every,
    )

    if include_dimensions:
        _add_dimension_columns(
            table,
            fields,
            rows=rows,
            index=index,
            field_prefix=field_prefix,
            null_every=null_every,
        )

    if include_data:
        data_name = f"{field_prefix}data"
        table[data_name] = [
            DataAccessor(source_desc_id=source_desc_id, item_id=item_id)
            for item_id in index
        ]
        fields.append(DataField(name=data_name))

    schema = Schema(fields=tuple(fields))
    schema.validate_table(table)
    return Dataset(state=DatasetState(schema=schema, table=table, couplings=CouplingSet()))


def make_index_pair(
    rows: int,
    *,
    value_cols: int = 8,
    string_cols: int = 1,
    overlap: str = "full",
    include_data: bool = True,
    include_dimensions: bool = True,
    null_every: int = 0,
    right_index_field: str = "item_id",
) -> BenchmarkPair:
    """Create datasets with prefixed fields and controlled index overlap."""
    if overlap == "full":
        right_start = 0
    elif overlap == "half":
        right_start = rows // 2
    elif overlap == "none":
        right_start = rows
    else:
        raise ValueError("overlap must be one of: full, half, none.")

    return BenchmarkPair(
        left=make_multidim_dataset(
            rows,
            field_prefix="left_",
            index_start=0,
            value_cols=value_cols,
            string_cols=string_cols,
            include_data=include_data,
            include_dimensions=include_dimensions,
            null_every=null_every,
            source_desc_id=1,
        ),
        right=make_multidim_dataset(
            rows,
            field_prefix="right_",
            index_start=right_start,
            index_field=right_index_field,
            value_cols=value_cols,
            string_cols=string_cols,
            include_data=include_data,
            include_dimensions=include_dimensions,
            null_every=null_every,
            source_desc_id=2,
        ),
    )


def make_collision_pair(
    rows: int,
    *,
    value_cols: int = 8,
    string_cols: int = 1,
    include_data: bool = True,
    include_dimensions: bool = True,
    null_every: int = 0,
) -> BenchmarkPair:
    """Create same-schema datasets for collision policy benchmarks."""
    return BenchmarkPair(
        left=make_multidim_dataset(
            rows,
            value_cols=value_cols,
            string_cols=string_cols,
            include_data=include_data,
            include_dimensions=include_dimensions,
            null_every=null_every,
            source_desc_id=1,
        ),
        right=make_multidim_dataset(
            rows,
            value_cols=value_cols,
            string_cols=string_cols,
            include_data=include_data,
            include_dimensions=include_dimensions,
            null_every=0,
            source_desc_id=2,
        ),
    )


def _add_value_columns(
    table: pd.DataFrame,
    fields: list[Any],
    *,
    rows: int,
    index: pd.Index,
    field_prefix: str,
    value_cols: int,
    string_cols: int,
    null_every: int,
) -> None:
    base = np.arange(rows)
    for i in range(value_cols):
        name = f"{field_prefix}value_{i}"
        if i < string_cols:
            values = pd.array([f"label_{value % 1024}" for value in base], dtype="string")
            _apply_nulls(values, null_every, offset=i)
            table[name] = pd.Series(values, index=index)
            fields.append(ValueField(name=name, dtype=str))
        elif i % 2 == 0:
            values = pd.array(base + i, dtype="Int64")
            _apply_nulls(values, null_every, offset=i)
            table[name] = pd.Series(values, index=index)
            fields.append(ValueField(name=name, dtype=int))
        else:
            values = pd.array((base + i) / 10.0, dtype="Float64")
            _apply_nulls(values, null_every, offset=i)
            table[name] = pd.Series(values, index=index)
            fields.append(ValueField(name=name, dtype=float))


def _add_dimension_columns(
    table: pd.DataFrame,
    fields: list[Any],
    *,
    rows: int,
    index: pd.Index,
    field_prefix: str,
    null_every: int,
) -> None:
    dims = benchmark_dimensions()
    base = np.arange(rows)
    time_start = pd.array(base / 100.0, dtype="Float64")
    time_stop = pd.array((base / 100.0) + 0.5, dtype="Float64")
    x_start = pd.array(base % 512, dtype="Int64")
    x_stop = pd.array((base % 512) + 64, dtype="Int64")
    y_start = pd.array((base * 3) % 512, dtype="Int64")
    y_stop = pd.array(((base * 3) % 512) + 64, dtype="Int64")

    columns = (
        ("time_start", time_start, dims["time"], float),
        ("time_stop", time_stop, dims["time"], float),
        ("x_start", x_start, dims["x"], int),
        ("x_stop", x_stop, dims["x"], int),
        ("y_start", y_start, dims["y"], int),
        ("y_stop", y_stop, dims["y"], int),
    )
    for offset, (suffix, values, dimension, dtype) in enumerate(columns):
        name = f"{field_prefix}{suffix}"
        _apply_nulls(values, null_every, offset=offset)
        table[name] = pd.Series(values, index=index)
        fields.append(DimensionField.from_dim(dimension, name, dtype=dtype))


def _nullable_int(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(pd.array(values, dtype="Int64"), index=index)


def _apply_nulls(values: Any, null_every: int, *, offset: int) -> None:
    if null_every <= 0 or len(values) == 0:
        return
    values[offset % null_every :: null_every] = pd.NA
