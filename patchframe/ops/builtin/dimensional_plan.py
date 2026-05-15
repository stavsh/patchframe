"""Plan operators for dimensional slice expansion."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from patchframe.data.dimensioned_slice_array import DimensionedSliceArray
from patchframe.data.windows import AxisWindow
from patchframe.dataset.couplings import BindDimensions
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    DimensionedSliceField,
    DimensionField,
    IndexField,
    ValueField,
)
from patchframe.dataset.schema import Schema
from patchframe.ops.base import PlanOperator

PLAN_INDEX_NAME = "plan_id"
SOURCE_INDEX_FIELD = "source_index"
PLAN_SLICE_FIELD = "slice"
PLAN_METADATA_KEY = "patchframe.plan"


class make_dimensional_plan(PlanOperator):
    """Build a plan dataset by tiling bounded dimensional slices.

    Exactly one extent input is required:

    - ``field``: a single ``DimensionedSliceField`` column. Null rows are
      skipped.
    - ``bindings``: one or more ``DimensionField`` bindings in the same format
      accepted by ``bind_dimensions``. Null values are rejected because partial
      multi-column bounds are ambiguous.
    """

    plan_index_name = PLAN_INDEX_NAME
    required_plan_fields = (SOURCE_INDEX_FIELD, PLAN_SLICE_FIELD)

    def __call__(
        self,
        dataset: Dataset,
        *,
        windows: Mapping[str, AxisWindow],
        field: str | None = None,
        bindings: Any | None = None,
        source_index_field: str = SOURCE_INDEX_FIELD,
        slice_field: str = PLAN_SLICE_FIELD,
        plan_index_name: str = PLAN_INDEX_NAME,
    ) -> Dataset:
        if (field is None) == (bindings is None):
            raise ValueError("make_dimensional_plan requires exactly one of field or bindings.")
        if not windows:
            raise ValueError("make_dimensional_plan requires at least one window.")
        _warn_if_planning_over_plan(dataset)

        extent_array = (
            _slice_array_from_field(dataset, field)
            if field is not None
            else _slice_array_from_bindings(dataset, bindings)
        )
        parent_positions, slices = extent_array.explode_windows(windows)
        table = _plan_table(
            source_index=dataset.table.index.to_numpy(dtype=object)[parent_positions],
            slices=slices,
            source_index_field=source_index_field,
            slice_field=slice_field,
            plan_index_name=plan_index_name,
        )
        schema = Schema(
            fields=(
                IndexField(name=plan_index_name),
                ValueField(name=source_index_field, nullable=False),
                DimensionedSliceField(name=slice_field, nullable=False),
            )
        )
        return self.build_plan_dataset(
            schema=schema,
            table=table,
            sources=tuple(dataset.sources),
            source_manager=dataset.source_manager,
            metadata=_dimensional_plan_metadata(
                dataset,
                extent_kind="field" if field is not None else "bindings",
                extent_field=field,
                source_index_field=source_index_field,
                slice_field=slice_field,
                plan_index_name=plan_index_name,
                window_dimensions=tuple(windows),
            ),
            plan_index_name=plan_index_name,
            required_plan_fields=(source_index_field, slice_field),
        )


def _plan_table(
    *,
    source_index: np.ndarray,
    slices: DimensionedSliceArray,
    source_index_field: str,
    slice_field: str,
    plan_index_name: str,
) -> pd.DataFrame:
    index = pd.RangeIndex(len(source_index), name=plan_index_name)
    table = pd.DataFrame({source_index_field: source_index}, index=index)
    table[slice_field] = pd.Series(slices, index=index)
    return table


def _slice_array_from_field(dataset: Dataset, field: str) -> DimensionedSliceArray:
    field_def = dataset.schema.get(field)
    series = dataset.table[field]

    if isinstance(field_def, DimensionedSliceField):
        if isinstance(series.array, DimensionedSliceArray):
            return series.array
        return DimensionedSliceArray._from_sequence(series.to_numpy(dtype=object))

    raise TypeError(
        "make_dimensional_plan field must be a DimensionedSliceField; "
        f"got {type(field_def).__name__}."
    )


def _slice_array_from_bindings(dataset: Dataset, bindings: Any) -> DimensionedSliceArray:
    dimension_names, normalized = BindDimensions._normalize_bindings(bindings)
    dimensions = []
    selector_columns = []

    for dimension_name, binding in zip(dimension_names, normalized, strict=True):
        if not binding:
            raise ValueError("make_dimensional_plan bindings must not be empty.")

        field_defs = []
        columns = []
        for ref in binding:
            field_def = dataset.schema.get(ref.name)
            if not isinstance(field_def, DimensionField):
                raise TypeError(
                    f"make_dimensional_plan binding field {ref.name!r} "
                    "is not a DimensionField."
                )
            column = dataset.table[ref.name]
            if column.isna().any():
                raise ValueError(
                    "make_dimensional_plan does not allow null values in multi-field "
                    f"bindings; field {ref.name!r} contains nulls."
                )
            field_defs.append(field_def)
            columns.append(column.to_numpy(copy=True))

        dimension = field_defs[0].dimension
        if any(field_def.dimension != dimension for field_def in field_defs[1:]):
            names = [field_def.name for field_def in field_defs]
            raise ValueError(f"make_dimensional_plan bindings span dimensions: {names}")
        if dimension_name is not None and dimension.name != dimension_name:
            raise ValueError(
                f"make_dimensional_plan mapping key {dimension_name!r} does not match "
                f"DimensionField dimension {dimension.name!r}."
            )
        dimensions.append(dimension)
        selector_columns.append(tuple(columns))

    return DimensionedSliceArray.from_columns(
        dimensions=tuple(dimensions),
        selector_columns=tuple(selector_columns),
    )


def _warn_if_planning_over_plan(dataset: Dataset) -> None:
    if PLAN_METADATA_KEY not in dataset.state.metadata:
        return
    warnings.warn(
        "make_dimensional_plan was called on a dataset already marked as a plan. "
        "The new plan's source_index values will point to the input plan rows, "
        "not to the original source dataset. Plan refinement semantics are not "
        "implemented yet.",
        UserWarning,
        stacklevel=3,
    )


def _dimensional_plan_metadata(
    dataset: Dataset,
    *,
    extent_kind: str,
    extent_field: str | None,
    source_index_field: str,
    slice_field: str,
    plan_index_name: str,
    window_dimensions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        PLAN_METADATA_KEY: {
            "type": "dimensional",
            "operator": "make_dimensional_plan",
            "source_index_field": source_index_field,
            "slice_field": slice_field,
            "plan_index_name": plan_index_name,
            "input_index_name": dataset.table.index.name,
            "extent_kind": extent_kind,
            "extent_field": extent_field,
            "window_dimensions": window_dimensions,
        }
    }
