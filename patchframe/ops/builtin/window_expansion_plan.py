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
)
from patchframe.ops.base import OperatorCall, PlanOperator
from patchframe.ops.signature import (
    DatasetInput,
    FieldOutput,
    FieldReturn,
    ParamInput,
)
from patchframe.ops.transitions import PerRowIndependence
from patchframe.ops.builtin.assign import assign
from patchframe.ops.builtin.make_plan import make_plan

PLAN_INDEX_NAME = "plan_id"
SOURCE_INDEX_FIELD = "source_index"
PLAN_SLICE_FIELD = "slice"
PLAN_METADATA_KEY = "patchframe.plan"


class window_expansion_plan(PlanOperator):
    """Build a plan dataset by tiling bounded dimensional slices.

    Exactly one extent input is required:

    - ``field``: a single ``DimensionedSliceField`` column. Null rows are
      skipped.
    - ``bindings``: one or more ``DimensionField`` bindings in the same format
      accepted by ``compose_slice``. Null values are rejected because partial
      multi-column bounds are ambiguous.
    """

    plan_index_name = PLAN_INDEX_NAME
    required_plan_fields = (SOURCE_INDEX_FIELD, PLAN_SLICE_FIELD)
    per_row_independent = PerRowIndependence.INDEPENDENT
    # Operand-dispatch law (lazy-duality-plan.md, 2026-06-11): a FieldHandle
    # input — anywhere, including nested in `bindings` — selects the lazy arm
    # (a bundle-lift, like explode); there is NO eager handle resolution. Eager
    # calls pass the dataset plus field *names*. `field`/`bindings` are
    # ParamInput: replay data resolved against the (possibly deferred) source
    # at run time, never handle operands — the set_index/`on=` pattern.
    dataset = DatasetInput()
    field = ParamInput(default=None)
    bindings = ParamInput(default=None)
    out = FieldOutput()
    returns = FieldReturn()

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
        out: str | None = None,
    ) -> Dataset:
        # __call__ stays only to document the public signature and apply the
        # defaults; `out` names the deferred result cell on the lazy arm and is
        # ignored by the eager path.
        return PlanOperator.__call__(
            self,
            dataset,
            windows=windows,
            field=field,
            bindings=bindings,
            source_index_field=source_index_field,
            slice_field=slice_field,
            plan_index_name=plan_index_name,
            out=out,
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        dataset = call.datasets[0]
        kwargs = dict(call.kwargs)
        windows = kwargs["windows"]
        field = kwargs["field"]
        bindings = kwargs["bindings"]
        source_index_field = kwargs["source_index_field"]
        slice_field = kwargs["slice_field"]
        plan_index_name = kwargs["plan_index_name"]

        if (field is None) == (bindings is None):
            raise ValueError("window_expansion_plan requires exactly one of field or bindings.")
        if not windows:
            raise ValueError("window_expansion_plan requires at least one window.")
        _warn_if_planning_over_plan(dataset)

        if field is not None:
            extent_array = _slice_array_from_field(dataset, field)
        else:
            extent_array = _slice_array_from_bindings(dataset, bindings)
        parent_positions, slices = extent_array.explode_windows(windows)
        metadata = _window_expansion_plan_metadata(
            dataset,
            extent_kind="field" if field is not None else "bindings",
            extent_field=field,
            source_index_field=source_index_field,
            slice_field=slice_field,
            plan_index_name=plan_index_name,
            window_dimensions=tuple(windows),
        )
        plan = make_plan(
            dataset,
            dataset.table.index.to_numpy(dtype=object)[parent_positions],
            source_index_field=source_index_field,
            plan_index_name=plan_index_name,
            metadata=metadata,
        )
        result = assign(
            plan,
            **{
                slice_field: (
                    DimensionedSliceField(name=slice_field, nullable=False),
                    slices,
                )
            },
        )
        self.validate_plan_schema(
            result.schema,
            result.table,
            plan_index_name=plan_index_name,
            required_plan_fields=(source_index_field, slice_field),
        )
        return result

    def plan_validation_options(self, call: OperatorCall) -> dict[str, Any]:
        kwargs = dict(call.kwargs)
        return {
            "plan_index_name": kwargs.get("plan_index_name", PLAN_INDEX_NAME),
            "required_plan_fields": (
                kwargs.get("source_index_field", SOURCE_INDEX_FIELD),
                kwargs.get("slice_field", PLAN_SLICE_FIELD),
            ),
        }


def _slice_array_from_field(dataset: Dataset, field: str) -> DimensionedSliceArray:
    field_def = dataset.schema.get(field)
    series = dataset.table[field]

    if isinstance(field_def, DimensionedSliceField):
        if isinstance(series.array, DimensionedSliceArray):
            return series.array
        return DimensionedSliceArray._from_sequence(series.to_numpy(dtype=object))

    raise TypeError(
        "window_expansion_plan field must be a DimensionedSliceField; "
        f"got {type(field_def).__name__}."
    )


def _slice_array_from_bindings(dataset: Dataset, bindings: Any) -> DimensionedSliceArray:
    dimension_names, normalized = BindDimensions._normalize_bindings(bindings)
    dimensions = []
    selector_columns = []

    for dimension_name, binding in zip(dimension_names, normalized, strict=True):
        if not binding:
            raise ValueError("window_expansion_plan bindings must not be empty.")

        field_defs = []
        columns = []
        for ref in binding:
            field_def = dataset.schema.get(ref.name)
            if not isinstance(field_def, DimensionField):
                raise TypeError(
                    f"window_expansion_plan binding field {ref.name!r} "
                    "is not a DimensionField."
                )
            column = dataset.table[ref.name]
            if column.isna().any():
                raise ValueError(
                    "window_expansion_plan does not allow null values in multi-field "
                    f"bindings; field {ref.name!r} contains nulls."
                )
            field_defs.append(field_def)
            columns.append(column.to_numpy(copy=True))

        dimension = field_defs[0].dimension
        if any(field_def.dimension != dimension for field_def in field_defs[1:]):
            names = [field_def.name for field_def in field_defs]
            raise ValueError(f"window_expansion_plan bindings span dimensions: {names}")
        if dimension_name is not None and dimension.name != dimension_name:
            raise ValueError(
                f"window_expansion_plan mapping key {dimension_name!r} does not match "
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
        "window_expansion_plan was called on a dataset already marked as a plan. "
        "The new plan's source_index values will point to the input plan rows, "
        "not to the original source dataset. Plan refinement semantics are not "
        "implemented yet.",
        UserWarning,
        stacklevel=3,
    )


def _window_expansion_plan_metadata(
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
            "type": "window_expansion",
            "operator": "window_expansion_plan",
            "source_index_field": source_index_field,
            "slice_field": slice_field,
            "plan_index_name": plan_index_name,
            "input_index_name": dataset.table.index.name,
            "extent_kind": extent_kind,
            "extent_field": extent_field,
            "window_dimensions": window_dimensions,
        }
    }
