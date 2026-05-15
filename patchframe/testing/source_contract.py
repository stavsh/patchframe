"""Source-author contract checks."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from patchframe.data.accessor import DataAccessor
from patchframe.data.array_source import ArrayDataSource
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.source import DataSource


def assert_source_contract(
    source: DataSource,
    *,
    item_id: Any,
    dim_slice: DimensionedSlice | None = None,
    compare_partial: bool = False,
) -> None:
    """Assert that a source satisfies its declared capability contract."""

    descriptor = source.describe()
    if not isinstance(descriptor, SourceDescriptor):
        raise AssertionError("source.describe() must return a SourceDescriptor.")

    accessor = DataAccessor(source_desc_id=1, item_id=item_id)
    _ = source.materialize(accessor)
    _assert_unknown_dimension_rejected(source, accessor)

    if dim_slice is not None:
        sliced = source.slice_accessor(accessor, dim_slice)
        source_value = source.materialize(sliced)
        if compare_partial:
            _assert_partial_matches_full(source, accessor, dim_slice, source_value)

    if not getattr(source, "reopenable", False):
        return

    if getattr(source, "portable", False):
        descriptor = pickle.loads(pickle.dumps(descriptor))

    reopened = type(source).open(descriptor)
    reopened_descriptor = reopened.describe()

    if reopened_descriptor.source_type != descriptor.source_type:
        raise AssertionError("Reopened source changed source_type.")
    if reopened_descriptor.source_id != descriptor.source_id:
        raise AssertionError("Reopened source changed source_id.")

    source_dimensions = getattr(source, "dimensions", None)
    reopened_dimensions = getattr(reopened, "dimensions", None)
    if source_dimensions != reopened_dimensions:
        raise AssertionError("Source dimensions changed during descriptor roundtrip.")

    _ = reopened.materialize(accessor)
    if dim_slice is not None:
        reopened_sliced = reopened.slice_accessor(accessor, dim_slice)
        reopened_value = reopened.materialize(reopened_sliced)
        _assert_values_equal(source_value, reopened_value)


def _assert_unknown_dimension_rejected(source: DataSource, accessor: DataAccessor) -> None:
    dimensions = getattr(source, "dimensions", None)
    if dimensions is None or not dimensions.names():
        return

    unknown = DimensionedSlice(dims={"__patchframe_unknown_dimension__": slice(0, 1)})
    try:
        source.slice_accessor(accessor, unknown)
    except ValueError:
        return
    raise AssertionError("source.slice_accessor() must reject unknown dimensions.")


def _assert_partial_matches_full(
    source: DataSource,
    accessor: DataAccessor,
    dim_slice: DimensionedSlice,
    partial_value: Any,
) -> None:
    if not isinstance(source, ArrayDataSource):
        raise AssertionError("compare_partial requires an ArrayDataSource.")
    if not source.supports_partial_read:
        raise AssertionError("compare_partial requires supports_partial_read=True.")

    resolved = source.resolve_slice(dim_slice)
    expected = source.apply_resolved_slice(source.read_full(accessor.item_id, accessor), resolved)
    _assert_values_equal(partial_value, expected)


def _assert_values_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return
    if left != right:
        raise AssertionError("Materialized values differ.")
