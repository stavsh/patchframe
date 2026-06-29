"""Dataset-source author contract checks.

The ``DatasetSource`` sibling of ``assert_source_contract``. It reuses the
descriptor / reopen-roundtrip discipline (the half that generalises); the slice
half is **index-range** (positional, the §8 row axis) rather than array
dimensions, so it is a sibling rather than a parametrization
(docs/design/dataset-accessor.md §1).
"""

from __future__ import annotations

import pandas as pd

from patchframe.data.dataset_accessor import DatasetAccessor
from patchframe.data.dataset_source import DatasetSource
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice


def assert_dataset_source_contract(
    source: DatasetSource,
    *,
    index_slice: DimensionedSlice | None = None,
) -> None:
    """Assert that a ``DatasetSource`` satisfies its declared contract.

    Checks: ``describe()`` shape; ``length()``/``shape()`` answered without a
    materialize mismatch (``shape[0] == length`` and both match the materialized
    row count); unknown-dimension rejection; the ``read_partial(index)``-vs-full-
    iloc agreement (§8); and, when ``reopenable``, that ``open(describe())``
    round-trips identity/dimensions and re-materializes equally.
    """
    from patchframe.dataset.dataset import Dataset

    descriptor = source.describe()
    if not isinstance(descriptor, SourceDescriptor):
        raise AssertionError("source.describe() must return a SourceDescriptor.")

    accessor = DatasetAccessor(source_desc_id=1)
    full = source.materialize(accessor)
    if not isinstance(full, Dataset):
        raise AssertionError("DatasetSource.materialize() must return a Dataset.")
    if len(full) != source.length():
        raise AssertionError("source.length() must equal the full materialized row count.")

    whole_shape = source.shape(accessor)
    if not isinstance(whole_shape, tuple) or not whole_shape:
        raise AssertionError("source.shape() must return a non-empty tuple.")
    if whole_shape[0] != source.length():
        raise AssertionError("shape(no-slice)[0] must equal length().")

    _assert_unknown_dimension_rejected(source, accessor)

    if index_slice is not None:
        sliced = source.slice_accessor(accessor, index_slice)
        partial = source.materialize(sliced)
        if not isinstance(partial, Dataset):
            raise AssertionError("Index-sliced materialize() must return a Dataset.")
        if source.shape(sliced)[0] != len(partial):
            raise AssertionError("shape(sliced)[0] must equal the sliced row count.")
        # An efficient read_partial must agree with the full-load + iloc fallback.
        expected = source.apply_index_slice(full, index_slice)
        pd.testing.assert_frame_equal(partial.table, expected.table)

    if not getattr(source, "reopenable", False):
        return

    reopened = type(source).open(descriptor)
    reopened_descriptor = reopened.describe()
    if reopened_descriptor.source_type != descriptor.source_type:
        raise AssertionError("Reopened source changed source_type.")
    if reopened_descriptor.source_id != descriptor.source_id:
        raise AssertionError("Reopened source changed source_id.")
    if source.dimensions != reopened.dimensions:
        raise AssertionError("Source dimensions changed during descriptor roundtrip.")

    reopened_full = reopened.materialize(accessor)
    pd.testing.assert_frame_equal(reopened_full.table, full.table)


def _assert_unknown_dimension_rejected(source: DatasetSource, accessor: DatasetAccessor) -> None:
    if not source.dimensions.names():
        return
    unknown = DimensionedSlice(dims={"__patchframe_unknown_dimension__": slice(0, 1)})
    try:
        source.slice_accessor(accessor, unknown)
    except ValueError:
        return
    raise AssertionError("source.slice_accessor() must reject unknown dimensions.")
