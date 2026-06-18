"""Small smoke coverage for benchmark factories and non-IO benchmark paths."""

from __future__ import annotations

from benchmarks.factories import dimension_bindings, make_index_pair, make_multidim_dataset
from patchframe.data.accessor import DataAccessor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensioned_slice_array import DimensionedSliceArray
from patchframe.data.windows import AxisWindow
from patchframe.dataset.field_composition import ColumnCollisionStrategy
from patchframe.ops.builtin.compose_slice import compose_slice
from patchframe.ops.builtin.slice_data import slice_data
from patchframe.ops.builtin.concat import concat_columns, concat_rows
from patchframe.ops.builtin.consume import consume
from patchframe.ops.builtin.join import join
from patchframe.ops.builtin.merge import merge
from patchframe.ops.builtin.window_expansion_plan import window_expansion_plan


def test_operator_benchmark_factories_cover_concat_join_merge_paths():
    row_pair = make_index_pair(1_000, value_cols=4, string_cols=0, overlap="none")
    row_result = concat_rows(row_pair.left, row_pair.right)
    assert row_result.table.shape[0] == 2_000

    column_pair = make_index_pair(
        1_000,
        value_cols=4,
        string_cols=0,
        overlap="full",
        right_index_field="right_id",
    )
    keep_left = ColumnCollisionStrategy(mode="keep", side="left")
    column_result = concat_columns(column_pair.left, column_pair.right, collision=keep_left)
    assert column_result.table.shape[0] == 1_000

    merge_pair = make_index_pair(1_000, value_cols=4, string_cols=0, overlap="full")
    plan = join(merge_pair.left, merge_pair.right)
    merge_result = merge(merge_pair.left, merge_pair.right, plan, collision=keep_left)

    assert plan.table.shape[0] == 1_000
    assert merge_result.table.shape[0] == 1_000
    assert {"left_index", "right_index"}.issubset(merge_result.table.columns)


def test_consume_bind_dimensions_keeps_dimensioned_slice_array_columnar():
    ds = make_multidim_dataset(
        1_000,
        value_cols=2,
        string_cols=0,
        include_data=False,
    )
    ds = compose_slice(ds, slice_field="slice", bindings=dimension_bindings())

    result = consume(ds, "slice")

    assert isinstance(result.table["slice"].array, DimensionedSliceArray)
    scalar = result.table["slice"].iloc[0]
    assert isinstance(scalar, DimensionedSlice)
    assert set(scalar.dims) == {"time", "x", "y"}


def test_window_expansion_plan_benchmark_path_expands_dimension_bindings():
    ds = make_multidim_dataset(
        1_000,
        value_cols=2,
        string_cols=0,
        include_data=False,
    )

    plan = window_expansion_plan(
        ds,
        bindings={"x": dimension_bindings()["x"], "y": dimension_bindings()["y"]},
        windows={"x": AxisWindow(32, 32), "y": AxisWindow(32, 32)},
    )

    assert plan.table.shape[0] == 4_000
    assert plan.table["source_index"].iloc[0] == 0
    assert isinstance(plan.table["slice"].array, DimensionedSliceArray)


def test_consume_chained_bind_dimensions_keeps_dimensioned_slice_array_columnar():
    ds = make_multidim_dataset(
        1_000,
        value_cols=2,
        string_cols=0,
        include_data=False,
    )
    bindings = dimension_bindings()
    ds = compose_slice(ds, slice_field="slice", bindings={"time": bindings["time"]})
    ds = compose_slice(
        ds,
        slice_field="slice",
        bindings={"x": bindings["x"], "y": bindings["y"]},
    )

    result = consume(ds, "slice")

    assert isinstance(result.table["slice"].array, DimensionedSliceArray)
    scalar = result.table["slice"].iloc[0]
    assert isinstance(scalar, DimensionedSlice)
    assert set(scalar.dims) == {"time", "x", "y"}


def test_eager_slice_data_attaches_slice_without_decoding():
    # compose_slice and slice_data are eager on a Dataset operand (metadata, not
    # IO): the slice is composed and attached to the data accessors now — but
    # the accessors stay lazy, because the *decode* is materialize's job and is
    # still deferred. So a sliced-but-undecoded accessor results, no IO.
    ds = make_multidim_dataset(1_000, value_cols=0, string_cols=0, include_data=True)
    ds = compose_slice(ds, slice_field="slice", bindings=dimension_bindings())
    ds = slice_data(ds, slice_field="slice", data_field="data")

    accessor = ds.table["data"].iloc[0]
    assert isinstance(accessor, DataAccessor)
    assert accessor.dimensioned_slice is not None
    assert set(accessor.dimensioned_slice.dims) == {"time", "x", "y"}
