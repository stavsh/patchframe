"""Tests for add_column, BindSlice/Materialize couplings, and the consume operator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import patchframe as pf
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensioned_slice_array import DimensionedSliceArray
from patchframe.data.dimensions import Dimensions, IndexDimension, TemporalDimension
from patchframe.data.manager import reset_default_manager
from patchframe.dataset.couplings import BindDimensions, BindSlice, FieldRef, Materialize
from patchframe.dataset.fields import (
    DimensionedSliceField,
    DimensionField,
    ValueField,
)
from patchframe.ops.builtin.add_column import add_column
from patchframe.ops.builtin.consume import consume
from patchframe.testing.mock import make_mock_dataset


@pytest.fixture(autouse=True)
def fresh_manager():
    reset_default_manager()


# ---------------------------------------------------------------------------
# add_column
# ---------------------------------------------------------------------------

class TestAddColumn:
    def test_adds_field_to_schema(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 10))
        field_def = ValueField(name="label", dtype=str)
        ds2 = add_column(ds, field_def, ["cat", "dog"])
        assert ds2.schema.has("label")
        assert isinstance(ds2.schema.get("label"), ValueField)

    def test_adds_column_to_table(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 10))
        ds2 = add_column(ds, ValueField(name="score", dtype=float), [1.0, 2.0])
        assert "score" in ds2.table.columns
        assert list(ds2.table["score"]) == [1.0, 2.0]

    def test_original_dataset_unchanged(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        add_column(ds, ValueField(name="extra", dtype=float), [0.0])
        assert "extra" not in ds.table.columns
        assert not ds.schema.has("extra")

    def test_adds_couplings(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        slice_field = DimensionedSliceField(name="clip")
        coupling = BindSlice(slice_field="clip", data_field="data")
        ds2 = add_column(ds, slice_field, [dim.spec(0, 5)], couplings=(coupling,))
        assert len(ds2.couplings.couplings) == 1
        assert isinstance(ds2.couplings.couplings[0], BindSlice)

    def test_no_couplings_by_default(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        ds2 = add_column(ds, ValueField(name="x2", dtype=float), [0.0])
        assert len(ds2.couplings.couplings) == 0

    def test_existing_couplings_preserved(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        c1 = BindSlice(slice_field="s1", data_field="data")
        ds = add_column(ds, DimensionedSliceField(name="s1"), [dim.spec(0, 5)], couplings=(c1,))
        c2 = BindSlice(slice_field="s2", data_field="data")
        ds2 = add_column(ds, DimensionedSliceField(name="s2"), [dim.spec(0, 5)], couplings=(c2,))
        assert len(ds2.couplings.couplings) == 2


# ---------------------------------------------------------------------------
# Dataset.__getitem__ — column access
# ---------------------------------------------------------------------------

class TestGetitemColumnAccess:
    def test_returns_series_for_column_name(self):
        import pandas as pd
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 10))
        result = ds["data"]
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_series_has_field_attr(self):
        from patchframe.dataset.extension import _FIELD
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        series = ds["data"]
        assert series.attrs.get(_FIELD) is ds.schema.get("data")


# ---------------------------------------------------------------------------
# Dataset.__getitem__ — row access with BindSlice coupling
# ---------------------------------------------------------------------------

class TestGetitemRowAccess:
    def _make_sliced_dataset(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        slices = [dim.spec(0, 20), dim.spec(10, 50)]
        coupling = BindSlice(slice_field="clip", data_field="data")
        return add_column(ds, DimensionedSliceField(name="clip"), slices, couplings=(coupling,))

    def test_returns_dict(self):
        ds = self._make_sliced_dataset()
        row = ds["a"]
        assert isinstance(row, dict)

    def test_index_field_included_in_row(self):
        ds = self._make_sliced_dataset()
        row = ds["a"]
        assert "item_id" in row
        assert row["item_id"] == "a"

    def test_data_accessor_has_slice_applied(self):
        from patchframe.data.accessor import DataAccessor
        from patchframe.data.dimensioned_slice import DimensionedSlice
        ds = self._make_sliced_dataset()
        row = ds["a"]
        acc = row["data"]
        assert isinstance(acc, DataAccessor)
        assert isinstance(acc.dimensioned_slice, DimensionedSlice)

    def test_materialized_shape_matches_per_row_slice(self):
        ds = self._make_sliced_dataset()
        assert ds["a"]["data"].materialize().shape == (20,)
        assert ds["b"]["data"].materialize().shape == (40,)

    def test_no_coupling_row_returns_full_accessor(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 50), seed=0)
        row = ds["a"]
        assert row["data"].materialize().shape == (50,)

    def test_temporal_dimension_coupling(self):
        dim = TemporalDimension(name="time", sample_rate=16000)
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0.0, 1.0), seed=0)
        coupling = BindSlice(slice_field="window", data_field="data")
        ds2 = add_column(ds, DimensionedSliceField(name="window"),
                         [dim.spec(0.0, 0.5)], couplings=(coupling,))
        arr = ds2["a"]["data"].materialize()
        assert arr.shape == (8000,)

    def test_chain_order_preserved(self):
        """Two BindSlice couplings on the same data column form a chain in declaration order."""
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        c1 = BindSlice(slice_field="s1", data_field="data")
        c2 = BindSlice(slice_field="s2", data_field="data")
        ds2 = add_column(ds, DimensionedSliceField(name="s1"), [dim.spec(0, 60)], couplings=(c1,))
        ds3 = add_column(ds2, DimensionedSliceField(name="s2"), [dim.spec(10, 30)], couplings=(c2,))
        arr = ds3["a"]["data"].materialize()
        assert arr.shape == (20,)


# ---------------------------------------------------------------------------
# BindDimensions
# ---------------------------------------------------------------------------

class TestBindDimensions:
    def test_row_access_derives_slice_from_mapping_bindings(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField(name="start", dimension=dim, dtype=int), [0, 10])
        ds = add_column(ds, DimensionField(name="end", dimension=dim, dtype=int), [20, 50])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None, None],
            couplings=(
                BindDimensions(slice_field="clip", bindings={"x": ("start", "end")}),
                BindSlice("clip", "data"),
            ),
        )

        row = ds["a"]

        assert isinstance(row["clip"], DimensionedSlice)
        assert row["clip"].dims["x"] == slice(0, 20)
        assert row["data"].materialize().shape == (20,)

    def test_consume_runs_dimension_slice_upstream_of_bind_slice(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField(name="start", dimension=dim, dtype=int), [0, 10])
        ds = add_column(ds, DimensionField(name="end", dimension=dim, dtype=int), [20, 50])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None, None],
            couplings=(
                BindDimensions(slice_field="clip", bindings={"x": ("start", "end")}),
                BindSlice("clip", "data"),
            ),
        )

        ds2 = consume(ds, "data")

        assert isinstance(ds2.table["clip"].iloc[0], DimensionedSlice)
        assert ds2.table["clip"].iloc[0].dims["x"] == slice(0, 20)
        assert ds2.table["data"].iloc[0].materialize().shape == (20,)
        assert ds2.table["data"].iloc[1].materialize().shape == (40,)

    def test_consume_slice_field_returns_columnar_slice_array(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField(name="start", dimension=dim, dtype=int), [0, 10])
        ds = add_column(ds, DimensionField(name="end", dimension=dim, dtype=int), [20, 50])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None, None],
            couplings=(BindDimensions(slice_field="clip", bindings={"x": ("start", "end")}),),
        )

        ds2 = consume(ds, "clip")

        assert isinstance(ds2.table["clip"].array, DimensionedSliceArray)
        assert isinstance(ds2.table["clip"].iloc[0], DimensionedSlice)
        assert ds2.table["clip"].iloc[0].dims["x"] == slice(0, 20)

    def test_consume_slice_field_marks_missing_selector_rows_null(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField(name="start", dimension=dim), [0, None])
        ds = add_column(ds, DimensionField(name="end", dimension=dim), [20, 50])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None, None],
            couplings=(BindDimensions(slice_field="clip", bindings={"x": ("start", "end")}),),
        )

        ds2 = consume(ds, "clip")

        assert ds2.table["clip"].isna().tolist() == [False, True]
        assert ds2.table["clip"].iloc[0].dims["x"] == slice(0, 20)
        assert pd.isna(ds2.table["clip"].iloc[1])

    def test_dimensioned_slice_array_accepts_empty_and_all_null_sequences(self):
        empty = DimensionedSliceArray._from_sequence([])
        nulls = DimensionedSliceArray._from_sequence([None, np.nan])

        assert len(empty) == 0
        assert nulls.isna().tolist() == [True, True]
        assert pd.isna(nulls[0])

    def test_tuple_bindings_infer_dimensions_from_fields(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        extent = DimensionedSlice(dims={"x": slice(0, 100), "y": slice(0, 50)})
        ds = make_mock_dataset(["a"], Dimensions((x, y)), extent, seed=0)
        ds = add_column(ds, DimensionField(name="x0", dimension=x, dtype=int), [10])
        ds = add_column(ds, DimensionField(name="x1", dimension=x, dtype=int), [30])
        ds = add_column(ds, DimensionField(name="y0", dimension=y, dtype=int), [5])
        ds = add_column(ds, DimensionField(name="y1", dimension=y, dtype=int), [15])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None],
            couplings=(
                BindDimensions(slice_field="clip", bindings=(("x0", "x1"), ("y0", "y1"))),
                BindSlice("clip", "data"),
            ),
        )

        row = ds["a"]

        assert row["clip"].dims == {"x": slice(10, 30), "y": slice(5, 15)}
        assert row["data"].materialize().shape == (20, 10)

    def test_chained_dimension_bindings_stay_columnar_on_consume(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        extent = DimensionedSlice(dims={"x": slice(0, 100), "y": slice(0, 50)})
        ds = make_mock_dataset(["a"], Dimensions((x, y)), extent, seed=0)
        ds = add_column(ds, DimensionField(name="x0", dimension=x, dtype=int), [10])
        ds = add_column(ds, DimensionField(name="x1", dimension=x, dtype=int), [30])
        ds = add_column(ds, DimensionField(name="y0", dimension=y, dtype=int), [5])
        ds = add_column(ds, DimensionField(name="y1", dimension=y, dtype=int), [15])
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None],
            couplings=(
                BindDimensions(slice_field="clip", bindings=(("x0", "x1"),)),
                BindDimensions(slice_field="clip", bindings=(("y0", "y1"),)),
            ),
        )

        ds2 = consume(ds, "clip")

        assert isinstance(ds2.table["clip"].array, DimensionedSliceArray)
        assert ds2.table["clip"].iloc[0].dims == {
            "x": slice(10, 30),
            "y": slice(5, 15),
        }

    def test_concat_rejects_incompatible_non_null_slice_arrays(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        x_array = DimensionedSliceArray.from_columns(
            dimensions=(x,),
            selector_columns=(([0], [10]),),
        )
        y_array = DimensionedSliceArray.from_columns(
            dimensions=(y,),
            selector_columns=(([0], [10]),),
        )

        with pytest.raises(TypeError, match="incompatible"):
            DimensionedSliceArray._concat_same_type([x_array, y_array])

    def test_concat_allows_incompatible_all_null_slice_array(self):
        x = IndexDimension(name="x")
        x_array = DimensionedSliceArray.from_columns(
            dimensions=(x,),
            selector_columns=(([0], [10]),),
        )
        null_array = DimensionedSliceArray._from_sequence([None])

        combined = DimensionedSliceArray._concat_same_type([x_array, null_array])

        assert isinstance(combined, DimensionedSliceArray)
        assert combined.isna().tolist() == [False, True]
        assert combined[0].dims["x"] == slice(0, 10)
        assert pd.isna(combined[1])

    def test_single_value_dimension_binding_is_not_interval_specific(self):
        from dataclasses import dataclass
        from typing import Any

        from patchframe.data.dimensions import Dimension, DimensionIndex

        @dataclass(frozen=True, slots=True)
        class LabelDimension(Dimension):
            def spec(self, *values: Any) -> DimensionedSlice:
                if len(values) != 1:
                    raise ValueError("LabelDimension expects exactly one selector value.")
                return DimensionedSlice(dims={self.name: values[0]})

            def to_index(self, value: Any) -> DimensionIndex:
                return DimensionIndex(name=self.name, value=value)

        x = IndexDimension(name="x")
        label = LabelDimension(name="label")
        ds = make_mock_dataset(["a"], Dimensions((x,)), x.spec(0, 100), seed=0)
        ds = add_column(
            ds,
            DimensionField(name="label_value", dimension=label, dtype=str),
            ["speech"],
        )
        ds = add_column(
            ds,
            DimensionedSliceField(name="clip"),
            [None],
            couplings=(BindDimensions(slice_field="clip", bindings={"label": "label_value"}),),
        )

        row = ds["a"]

        assert row["clip"].dims == {"label": "speech"}


# ---------------------------------------------------------------------------
# consume — bulk materialization
# ---------------------------------------------------------------------------

class TestConsume:
    def _make_sliced_dataset(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        slices = [dim.spec(0, 20), dim.spec(10, 50)]
        coupling = BindSlice(slice_field="clip", data_field="data")
        return add_column(ds, DimensionedSliceField(name="clip"), slices, couplings=(coupling,))

    def test_consume_column_applies_chain(self):
        from patchframe.data.accessor import DataAccessor
        from patchframe.data.dimensioned_slice import DimensionedSlice
        ds = self._make_sliced_dataset()
        ds2 = consume(ds, "data")
        # After consume, "data" column holds sliced accessors (BindSlice ran)
        acc = ds2.table["data"].iloc[0]
        assert isinstance(acc, DataAccessor)
        assert isinstance(acc.dimensioned_slice, DimensionedSlice)
        # And materialization respects the slice
        assert acc.materialize().shape == (20,)

    def test_consume_with_materialize_coupling_yields_arrays(self):
        ds = self._make_sliced_dataset()
        materialize = Materialize(field="data")
        ds = ds.replace_state(couplings=ds.couplings.add(materialize))
        ds2 = consume(ds, "data")
        assert isinstance(ds2.table["data"].iloc[0], np.ndarray)
        assert ds2.table["data"].iloc[0].shape == (20,)
        assert ds2.table["data"].iloc[1].shape == (40,)

    def test_consume_partial_via_coupling_target(self):
        """Targeting BindSlice runs only up to it — Materialize is skipped."""
        from patchframe.data.accessor import DataAccessor
        ds = self._make_sliced_dataset()
        bind = ds.couplings.couplings[0]
        materialize = Materialize(field="data")
        ds = ds.replace_state(couplings=ds.couplings.add(materialize))
        ds2 = consume(ds, bind)
        # After partial consume, "data" is a sliced accessor, NOT an array
        assert isinstance(ds2.table["data"].iloc[0], DataAccessor)

    def test_consume_unknown_column_raises(self):
        ds = self._make_sliced_dataset()
        with pytest.raises(ValueError, match="No couplings produce"):
            consume(ds, "nonexistent")

    def test_consume_preserves_schema_and_couplings(self):
        ds = self._make_sliced_dataset()
        ds2 = consume(ds, "data")
        assert ds2.schema is ds.schema or ds2.schema.names() == ds.schema.names()
        assert len(ds2.couplings.couplings) == len(ds.couplings.couplings)


# ---------------------------------------------------------------------------
# DatasetContext + FieldHandle dispatch
# ---------------------------------------------------------------------------

class TestFieldHandleDispatch:
    def _dataset(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        return add_column(ds, DimensionedSliceField(name="clip"), [dim.spec(10, 30)])

    def test_handle_dispatch_builds_local_couplings_and_advances_context(self):
        ctx = self._dataset().context()
        clip = ctx.field("clip")
        data = ctx.field("data")

        pf.bind_slice(clip, data)
        pf.bind_materialize(data)
        result = consume(data)

        assert ctx.dataset is result
        assert result.table["data"].iloc[0].shape == (20,)
        bind, materialize = result.couplings.couplings
        assert type(bind.slice_field) is FieldRef
        assert type(bind.data_field) is FieldRef
        assert type(materialize.field) is FieldRef

    def test_ambient_string_dispatch_builds_and_consumes_couplings(self):
        with self._dataset().context() as ctx:
            pf.bind_slice("clip", "data")
            pf.bind_materialize("data")
            result = consume("data")

        assert ctx.dataset is result
        assert result.table["data"].iloc[0].shape == (20,)

    def test_bind_dimensions_accepts_nested_field_handles(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField(name="start", dimension=dim), [10])
        ds = add_column(ds, DimensionField(name="stop", dimension=dim), [30])
        ctx = ds.context()

        pf.bind_dimensions.instance(dataset_context=ctx)(
            slice_field="clip",
            bindings={"x": (ctx.field("start"), ctx.field("stop"))},
        )
        result = consume(ctx.field("clip"))

        assert result.table["clip"].iloc[0].dims["x"] == slice(10, 30)

    def test_handles_from_different_contexts_are_rejected(self):
        left = self._dataset().context()
        right = self._dataset().context()

        with pytest.raises(ValueError, match="must share one DatasetContext"):
            pf.bind_slice(left.field("clip"), right.field("data"))

    def test_handle_with_explicit_stale_dataset_snapshot_is_rejected(self):
        initial = self._dataset()
        ctx = initial.context()
        clip = ctx.field("clip")
        data = ctx.field("data")
        pf.bind_slice(clip, data)

        with pytest.raises(ValueError, match="current dataset snapshot"):
            pf.bind_materialize(initial, data)
