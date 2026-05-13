"""Tests for drop, keep, bind_dimensions operators and make_mock_dataset_from_dims."""

from __future__ import annotations

import pytest

from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions, IndexDimension
from patchframe.data.manager import reset_default_manager
from patchframe.dataset.couplings import BindDimensions, BindSlice, CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    DimensionedSliceField,
    DimensionField,
    IndexField,
    ValueField,
)
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.builtin.add_column import add_column
from patchframe.ops.builtin.bind_dimensions import bind_dimensions
from patchframe.ops.builtin.consume import consume
from patchframe.ops.builtin.drop import drop
from patchframe.ops.builtin.keep import keep
from patchframe.testing.mock import make_mock_dataset, make_mock_dataset_from_dims


@pytest.fixture(autouse=True)
def fresh_manager():
    reset_default_manager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_ds():
    """Dataset with item_id, data, score, clip fields and one BindSlice coupling."""
    dim = IndexDimension(name="x")
    ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
    ds = add_column(ds, ValueField(name="score", dtype=float), [1.0, 2.0])
    slices = [dim.spec(0, 20), dim.spec(10, 50)]
    coupling = BindSlice(slice_field="clip", data_field="data")
    return add_column(ds, DimensionedSliceField(name="clip"), slices, couplings=(coupling,))


def _dim_ds():
    """Dataset with item_id + start/end DimensionField columns only (no data)."""
    dim = IndexDimension(name="x")
    import pandas as pd

    df = pd.DataFrame(
        {"start": [0, 10], "end": [20, 50]},
        index=pd.Index(["a", "b"], name="item_id"),
    )
    schema = Schema(
        fields=(
            IndexField(name="item_id"),
            DimensionField.from_dim(dim, "start", dtype=int),
            DimensionField.from_dim(dim, "end", dtype=int),
        )
    )
    return Dataset(state=DatasetState(schema=schema, table=df, couplings=CouplingSet()))


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------


class TestDrop:
    def test_removes_field_from_schema(self):
        ds = _base_ds()
        ds2 = drop(ds, ["score"])
        assert not ds2.schema.has("score")

    def test_removes_column_from_table(self):
        ds = _base_ds()
        ds2 = drop(ds, ["score"])
        assert "score" not in ds2.table.columns

    def test_drop_multiple_fields(self):
        ds = _base_ds()
        ds2 = drop(ds, ["score", "clip"])
        assert not ds2.schema.has("score")
        assert not ds2.schema.has("clip")
        assert "score" not in ds2.table.columns
        assert "clip" not in ds2.table.columns

    def test_prunes_coupling_whose_input_is_dropped(self):
        ds = _base_ds()
        assert len(ds.couplings.couplings) == 1
        ds2 = drop(ds, ["clip"])
        assert len(ds2.couplings.couplings) == 0

    def test_preserves_coupling_unrelated_to_dropped_field(self):
        ds = _base_ds()
        ds2 = drop(ds, ["score"])
        assert len(ds2.couplings.couplings) == 1

    def test_raises_on_unknown_field(self):
        ds = _base_ds()
        with pytest.raises(ValueError, match="not in schema"):
            drop(ds, ["nonexistent"])

    def test_original_unchanged(self):
        ds = _base_ds()
        drop(ds, ["score"])
        assert ds.schema.has("score")
        assert "score" in ds.table.columns

    def test_drop_index_field_removes_from_schema_only(self):
        ds = _base_ds()
        ds2 = drop(ds, ["item_id"])
        assert not ds2.schema.has("item_id")
        assert ds2.table.index.name == "item_id"


# ---------------------------------------------------------------------------
# keep
# ---------------------------------------------------------------------------


class TestKeep:
    def test_retains_only_listed_fields_in_schema(self):
        ds = _base_ds()
        ds2 = keep(ds, ["item_id", "data"])
        assert set(ds2.schema.names()) == {"item_id", "data"}

    def test_drops_unlisted_columns_from_table(self):
        ds = _base_ds()
        ds2 = keep(ds, ["item_id", "data"])
        assert "score" not in ds2.table.columns
        assert "clip" not in ds2.table.columns
        assert "data" in ds2.table.columns

    def test_preserves_schema_field_order(self):
        ds = _base_ds()
        ds2 = keep(ds, ["data", "item_id"])
        assert ds2.schema.names() == ("item_id", "data")

    def test_prunes_coupling_whose_field_is_excluded(self):
        ds = _base_ds()
        assert len(ds.couplings.couplings) == 1
        ds2 = keep(ds, ["item_id", "data", "score"])
        assert len(ds2.couplings.couplings) == 0

    def test_preserves_coupling_when_all_fields_kept(self):
        ds = _base_ds()
        ds2 = keep(ds, ["item_id", "data", "clip"])
        assert len(ds2.couplings.couplings) == 1

    def test_raises_on_unknown_field(self):
        ds = _base_ds()
        with pytest.raises(ValueError, match="not in schema"):
            keep(ds, ["item_id", "nonexistent"])

    def test_original_unchanged(self):
        ds = _base_ds()
        keep(ds, ["item_id", "data"])
        assert ds.schema.has("score")
        assert "score" in ds.table.columns


# ---------------------------------------------------------------------------
# bind_dimensions
# ---------------------------------------------------------------------------


class TestBindDimensions:
    def _ds_with_dim_fields(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        ds = add_column(ds, DimensionField.from_dim(dim, "start", dtype=int), [0, 10])
        ds = add_column(ds, DimensionField.from_dim(dim, "end", dtype=int), [20, 50])
        return ds

    def test_adds_dimensioned_slice_field_to_schema(self):
        ds = self._ds_with_dim_fields()
        ds2 = bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
        assert ds2.schema.has("clip")
        assert isinstance(ds2.schema.get("clip"), DimensionedSliceField)

    def test_adds_null_column_to_table(self):
        ds = self._ds_with_dim_fields()
        ds2 = bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
        assert "clip" in ds2.table.columns
        assert ds2.table["clip"].isna().all()

    def test_adds_bind_dimensions_coupling(self):
        ds = self._ds_with_dim_fields()
        ds2 = bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
        assert len(ds2.couplings.couplings) == 1
        assert isinstance(ds2.couplings.couplings[0], BindDimensions)

    def test_consume_produces_correct_slices(self):
        ds = self._ds_with_dim_fields()
        ds2 = bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
        ds3 = consume(ds2, "clip")
        assert ds3.table["clip"].iloc[0].dims["x"] == slice(0, 20)
        assert ds3.table["clip"].iloc[1].dims["x"] == slice(10, 50)

    def test_chain_second_dimension(self):
        x = IndexDimension(name="x")
        y = IndexDimension(name="y")
        extent = DimensionedSlice(dims={"x": slice(0, 100), "y": slice(0, 50)})
        ds = make_mock_dataset(["a"], Dimensions((x, y)), extent, seed=0)
        ds = add_column(ds, DimensionField.from_dim(x, "x0", dtype=int), [5])
        ds = add_column(ds, DimensionField.from_dim(x, "x1", dtype=int), [15])
        ds = add_column(ds, DimensionField.from_dim(y, "y0", dtype=int), [2])
        ds = add_column(ds, DimensionField.from_dim(y, "y1", dtype=int), [8])
        ds = bind_dimensions(ds, slice_field="clip", bindings={"x": ("x0", "x1")})
        ds = bind_dimensions(ds, slice_field="clip", bindings={"y": ("y0", "y1")})
        ds2 = consume(ds, "clip")
        assert ds2.table["clip"].iloc[0].dims == {"x": slice(5, 15), "y": slice(2, 8)}

    def test_idempotent_on_duplicate_call(self):
        ds = self._ds_with_dim_fields()
        ds2 = bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
        ds3 = bind_dimensions(ds2, slice_field="clip", bindings={"x": ("start", "end")})
        assert len(ds3.couplings.couplings) == 1

    def test_raises_if_slice_field_exists_as_wrong_type(self):
        ds = self._ds_with_dim_fields()
        ds = add_column(ds, ValueField(name="clip", dtype=float), [1.0, 2.0])
        with pytest.raises(TypeError, match="not DimensionedSliceField"):
            bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})

    def test_raises_if_binding_field_missing(self):
        ds = self._ds_with_dim_fields()
        with pytest.raises(ValueError, match="not in schema"):
            bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "nonexistent")})

    def test_raises_if_binding_field_not_dimension_field(self):
        ds = self._ds_with_dim_fields()
        ds = add_column(ds, ValueField(name="label", dtype=str), ["a", "b"])
        with pytest.raises(TypeError, match="not a DimensionField"):
            bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "label")})


# ---------------------------------------------------------------------------
# DimensionField.from_dim
# ---------------------------------------------------------------------------


class TestDimensionFieldFromDim:
    def test_matches_explicit_construction(self):
        dim = IndexDimension(name="x")
        assert DimensionField.from_dim(dim, "start", dtype=int) == DimensionField(
            name="start", dimension=dim, dtype=int
        )

    def test_no_dtype_defaults(self):
        dim = IndexDimension(name="x")
        f = DimensionField.from_dim(dim, "onset")
        assert f.name == "onset"
        assert f.dimension is dim
        assert f.dtype is None


# ---------------------------------------------------------------------------
# make_mock_dataset_from_dims
# ---------------------------------------------------------------------------


class TestMakeMockDatasetFromDims:
    def test_creates_data_column(self):
        from patchframe.dataset.fields import DataField

        ds = _dim_ds()
        result = make_mock_dataset_from_dims(ds, data_field="audio")
        assert result.schema.has("audio")
        assert isinstance(result.schema.get("audio"), DataField)

    def test_arrays_sized_by_dimension_fields(self):
        ds = _dim_ds()
        result = make_mock_dataset_from_dims(ds)
        assert result.table["data"].iloc[0].materialize().shape == (20,)
        assert result.table["data"].iloc[1].materialize().shape == (40,)

    def test_custom_data_field_name(self):
        ds = _dim_ds()
        result = make_mock_dataset_from_dims(ds, data_field="waveform")
        assert "waveform" in result.table.columns
        assert "data" not in result.table.columns

    def test_raises_with_no_dimension_fields(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 10))
        with pytest.raises(ValueError, match="no DimensionField columns"):
            make_mock_dataset_from_dims(ds)

    def test_seed_is_reproducible(self):
        import numpy as np

        ds = _dim_ds()
        r1 = make_mock_dataset_from_dims(ds, seed=42)
        reset_default_manager()
        r2 = make_mock_dataset_from_dims(ds, seed=42)
        arr1 = r1.table["data"].iloc[0].materialize()
        arr2 = r2.table["data"].iloc[0].materialize()
        assert np.array_equal(arr1, arr2)
