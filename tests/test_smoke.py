"""Smoke tests for make_mock_dataset and make_from_dataframe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patchframe.data.dimensions import Dimensions, IndexDimension, TemporalDimension
from patchframe.data.manager import reset_default_manager
from patchframe.dataset.fields import DataField, IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.builtin.make_from_dataframe import make_from_dataframe
from patchframe.testing.mock import make_mock_dataset


@pytest.fixture(autouse=True)
def fresh_manager():
    reset_default_manager()


# ---------------------------------------------------------------------------
# make_mock_dataset
# ---------------------------------------------------------------------------

class TestMakeMockDataset:
    def _dim(self) -> IndexDimension:
        return IndexDimension(name="x")

    def test_table_shape(self):
        dim = self._dim()
        ds = make_mock_dataset(["a", "b", "c"], Dimensions((dim,)), dim.spec(0, 10))
        assert len(ds.table) == 3
        assert "data" in ds.table.columns

    def test_schema_fields(self):
        dim = self._dim()
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 10))
        assert ds.schema.has("item_id")
        assert ds.schema.has("data")
        assert isinstance(ds.schema.get("item_id"), IndexField)
        assert isinstance(ds.schema.get("data"), DataField)

    def test_materialize_full_shape_and_dtype(self):
        dim = self._dim()
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 64), seed=0)
        arr = ds.table["data"].iloc[0].materialize()
        assert arr.shape == (64,)
        assert arr.dtype == np.float32

    def test_materialize_sliced(self):
        dim = self._dim()
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 100), seed=0)
        arr = ds.table["data"].iloc[0].slice(dim.spec(20, 60)).materialize()
        assert arr.shape == (40,)

    def test_temporal_dimension_slice(self):
        dim = TemporalDimension(name="time", sample_rate=16000)
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0.0, 1.0), seed=0)
        # 0.5 s at 16 kHz = 8000 samples
        arr = ds.table["data"].iloc[0].slice(dim.spec(0.0, 0.5)).materialize()
        assert arr.shape == (8000,)

    def test_seed_reproducibility(self):
        dim = self._dim()
        extent = dim.spec(0, 32)
        ds1 = make_mock_dataset(["a"], Dimensions((dim,)), extent, seed=7)
        arr1 = ds1.table["data"].iloc[0].materialize()
        reset_default_manager()
        ds2 = make_mock_dataset(["a"], Dimensions((dim,)), extent, seed=7)
        arr2 = ds2.table["data"].iloc[0].materialize()
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_items_differ(self):
        dim = self._dim()
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 32), seed=1)
        arr_a = ds.table["data"].loc["a"].materialize()
        arr_b = ds.table["data"].loc["b"].materialize()
        assert not np.array_equal(arr_a, arr_b)

    def test_per_item_extents(self):
        dim = self._dim()
        extents = {"short": dim.spec(0, 10), "long": dim.spec(0, 50)}
        ds = make_mock_dataset(list(extents), Dimensions((dim,)), extents, seed=0)
        assert ds.table["data"].loc["short"].materialize().shape == (10,)
        assert ds.table["data"].loc["long"].materialize().shape == (50,)

    def test_multiple_data_fields(self):
        dim = self._dim()
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 16),
                               data_fields=("audio", "video"))
        assert ds.table["audio"].iloc[0].materialize().shape == (16,)
        assert ds.table["video"].iloc[0].materialize().shape == (16,)

    def test_multidim_shape(self):
        dims = Dimensions((IndexDimension(name="time"), IndexDimension(name="freq")))
        extent = dims.dims[0].spec(0, 10)
        from patchframe.data.dimensioned_slice import DimensionedSlice
        extent = DimensionedSlice(dims={"time": slice(0, 10), "freq": slice(0, 5)})
        ds = make_mock_dataset(["a"], dims, extent, seed=0)
        assert ds.table["data"].iloc[0].materialize().shape == (10, 5)

    def test_source_manager_on_dataset(self):
        dim = self._dim()
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 8))
        assert ds.source_manager is not None

    def test_extent_for(self):
        dim = self._dim()
        extent = dim.spec(0, 32)
        ds = make_mock_dataset(["a"], Dimensions((dim,)), extent)
        source = ds.source_manager.get_source_by_descriptor_id(
            ds.table["data"].iloc[0].source_desc_id
        )
        assert source.extent_for("a") == extent


# ---------------------------------------------------------------------------
# make_from_dataframe
# ---------------------------------------------------------------------------

class TestMakeFromDataframe:
    def test_dataset_state_rejects_duplicate_index(self):
        df = pd.DataFrame({"x": [1, 2]}, index=["a", "a"])
        schema = Schema(fields=(ValueField(name="x", dtype=int),))

        with pytest.raises(ValueError, match="index must be unique"):
            DatasetState(schema=schema, table=df)

    def test_make_from_dataframe_rejects_duplicate_index(self):
        df = pd.DataFrame({"x": [1, 2]}, index=["a", "a"])
        schema = Schema(fields=(ValueField(name="x", dtype=int),))

        with pytest.raises(ValueError, match="index must be unique"):
            make_from_dataframe(df, schema)

    def test_basic_value_fields(self):
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
        schema = Schema(fields=(
            ValueField(name="score", dtype=float),
            ValueField(name="label", dtype=str),
        ))
        ds = make_from_dataframe(df, schema)
        assert len(ds.table) == 3
        assert list(ds.table.columns) == ["score", "label"]

    def test_schema_preserved(self):
        df = pd.DataFrame({"x": [1, 2]})
        schema = Schema(fields=(ValueField(name="x", dtype=int),))
        ds = make_from_dataframe(df, schema)
        assert ds.schema.has("x")

    def test_copy_true_isolates_from_original(self):
        df = pd.DataFrame({"x": pd.array([1, 2], dtype=pd.Int64Dtype())})
        schema = Schema(fields=(ValueField(name="x", dtype=int),))
        ds = make_from_dataframe(df, schema)
        df["x"] = pd.array([99, 99], dtype=pd.Int64Dtype())
        assert ds.table["x"].iloc[0] != 99

    def test_copy_false_shares_data(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        schema = Schema(fields=(ValueField(name="x", dtype=float),))
        ds = make_from_dataframe.instance(copy=False)(df, schema)
        assert ds.table is df

    def test_wraps_mock_dataset_accessors(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a", "b"], Dimensions((dim,)), dim.spec(0, 20), seed=3)
        ds2 = make_from_dataframe(ds.table, ds.schema)
        assert ds2.table["data"].iloc[0].materialize().shape == (20,)

    def test_wrapped_data_matches_original(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 20), seed=3)
        original = ds.table["data"].iloc[0].materialize()
        ds2 = make_from_dataframe(ds.table, ds.schema)
        np.testing.assert_array_equal(ds2.table["data"].iloc[0].materialize(), original)

    def test_sources_registered_in_dataset_manager(self):
        dim = IndexDimension(name="x")
        ds = make_mock_dataset(["a"], Dimensions((dim,)), dim.spec(0, 8), seed=0)
        ds2 = make_from_dataframe(ds.table, ds.schema)
        accessor = ds2.table["data"].iloc[0]
        assert accessor.materialize(ds2.source_manager).shape == (8,)
