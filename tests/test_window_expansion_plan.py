"""Tests for window expansion plan construction."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.testing.mock import make_mock_dataset


def _dimension_field_dataset() -> pf.Dataset:
    x = pf.IndexDimension(name="x")
    y = pf.IndexDimension(name="y")
    table = pd.DataFrame(
        {
            "x0": [0, 10],
            "x1": [40, 50],
            "y0": [0, 0],
            "y1": [30, 10],
        },
        index=pd.Index(["a", "b"], name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.DimensionField.from_dim(x, "x0", dtype=int),
            pf.DimensionField.from_dim(x, "x1", dtype=int),
            pf.DimensionField.from_dim(y, "y0", dtype=int),
            pf.DimensionField.from_dim(y, "y1", dtype=int),
        )
    )
    return pf.Dataset(state=pf.DatasetState(schema=schema, table=table))


def test_window_expansion_plan_from_dimension_field_bindings():
    ds = _dimension_field_dataset()

    plan = pf.window_expansion_plan(
        ds,
        bindings={"x": ("x0", "x1"), "y": ("y0", "y1")},
        windows={"x": pf.AxisWindow(20, 20), "y": pf.AxisWindow(10, 10)},
    )

    assert plan.table.index.name == "plan_id"
    assert plan.schema.has("source_index")
    assert isinstance(plan.schema.get("slice"), pf.DimensionedSliceField)
    assert plan.table["source_index"].tolist() == ["a"] * 6 + ["b"] * 2
    assert isinstance(plan.table["slice"].array, pf.DimensionedSliceArray)
    assert plan.table["slice"].iloc[0].dims == {
        "x": slice(0, 20),
        "y": slice(0, 10),
    }
    assert plan.table["slice"].iloc[-1].dims == {
        "x": slice(30, 50),
        "y": slice(0, 10),
    }


def test_window_expansion_plan_from_slice_field_skips_null_rows():
    x = pf.IndexDimension(name="x")
    table = pd.DataFrame(
        {
            "extent": [
                x.spec(0, 5),
                pd.NA,
                x.spec(10, 14),
            ]
        },
        index=pd.Index(["a", "b", "c"], name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.DimensionedSliceField(name="extent"),
        )
    )
    ds = pf.Dataset(state=pf.DatasetState(schema=schema, table=table))

    plan = pf.window_expansion_plan(
        ds,
        field="extent",
        windows={"x": pf.AxisWindow(2, 2, include_partial=True)},
    )

    assert plan.table["source_index"].tolist() == ["a", "a", "a", "c", "c"]
    assert [s.dims["x"] for s in plan.table["slice"]] == [
        slice(0, 2),
        slice(2, 4),
        slice(4, 5),
        slice(10, 12),
        slice(12, 14),
    ]


def test_window_expansion_plan_handle_operand_defers_to_bundle_arm():
    """Operand-dispatch law: a FieldHandle input selects the lazy arm.

    The deferred form lifts onto a BundleField carrier (like explode): the
    source rides in a cell, field/bindings ride as replay params, and collect
    runs the eager plan construction per cell.
    """

    ds = _dimension_field_dataset()
    eager = pf.window_expansion_plan(
        ds,
        bindings={"x": ("x0", "x1")},
        windows={"x": pf.AxisWindow(20, 20)},
    )

    b = pf.bundle(src=ds)
    handle = pf.window_expansion_plan(
        b.field("src"),
        bindings={"x": ("x0", "x1")},
        windows={"x": pf.AxisWindow(20, 20)},
        out="plan",
    )

    assert isinstance(handle, pf.FieldHandle)
    plan = handle.collect()
    assert plan.table["source_index"].tolist() == eager.table["source_index"].tolist()
    assert plan.table.index.name == eager.table.index.name


def test_window_expansion_plan_field_handles_never_resolve_eagerly():
    """The law: handles mean lazy — there is no eager handle resolution.

    A mixed call (eager Dataset operand + field handles) is rejected; eager
    calls pass names, deferral goes through a bundle cell.
    """

    x = pf.IndexDimension(name="x")
    table = pd.DataFrame(
        {"extent": [x.spec(0, 5)]},
        index=pd.Index(["a"], name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.DimensionedSliceField(name="extent"),
        )
    )
    ds = pf.Dataset(state=pf.DatasetState(schema=schema, table=table))
    ctx = ds.context()

    with pytest.raises(TypeError, match="bundle FieldHandles"):
        pf.window_expansion_plan(
            ds,
            field=ctx.field("extent"),
            windows={"x": pf.AxisWindow(2, 2, include_partial=True)},
            out="plan",
        )

    bounded = _dimension_field_dataset()
    bounded_ctx = bounded.context()
    with pytest.raises(TypeError, match="bundle FieldHandles"):
        pf.window_expansion_plan(
            bounded,
            bindings={"x": (bounded_ctx.field("x0"), bounded_ctx.field("x1"))},
            windows={"x": pf.AxisWindow(20, 20)},
            out="plan",
        )


def test_window_expansion_plan_from_columnar_slice_field():
    x = pf.IndexDimension(name="x")
    extents = pf.DimensionedSliceArray.from_columns(
        dimensions=(x,),
        selector_columns=(([0, 10], [5, 14]),),
    )
    table = pd.DataFrame(index=pd.Index(["a", "b"], name="item_id"))
    table["extent"] = pd.Series(extents, index=table.index)
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.DimensionedSliceField(name="extent"),
        )
    )
    ds = pf.Dataset(state=pf.DatasetState(schema=schema, table=table))

    plan = pf.window_expansion_plan(
        ds,
        field="extent",
        windows={"x": pf.AxisWindow(2, 2)},
    )

    assert plan.table["source_index"].tolist() == ["a", "a", "b", "b"]
    assert plan.table["slice"].iloc[0].dims["x"] == slice(0, 2)
    assert plan.table["slice"].iloc[-1].dims["x"] == slice(12, 14)


def test_window_expansion_plan_rejects_data_field_for_now():
    x = pf.IndexDimension(name="x")
    ds = make_mock_dataset(
        ["a"],
        pf.Dimensions((x,)),
        x.spec(0, 5),
    )

    with pytest.raises(TypeError, match="must be a DimensionedSliceField"):
        pf.window_expansion_plan(
            ds,
            field="data",
            windows={"x": pf.AxisWindow(2, 2, include_partial=True)},
        )


def test_window_expansion_plan_marks_plan_metadata_and_warns_on_plan_input():
    ds = _dimension_field_dataset()
    plan = pf.window_expansion_plan(
        ds,
        bindings={"x": ("x0", "x1")},
        windows={"x": pf.AxisWindow(20, 20)},
    )

    metadata = plan.state.metadata["patchframe.plan"]
    assert metadata["type"] == "window_expansion"
    assert metadata["operator"] == "window_expansion_plan"
    assert metadata["source_index_field"] == "source_index"
    assert metadata["slice_field"] == "slice"
    assert metadata["input_index_name"] == "item_id"

    with pytest.warns(UserWarning, match="already marked as a plan"):
        pf.window_expansion_plan(
            plan,
            field="slice",
            windows={"x": pf.AxisWindow(10, 10)},
        )


def test_window_expansion_plan_rejects_null_multi_field_bindings():
    ds = _dimension_field_dataset()
    table = ds.table.copy()
    table.loc["b", "x0"] = pd.NA
    ds = ds.replace_state(table=table)

    with pytest.raises(ValueError, match="does not allow null values"):
        pf.window_expansion_plan(
            ds,
            bindings={"x": ("x0", "x1")},
            windows={"x": pf.AxisWindow(20, 20)},
        )


def test_dimensioned_slice_array_explode_windows_preserves_base_dimensions():
    base = pf.DimensionedSliceArray._from_sequence([
        pf.DimensionedSlice(dims={"time": slice(1.0, 2.0), "x": slice(0, 5)})
    ])

    parent_positions, slices = base.explode_windows({"x": pf.AxisWindow(2, 2)})

    assert parent_positions.tolist() == [0, 0]
    assert slices[0].dims == {"time": slice(1.0, 2.0), "x": slice(0, 2)}
    assert slices[1].dims == {"time": slice(1.0, 2.0), "x": slice(2, 4)}


def test_dimensioned_slice_array_explode_windows_uses_latest_duplicate_dimension():
    x = pf.IndexDimension(name="x")
    base = pf.DimensionedSliceArray.from_columns(
        dimensions=(x,),
        selector_columns=(([0], [5]),),
    )
    _, first_tiles = base.explode_windows({"x": pf.AxisWindow(2, 2)})

    _, second_tiles = first_tiles.explode_windows({"x": pf.AxisWindow(1, 1)})

    assert [s.dims["x"] for s in second_tiles] == [
        slice(0, 1),
        slice(1, 2),
        slice(2, 3),
        slice(3, 4),
    ]
