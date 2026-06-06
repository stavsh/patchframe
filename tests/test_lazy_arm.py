"""Tests for the in-level lazy arm — the convergence point.

`defer_in_level` records a deferred operator as an `ApplyOperator` on a bundle
carrier (the 99% path: the user bundled first), producing a caller-named output
column (`FieldOutput` acting) and returning a chaining handle. `collect()` runs
it. This exercises bundle + ApplyOperator + the terminal + chaining together.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.couplings import ApplyOperator
from patchframe.ops.bundle import defer_in_level


def test_coupling_able_drives_same_level_vs_bundle_routing():
    # The derived gate the interpreter reads to choose the lazy arm: coupling-able
    # (schema preserve/extend + one-to-one + per-row-independent) -> same-level;
    # otherwise -> BundleField carrier. No separate declaration.
    assert pf.bind_materialize.instance().coupling_able() is True  # preserve
    assert pf.bind_dimensions.instance().coupling_able() is True   # extend
    assert pf.bind_slice.instance().coupling_able() is True        # preserve
    assert pf.where.instance().coupling_able() is False            # FILTER cardinality
    assert pf.merge.instance().coupling_able() is False            # compose + DEPENDENT


def _merge_inputs() -> tuple[pf.Dataset, pf.Dataset, pf.Dataset]:
    left = pf.make_from_dataframe(
        pd.DataFrame({"score": [10, 20]}, index=["a", "b"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="score", dtype=int))),
    )
    right = pf.make_from_dataframe(
        pd.DataFrame({"label": [200, 300]}, index=["b", "c"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="label", dtype=int))),
    )
    plan = pf.join(left, right, how="outer")
    return left, right, plan


def test_defer_in_level_records_apply_operator_and_returns_handle():
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)

    merged = defer_in_level(
        pf.merge, b.field("left"), b.field("right"), b.field("plan"), out="merged"
    )

    assert isinstance(merged, pf.FieldHandle)
    assert merged.name == "merged"
    carrier = merged.dataset_context.dataset
    assert isinstance(carrier.schema.get("merged"), pf.BundleField)
    assert any(
        isinstance(c, ApplyOperator) and c.output_field() == "merged"
        for c in carrier.couplings.couplings
    )
    # The produced cell is declared but unmaterialized until collect.
    assert pd.isna(carrier.table.at[0, "merged"])


def test_defer_in_level_collect_equals_eager_merge():
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)

    merged = defer_in_level(
        pf.merge, b.field("left"), b.field("right"), b.field("plan"), out="merged"
    )
    result = merged.collect()
    eager = pf.merge(left, right, plan)

    assert result.schema.names() == eager.schema.names()
    pd.testing.assert_frame_equal(result.table, eager.table)


def test_defer_in_level_propagates_the_context_for_chaining():
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)
    context = b.field("left").dataset_context

    merged = defer_in_level(
        pf.merge, b.field("left"), b.field("right"), b.field("plan"), out="merged"
    )

    # The carrier advanced in place; the returned handle and re-fetched handles
    # share the one context (propagation), so the chain can continue.
    assert merged.dataset_context is context
    assert context.dataset.schema.has("merged")
    assert context.field("merged").name == "merged"


def test_defer_in_level_validates_bundle_handles_and_out():
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)

    with pytest.raises(ValueError, match="out"):
        defer_in_level(pf.merge, b.field("left"), out="")
    with pytest.raises(TypeError, match="must be bundle FieldHandles"):
        defer_in_level(pf.merge, "left", out="merged")


def test_merge_lazy_arm_end_to_end():
    # The user-facing duality: handle operands -> lazy; collect materializes.
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)

    merged = pf.merge(b.field("left"), b.field("right"), b.field("plan"), out="merged")

    assert isinstance(merged, pf.FieldHandle)
    result = merged.collect()
    eager = pf.merge(left, right, plan)
    pd.testing.assert_frame_equal(result.table, eager.table)


def test_merge_eager_arm_is_unchanged():
    # Dataset operands -> eager Dataset, exactly as before.
    left, right, plan = _merge_inputs()

    result = pf.merge(left, right, plan)

    assert isinstance(result, pf.Dataset)
    assert result.schema.names() == (
        "join_id",
        "left_index",
        "right_index",
        "score",
        "label",
    )


def _filter_dataset() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="v", dtype=int))),
    )


def test_where_lazy_arm_filters_per_fiber():
    # where is a per-row-independent lifting op; its lazy arm reuses
    # defer_in_level (one fiber cell + a predicate param), confirming the
    # mechanism is generic across op shapes.
    ds = _filter_dataset()
    b = pf.bundle(ds)  # cell_0 holds ds

    filtered = pf.where(b.field("cell_0"), lambda df: df["v"] > 1, out="filtered")

    assert isinstance(filtered, pf.FieldHandle)
    result = filtered.collect()
    eager = pf.where(ds, lambda df: df["v"] > 1)
    pd.testing.assert_frame_equal(result.table, eager.table)


def test_where_eager_arm_is_unchanged():
    ds = _filter_dataset()

    result = pf.where(ds, lambda df: df["v"] > 1)

    assert isinstance(result, pf.Dataset)
    assert result.table["v"].tolist() == [2, 3]


def test_bind_materialize_same_level_lazy_arm_returns_a_handle_no_bundle():
    # The non-lifting routing: a coupling-able op records its coupling directly
    # on the dataset (no bundle, no `out`) and returns a handle to the in-place
    # output field — contrast with merge/where, which lift onto a carrier.
    ds = _filter_dataset()
    ctx = ds.context()

    handle = pf.bind_materialize(ctx.field("v"))

    assert isinstance(handle, pf.FieldHandle)
    assert handle.name == "v"
    # Same-level: a Materialize coupling on the dataset itself, and no BundleField.
    assert any(isinstance(c, pf.Materialize) for c in ctx.dataset.couplings.couplings)
    assert all(not isinstance(field, pf.BundleField) for field in ctx.dataset.schema)


def test_bind_materialize_eager_arm_is_unchanged():
    ds = _filter_dataset()

    result = pf.bind_materialize(ds, "v")

    assert isinstance(result, pf.Dataset)
    assert any(isinstance(c, pf.Materialize) for c in result.couplings.couplings)


def test_bind_dimensions_same_level_lazy_arm_with_nested_handles():
    # The nested + fresh-output shape: handles inside `bindings` trigger the
    # same-level arm; the returned handle is the produced slice_field (which is
    # the coupling's output_field — fresh, not in-place).
    from patchframe.dataset.couplings import BindDimensions
    from patchframe.data.dimensions import IndexDimension

    dim = IndexDimension(name="x")
    ds = pf.make_from_dataframe(
        pd.DataFrame({"start": [10], "stop": [30]}, index=["a"]),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.DimensionField(name="start", dimension=dim),
                pf.DimensionField(name="stop", dimension=dim),
            )
        ),
    )
    ctx = ds.context()

    handle = pf.bind_dimensions(
        slice_field="clip",
        bindings={"x": (ctx.field("start"), ctx.field("stop"))},
    )

    assert isinstance(handle, pf.FieldHandle)
    assert handle.name == "clip"
    assert ctx.dataset.schema.has("clip")
    assert any(isinstance(c, BindDimensions) for c in ctx.dataset.couplings.couplings)
    assert all(not isinstance(field, pf.BundleField) for field in ctx.dataset.schema)
