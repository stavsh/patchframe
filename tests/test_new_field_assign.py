"""`new_field` (schema-mutating cursor primitive) + `assign`'s handle form.

`ds.new_field(field_def)` adds a null-filled field and returns a context-bound
handle; successive calls accrete on one cursor so the handles co-resolve. The
handle form `assign([h_a, h_b], values)` fills those fields from a frame/mapping
keyed by name and returns a `FieldSelection`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.context import FieldSelection


def _ds() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="v", dtype=int))),
    )


def test_new_field_adds_null_filled_field_and_returns_handle():
    ds = _ds()

    handle = ds.new_field(pf.ValueField(name="a1", dtype=int))

    assert isinstance(handle, pf.FieldHandle)
    assert handle.name == "a1"
    snapshot = handle.dataset_context.dataset
    assert snapshot.schema.has("a1")
    assert snapshot.table["a1"].isna().all()


def test_multiple_new_fields_share_one_context_and_coresolve():
    # The subtlety: new_field is a cursor op, so two new fields accrete on one
    # context and both resolve in the shared snapshot (not two forked snapshots).
    ds = _ds()

    h1 = ds.new_field(pf.ValueField(name="a1", dtype=int))
    h2 = ds.new_field(pf.ValueField(name="a2", dtype=int))

    assert h1.dataset_context is h2.dataset_context
    snapshot = h1.dataset_context.dataset
    assert snapshot.schema.has("a1") and snapshot.schema.has("a2")
    assert (h1.name, h2.name) == ("a1", "a2")


def test_explicit_context_new_field_shares_the_cursor():
    ctx = _ds().context()

    h1 = ctx.new_field(pf.ValueField(name="a1", dtype=int))
    h2 = ctx.new_field(pf.ValueField(name="a2", dtype=int))

    assert h1.dataset_context is ctx and h2.dataset_context is ctx
    assert ctx.dataset.schema.has("a1") and ctx.dataset.schema.has("a2")


def test_assign_handle_form_fills_new_fields_equals_eager():
    ds = _ds()
    h1 = ds.new_field(pf.ValueField(name="a1", dtype=int))
    h2 = ds.new_field(pf.ValueField(name="a2", dtype=int))

    selection = pf.assign([h1, h2], {"a1": [10, 20, 30], "a2": [40, 50, 60]})

    assert isinstance(selection, FieldSelection)
    assert selection.names() == ("a1", "a2")
    result = selection.collect()
    eager = pf.assign(
        _ds(),
        a1=(pf.ValueField(name="a1", dtype=int), [10, 20, 30]),
        a2=(pf.ValueField(name="a2", dtype=int), [40, 50, 60]),
    )
    assert result.schema.names() == eager.schema.names()
    pd.testing.assert_frame_equal(result.table, eager.table)


def test_assign_handle_form_accepts_a_dataframe_keyed_by_name():
    ds = _ds()
    h1 = ds.new_field(pf.ValueField(name="a1", dtype=int))

    frame = pd.DataFrame({"a1": [7, 8, 9]}, index=["a", "b", "c"])
    result = pf.assign([h1], frame).collect()

    assert result.table["a1"].tolist() == [7, 8, 9]


def test_assign_handle_form_requires_values():
    ds = _ds()
    h1 = ds.new_field(pf.ValueField(name="a1", dtype=int))

    with pytest.raises(TypeError, match="needs a values frame"):
        pf.assign([h1])


def test_assign_eager_kwarg_form_unchanged():
    result = pf.assign(_ds(), label=["x", "y", "z"])

    assert result.schema.has("label")
    assert result.table["label"].tolist() == ["x", "y", "z"]


def test_assign_values_remains_a_usable_column_name():
    # target/values are positional-only, so "values" still works as a column kwarg.
    result = pf.assign(_ds(), values=[10, 20, 30])

    assert result.table["values"].tolist() == [10, 20, 30]


def test_add_column_handle_form_fills_field_and_returns_handle():
    from patchframe.ops.builtin.add_column import add_column

    ds = _ds()
    h = ds.new_field(pf.ValueField(name="score", dtype=int))

    out = add_column(h, [10, 20, 30])

    assert isinstance(out, pf.FieldHandle)
    assert out.name == "score"
    result = out.collect()
    eager = add_column(_ds(), pf.ValueField(name="score", dtype=int), [10, 20, 30])
    pd.testing.assert_frame_equal(result.table, eager.table)


def test_add_column_handle_form_one_liner_and_requires_values():
    from patchframe.ops.builtin.add_column import add_column

    handle = add_column(_ds().new_field(pf.ValueField(name="s2", dtype=int)), [7, 8, 9])
    assert handle.collect().table["s2"].tolist() == [7, 8, 9]

    with pytest.raises(TypeError, match="needs values"):
        add_column(_ds().new_field(pf.ValueField(name="s3", dtype=int)))
