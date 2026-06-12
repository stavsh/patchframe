"""Assignment conventions: functional on Dataset, mutating on the session types.

Pandas has two assignment conventions, and they map onto patchframe's
immutability split: ``df.assign(c=v)`` (functional, returns new) is
``Dataset.assign``; ``df["c"] = v`` / ``df.loc[i, "c"] = v`` (in-place) live on
the mutable authoring types — ``ctx[name] = values`` and
``handle.loc[ids] = values`` — which desugar to the ``assign`` operator and
advance the shared cursor. ``Dataset`` itself stays immutable.

assign assigns values to fields — FieldHandle sugar does not change that, and
a coupling's output field is not special-cased: values land, and a later
consume or coupling-aware access recomputes over them (user-side concern).
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _double(a):
    return a * 2


def _base() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.Index(["x", "y", "z"], name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="a", dtype=int),
            )
        ),
    )


# -- functional tier: Dataset.assign ------------------------------------------


def test_dataset_assign_returns_new_dataset():
    ds = _base()

    out = ds.assign(b=[10, 20, 30])

    assert isinstance(out, pf.Dataset)
    assert out.table["b"].tolist() == [10, 20, 30]
    assert isinstance(out.schema.get("b"), pf.ValueField)  # bare values infer
    assert "b" not in ds.schema.names()  # functional: the original is untouched


def test_dataset_assign_typed_field():
    ds = _base()

    out = ds.assign(score=(pf.ValueField(name="score", dtype=float), [0.1, 0.2, 0.3]))

    assert out.schema.get("score").dtype == pd.Float64Dtype()


def test_dataset_item_assignment_is_unsupported():
    # Dataset is the immutable facade; the mutating convention lives on the
    # session types. (The type you are holding tells you the semantics.)
    ds = _base()

    with pytest.raises(TypeError):
        ds["b"] = [10, 20, 30]


# -- mutating tier: DatasetContext setitem -------------------------------------


def test_context_setitem_assigns_and_advances_cursor():
    ds = _base()
    ctx = ds.context()
    handle = ctx.field("a")  # taken before the assignment

    ctx["b"] = [10, 20, 30]

    assert ctx.dataset.table["b"].tolist() == [10, 20, 30]
    assert "b" not in ds.schema.names()  # the snapshot is untouched
    # The cursor threads forward: handles minted earlier see the new column.
    assert handle.dataset_context.dataset is ctx.dataset


def test_context_setitem_typed_field_and_existing_column():
    ds = _base()
    ctx = ds.context()

    ctx["score"] = (pf.ValueField(name="score", dtype=float), [0.1, 0.2, 0.3])
    ctx["a"] = [7, 8, 9]  # existing column: values assigned, field def kept

    assert ctx.dataset.schema.get("score").dtype == pd.Float64Dtype()
    assert ctx.dataset.table["a"].tolist() == [7, 8, 9]
    assert ctx.dataset.schema.get("a").dtype == pd.Int64Dtype()


def test_context_setitem_field_def_key():
    # The Field key states the name once: ctx[field_def] = values.
    ds = _base()
    ctx = ds.context()

    ctx[pf.ValueField(name="score", dtype=float)] = [0.1, 0.2, 0.3]

    assert ctx.dataset.schema.get("score").dtype == pd.Float64Dtype()
    assert ctx.dataset.table["score"].tolist() == [0.1, 0.2, 0.3]


def test_context_setitem_field_def_key_must_match_existing():
    ds = _base()
    ctx = ds.context()

    with pytest.raises(ValueError, match="different definition"):
        ctx[pf.ValueField(name="a", dtype=float)] = [1.0, 2.0, 3.0]


def test_context_setitem_rejects_other_key_types():
    ds = _base()
    ctx = ds.context()

    with pytest.raises(TypeError, match="field-name or Field key"):
        ctx[123] = [1, 2, 3]


# -- mutating tier: FieldHandle.loc setter --------------------------------------


def test_handle_loc_setter_scalar_label():
    ds = _base()
    ctx = ds.context()
    handle = ctx.field("a")

    handle.loc["y"] = 99

    assert ctx.dataset.table["a"].tolist() == [1, 99, 3]
    assert ds.table["a"].tolist() == [1, 2, 3]  # original snapshot untouched


def test_handle_loc_setter_label_list_and_mask():
    ds = _base()
    ctx = ds.context()
    handle = ctx.field("a")

    handle.loc[["x", "z"]] = [7, 8]
    assert ctx.dataset.table["a"].tolist() == [7, 2, 8]

    mask = ctx.dataset.table["a"] > 2
    handle.loc[mask] = 0
    assert ctx.dataset.table["a"].tolist() == [0, 2, 0]


def test_handle_loc_setter_object_cell_values():
    ds = _base()
    ctx = ds.context()
    ctx["tags"] = [None, None, None]
    handle = ctx.field("tags")

    handle.loc["x"] = ("car", "tree")

    assert ctx.dataset.table["tags"].loc["x"] == ("car", "tree")


# -- coupling output fields are not special-cased -------------------------------


def test_assign_into_coupling_output_lands_values():
    ds = _base()
    handle = pf.map_fields(ds.fields(["a"]), _double, out="c")  # pending column

    handle.loc["y"] = 123

    column = handle.dataset_context.dataset.table["c"]
    assert column.loc["y"] == 123
    assert column.drop("y").isna().all()


def test_consume_recomputes_over_assigned_values():
    # assign assigns; the coupling remains the field's recipe. Running it
    # recomputes the column over hand-set values — guarding against that is
    # the user's concern, not engine machinery (ruling 2026-06-11).
    ds = _base()
    handle = pf.map_fields(ds.fields(["a"]), _double, out="c")

    handle.loc["y"] = 123

    assert handle.collect().table["c"].tolist() == [2, 4, 6]
