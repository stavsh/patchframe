"""Tests for generic source-indexed plan construction."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _target() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"value": pd.Series([1, 2], index=["a", "b"], dtype="Int64")},
            index=pd.Index(["a", "b"], name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )


def test_make_plan_builds_foreign_index_into_target_dataset():
    target = _target()

    plan = pf.make_plan(target, ["a", "b", "a"])

    assert plan.schema.names() == ("plan_id", "source_index")
    assert plan.table.index.name == "plan_id"
    assert plan.table["source_index"].tolist() == ["a", "b", "a"]
    source_index = plan.schema.get("source_index")
    assert isinstance(source_index, pf.ForeignIndexField)
    assert source_index.target_identity == pf.primary_index_identity(target)


def test_make_plan_from_dataframe_infers_extra_value_fields():
    target = _target()
    table = pd.DataFrame(
        {
            "source_index": ["a", "b"],
            "score": [0.1, 0.2],
        },
        index=pd.Index([10, 20]),
    )

    plan = pf.make_plan.from_dataframe(target, table)

    assert plan.table.index.tolist() == [10, 20]
    assert plan.table.index.name == "plan_id"
    assert isinstance(plan.schema.get("source_index"), pf.ForeignIndexField)
    assert isinstance(plan.schema.get("score"), pf.ValueField)


def test_make_plan_from_series_uses_series_name_for_foreign_index_field():
    target = _target()

    plan = pf.make_plan.from_series(
        target,
        pd.Series(["a", "b"], name="parent_index"),
    )

    assert plan.schema.names() == ("plan_id", "parent_index")
    assert isinstance(plan.schema.get("parent_index"), pf.ForeignIndexField)


def test_make_plan_preserves_explicit_empty_metadata():
    target = _target()

    plan = pf.make_plan(target, ["a"], metadata={})

    assert plan.state.metadata == {}


def test_make_plan_rejects_null_and_missing_source_labels():
    target = _target()

    with pytest.raises(ValueError, match="contains null labels"):
        pf.make_plan(target, ["a", pd.NA])

    with pytest.raises(ValueError, match="missing from target dataset"):
        pf.make_plan(target, ["missing"])


def test_assign_adds_inferred_and_explicit_fields_in_one_operation():
    target = _target()
    plan = pf.make_plan(target, ["a", "b"])
    slices = pf.DimensionedSliceArray._from_sequence(
        [
            pf.DimensionedSlice(dims={"x": slice(0, 1)}),
            pf.DimensionedSlice(dims={"x": slice(1, 2)}),
        ]
    )

    result = pf.assign(
        plan,
        score=[0.1, 0.2],
        bbox=(pf.DimensionedSliceField(name="bbox", nullable=False), slices),
    )

    assert isinstance(result.schema.get("score"), pf.ValueField)
    assert isinstance(result.schema.get("bbox"), pf.DimensionedSliceField)
    assert isinstance(result.table["bbox"].array, pf.DimensionedSliceArray)


def test_assign_updates_existing_values_without_retyping_field():
    target = _target()

    result = pf.assign(target, value=[3, 4])

    assert result.table["value"].tolist() == [3, 4]
    assert result.schema.get("value") == target.schema.get("value")


def test_assign_rejects_explicit_retype_of_existing_field():
    target = _target()

    with pytest.raises(ValueError, match="field-casting operator"):
        pf.assign(
            target,
            value=(pf.ValueField(name="value", dtype=str), ["x", "y"]),
        )
