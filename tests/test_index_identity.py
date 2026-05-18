"""Tests for semantic index identity propagation."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _dataset(index: list[str], *, values: list[int] | None = None) -> pf.Dataset:
    table = pd.DataFrame(
        {"value": values or list(range(len(index)))},
        index=pd.Index(index, name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.ValueField(name="value", dtype=int),
        )
    )
    return pf.make_from_dataframe(table, schema)


def test_creation_mints_distinct_index_identities():
    left = _dataset(["a"])
    right = _dataset(["a"])

    assert pf.primary_index_identity(left) != pf.primary_index_identity(right)


def test_where_preserves_index_identity():
    ds = _dataset(["a", "b"], values=[1, 2])
    result = pf.where(ds, ds.table["value"] > 1)

    assert pf.primary_index_identity(result) == pf.primary_index_identity(ds)


def test_set_index_mints_new_identity_and_preserves_old_index_column_identity():
    ds = _dataset(["a", "b"], values=[1, 2])
    old_identity = pf.primary_index_identity(ds)

    result = pf.set_index(ds, "value")

    assert pf.primary_index_identity(result) != old_identity
    index_column = result.schema.get("item_id")
    assert isinstance(index_column, pf.IndexColumnField)
    assert index_column.index_identity == old_identity


def test_concat_rows_preserves_shared_index_identity():
    ds = _dataset(["a", "b"], values=[1, 2])
    left = pf.where(ds, ds.table["value"] == 1)
    right = pf.where(ds, ds.table["value"] == 2)

    result = pf.concat_rows(left, right)

    assert pf.primary_index_identity(result) == pf.primary_index_identity(ds)


def test_concat_rows_mints_identity_for_different_index_namespaces():
    left = _dataset(["a"])
    right = _dataset(["b"])

    result = pf.concat_rows(left, right)

    assert pf.primary_index_identity(result) != pf.primary_index_identity(left)
    assert pf.primary_index_identity(result) != pf.primary_index_identity(right)


def test_concat_columns_mints_identity_for_different_index_namespaces():
    left = _dataset(["a"], values=[1])
    right = pf.make_from_dataframe(
        pd.DataFrame({"other": [2]}, index=pd.Index(["a"], name="right_id")),
        pf.Schema(
            fields=(
                pf.IndexField(name="right_id"),
                pf.ValueField(name="other", dtype=int),
            )
        ),
    )

    result = pf.concat_columns(left, right)

    assert pf.primary_index_identity(result) != pf.primary_index_identity(left)
    assert pf.primary_index_identity(result) != pf.primary_index_identity(right)
    right_index = result.schema.get("right_id")
    assert isinstance(right_index, pf.IndexColumnField)
    assert right_index.index_identity == pf.primary_index_identity(right)


def test_index_identity_transition_declarations_are_specific():
    assert pf.PlanOperator.transitions.index_identity.mode == "mint"
    assert pf.concat_rows.transitions.index_identity.mode == "row_stack"
    assert pf.concat_columns.transitions.index_identity.mode == "align_rows"
    assert pf.join.transitions.index_identity.mode == "mint"
    assert pf.merge.transitions.index_identity.mode == "inherit"


def test_join_plan_mapping_fields_reference_input_identities():
    left = _dataset(["a"])
    right = _dataset(["a"])

    plan = pf.join(left, right)

    left_index = plan.schema.get("left_index")
    right_index = plan.schema.get("right_index")
    assert isinstance(left_index, pf.ForeignIndexField)
    assert isinstance(right_index, pf.ForeignIndexField)
    assert left_index.index_identity == pf.primary_index_identity(left)
    assert left_index.target_identity == pf.primary_index_identity(left)
    assert right_index.index_identity == pf.primary_index_identity(right)


def test_merge_output_identity_inherits_join_plan_identity():
    left = _dataset(["a"])
    right = _dataset(["a"])
    plan = pf.join(left, right)

    result = pf.merge(left, right, plan, collision="keep")

    assert pf.primary_index_identity(result) == pf.primary_index_identity(plan)


def test_merge_rejects_plan_referencing_different_input_identity():
    left = _dataset(["a"])
    right = _dataset(["a"])
    plan = pf.join(left, right)
    unrelated_left = _dataset(["a"])

    with pytest.raises(ValueError, match="left dataset index identity"):
        pf.merge(unrelated_left, right, plan, collision="keep")


def test_window_expansion_plan_source_index_references_source_identity():
    x = pf.IndexDimension(name="x")
    table = pd.DataFrame(
        {"x0": [0], "x1": [4]},
        index=pd.Index(["a"], name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.DimensionField.from_dim(x, "x0", dtype=int),
            pf.DimensionField.from_dim(x, "x1", dtype=int),
        )
    )
    ds = pf.Dataset(state=pf.DatasetState(schema=schema, table=table))

    plan = pf.window_expansion_plan(
        ds,
        bindings={"x": ("x0", "x1")},
        windows={"x": pf.AxisWindow(2, 2)},
    )

    source_index = plan.schema.get("source_index")
    assert isinstance(source_index, pf.ForeignIndexField)
    assert source_index.index_identity == pf.primary_index_identity(ds)


def test_foreign_index_field_requires_target_identity():
    with pytest.raises(ValueError, match="requires index_identity"):
        pf.ForeignIndexField(name="source_index")


def test_resolve_foreign_index_field_finds_unique_target():
    source = _dataset(["a"])
    other = _dataset(["a"])
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="plan_id"),
            pf.ForeignIndexField(
                name="source_index",
                index_identity=pf.primary_index_identity(source),
            ),
            pf.ForeignIndexField(
                name="other_index",
                index_identity=pf.primary_index_identity(other),
            ),
        )
    )

    field = pf.resolve_foreign_index_field(
        schema,
        pf.primary_index_identity(source),
    )

    assert field.name == "source_index"


def test_resolve_foreign_index_field_requires_explicit_name_when_ambiguous():
    source = _dataset(["a"])
    identity = pf.primary_index_identity(source)
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="plan_id"),
            pf.ForeignIndexField(name="source_index", index_identity=identity),
            pf.ForeignIndexField(name="parent_index", index_identity=identity),
        )
    )

    with pytest.raises(ValueError, match="expected exactly one"):
        pf.resolve_foreign_index_field(schema, identity)

    field = pf.resolve_foreign_index_field(
        schema,
        identity,
        field_name="parent_index",
    )
    assert field.name == "parent_index"
