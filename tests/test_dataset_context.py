"""Tests for the explicit mutable DatasetContext authoring substrate."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _dataset(values: list[int]) -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"value": pd.array(values, dtype="Int64")}),
        pf.Schema(fields=(pf.ValueField(name="value", dtype=int),)),
    )


def _indexed_dataset(values: list[int]) -> pf.Dataset:
    labels = [f"item-{index}" for index in range(len(values))]
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"value": pd.array(values, dtype="Int64")},
            index=pd.Index(labels, name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )


def test_dataset_context_is_inactive_by_default():
    assert pf.get_active_dataset_context() is None


def test_dataset_context_is_ambient_inside_with_block():
    ctx = pf.DatasetContext(_dataset([1, 2]))

    with ctx:
        assert pf.get_active_dataset_context() is ctx

    assert pf.get_active_dataset_context() is None


def test_nested_dataset_context_restores_outer_context():
    outer = pf.DatasetContext(_dataset([1]))
    inner = pf.DatasetContext(_dataset([2]))

    with outer:
        with inner:
            assert pf.get_active_dataset_context() is inner
        assert pf.get_active_dataset_context() is outer


def test_dataset_context_adopt_and_branch_are_explicit():
    initial = _dataset([1])
    updated = pf.assign(initial, value=[2])
    ctx = initial.context()
    branch = ctx.branch()

    ctx.adopt(updated)

    assert ctx.dataset is updated
    assert branch.dataset is initial


def test_dataset_context_field_handle_follows_rename_by_identity():
    ctx = _dataset([1]).context()
    value = ctx.field("value")

    with ctx:
        result = pf.rename({"value": "score"})

    assert ctx.dataset is result
    assert value.name == "score"


def test_dataset_context_field_handle_fails_after_drop():
    ctx = _dataset([1]).context()
    value = ctx.field("value")

    with ctx:
        pf.drop(["value"])

    with pytest.raises(ValueError, match="no longer exists"):
        _ = value.name


def test_dataset_context_ambient_unary_dispatch_advances_current_snapshot():
    ctx = _dataset([1, 2]).context()

    with ctx:
        result = pf.where(lambda table: table["value"] > 1)

    assert ctx.dataset is result
    assert ctx.dataset.table["value"].tolist() == [2]


def test_dataset_context_instance_bound_unary_dispatch_advances_current_snapshot():
    ctx = _dataset([1, 2]).context()

    result = pf.where.instance(dataset_context=ctx)(
        lambda table: table["value"] > 1
    )

    assert ctx.dataset is result
    assert ctx.dataset.table["value"].tolist() == [2]


def test_dataset_context_ambient_assign_dispatch_advances_current_snapshot():
    ctx = _dataset([1]).context()

    with ctx:
        result = pf.assign(value=[2], label=["updated"])

    assert ctx.dataset is result
    assert ctx.dataset.table["value"].tolist() == [2]
    assert ctx.dataset.table["label"].tolist() == ["updated"]


def test_dataset_context_composition_advances_when_current_snapshot_is_explicit_input():
    initial = _dataset([1])
    labels = pf.make_from_dataframe(
        pd.DataFrame({"label": ["one"]}),
        pf.Schema(fields=(pf.ValueField(name="label", dtype=str),)),
    )
    ctx = initial.context()

    with ctx:
        result = pf.concat(ctx.dataset, labels, axis=1)

    assert ctx.dataset is result
    assert ctx.dataset.schema.names() == ("value", "label")


def test_dataset_context_bound_concat_propagates_context_to_dispatched_operator():
    initial = _dataset([1])
    labels = pf.make_from_dataframe(
        pd.DataFrame({"label": ["one"]}),
        pf.Schema(fields=(pf.ValueField(name="label", dtype=str),)),
    )
    ctx = initial.context()

    result = pf.concat.instance(dataset_context=ctx)(ctx.dataset, labels, axis=1)

    assert ctx.dataset is result


def test_dataset_context_join_returns_sibling_plan_without_advancing():
    left = _indexed_dataset([1])
    right = _indexed_dataset([2])
    ctx = left.context()

    with ctx:
        plan = pf.join(ctx.dataset, right)

    assert ctx.dataset is left
    assert plan.schema.names() == ("join_id", "left_index", "right_index")


def test_dataset_context_creation_returns_sibling_without_advancing():
    initial = _dataset([1])
    ctx = initial.context()

    with ctx:
        created = _dataset([2])

    assert ctx.dataset is initial
    assert created is not initial


def test_dataset_context_explode_uses_ambient_source_and_advances():
    source = _indexed_dataset([1, 2])
    plan = pf.make_plan(source, ["item-1"])
    plan = pf.assign(plan, value=[20])
    ctx = source.context()

    with ctx:
        result = pf.explode(plan)

    assert ctx.dataset is result
    assert ctx.dataset.table["value"].tolist() == [20]


def test_operator_dataset_context_parameter_is_inherited_by_every_family():
    for operator in (
        pf.where,
        pf.make_from_dataframe,
        pf.make_plan,
        pf.join,
        pf.explode,
    ):
        assert "dataset_context" in operator.parameter_names()


def test_operator_resolves_ambient_dataset_context():
    ctx = _dataset([1]).context()

    with ctx:
        assert pf.where.instance().resolve_dataset_context() is ctx


def test_explicit_operator_dataset_context_overrides_ambient_context():
    ambient = _dataset([1]).context()
    explicit = _dataset([2]).context()
    operator = pf.where.instance(dataset_context=explicit)

    with ambient:
        assert operator.resolve_dataset_context() is explicit
