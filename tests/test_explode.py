"""Tests for generic source-indexed row expansion."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _source() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "value": [1, 2],
            "label": ["left", "right"],
        },
        index=pd.Index(["a", "b"], name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.ValueField(name="value", dtype=int),
            pf.ValueField(name="label"),
        )
    )
    return pf.make_from_dataframe(table, schema)


def _plan(source: pf.Dataset, table: pd.DataFrame, *fields: pf.Field) -> pf.Dataset:
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="plan_id"),
            pf.ForeignIndexField(
                name="source_index",
                index_identity=pf.primary_index_identity(source),
            ),
            *fields,
        )
    )
    return pf.make_from_dataframe(table, schema)


def test_explode_gathers_source_rows_and_overlays_matching_plan_fields():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {
                "source_index": ["a", "a", "b"],
                "value": [10, 20, 30],
            },
            index=pd.RangeIndex(3, name="plan_id"),
        ),
        pf.ValueField(name="value", dtype=int),
    )

    result = pf.explode(source, plan)

    assert result.schema.names() == ("plan_id", "value", "label")
    assert result.table.index.equals(plan.table.index)
    assert result.table["value"].tolist() == [10, 20, 30]
    assert result.table["label"].tolist() == ["left", "left", "right"]
    assert "source_index" not in result.table.columns
    assert pf.primary_index_identity(result) == pf.primary_index_identity(plan)


def test_explode_overlays_existing_dimensioned_slice_field():
    source = _source()
    table = source.table.copy()
    table["window"] = pd.NA
    source = pf.make_from_dataframe(
        table,
        source.schema.add(pf.DimensionedSliceField(name="window")),
    )
    x = pf.IndexDimension(name="x")
    plan = _plan(
        source,
        pd.DataFrame(
            {
                "source_index": ["a", "b"],
                "window": [x.spec(0, 2), x.spec(2, 4)],
            },
            index=pd.RangeIndex(2, name="plan_id"),
        ),
        pf.DimensionedSliceField(name="window", nullable=False),
    )

    result = pf.explode(source, plan)

    assert [window.dims["x"] for window in result.table["window"]] == [
        slice(0, 2),
        slice(2, 4),
    ]


def test_explode_uses_explicit_foreign_index_field_when_inference_is_ambiguous():
    source = _source()
    identity = pf.primary_index_identity(source)
    plan = pf.make_from_dataframe(
        pd.DataFrame(
            {
                "source_index": ["a"],
                "parent_index": ["b"],
                "value": [10],
            },
            index=pd.RangeIndex(1, name="plan_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="plan_id"),
                pf.ForeignIndexField(name="source_index", index_identity=identity),
                pf.ForeignIndexField(name="parent_index", index_identity=identity),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )

    with pytest.raises(ValueError, match="expected exactly one"):
        pf.explode(source, plan)

    result = pf.explode(source, plan, foreign_index_field="source_index")

    assert result.table["label"].tolist() == ["left"]
    assert result.table["value"].tolist() == [10]


def test_explode_rejects_missing_source_labels():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": ["missing"], "value": [10]},
            index=pd.RangeIndex(1, name="plan_id"),
        ),
        pf.ValueField(name="value", dtype=int),
    )

    with pytest.raises(ValueError, match="missing from target dataset"):
        pf.explode(source, plan)


def test_explode_rejects_null_source_labels():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": [pd.NA], "value": [10]},
            index=pd.RangeIndex(1, name="plan_id"),
        ),
        pf.ValueField(name="value", dtype=int),
    )

    with pytest.raises(ValueError, match="contains null labels"):
        pf.explode(source, plan)


def test_explode_without_shared_fields_is_a_pure_gather():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": ["a", "b", "a"]},
            index=pd.RangeIndex(3, name="plan_id"),
        ),
    )

    result = pf.explode(source, plan)

    assert result.schema.names() == ("plan_id", "value", "label")
    assert result.table["value"].tolist() == [1, 2, 1]


def test_explode_rejects_explicitly_empty_overlay_fields():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame({"source_index": ["a"]}, index=pd.RangeIndex(1, name="plan_id")),
    )

    with pytest.raises(ValueError, match="resolved to no fields"):
        pf.explode(source, plan, overlay_fields=())


def test_plan_columns_carry_by_identity_alignment():
    # The blessed composition for plan-only columns (slice fields, the
    # source_index mapping): explode inherits the plan's index identity, so
    # plan columns attach afterwards via concat_columns — identity alignment,
    # no collision strategy needed (join-dimensions-identity.md §5).
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": ["a", "b", "a"], "score": [0.1, 0.2, 0.3]},
            index=pd.RangeIndex(3, name="plan_id"),
        ),
        pf.ValueField(name="score", dtype=float),
    )

    exploded = pf.explode(source, plan)
    carried = pf.concat_columns(
        exploded, pf.keep(plan, ["plan_id", "source_index", "score"])
    )

    assert carried.schema.names() == (
        "plan_id",
        "value",
        "label",
        "source_index",
        "score",
    )
    assert carried.table["source_index"].tolist() == ["a", "b", "a"]
    assert carried.table["score"].tolist() == [0.1, 0.2, 0.3]
    # The aligned namespaces unified: row identity is still the plan's.
    assert pf.primary_index_identity(carried) == pf.primary_index_identity(plan)


def test_explode_rejects_overlay_field_missing_from_source():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": ["a"], "score": [0.5]},
            index=pd.RangeIndex(1, name="plan_id"),
        ),
        pf.ValueField(name="score", dtype=float),
    )

    with pytest.raises(ValueError, match="not present in the source"):
        pf.explode(source, plan, overlay_fields=("score",))


def test_explode_warns_when_plan_has_couplings():
    source = _source()
    plan = _plan(
        source,
        pd.DataFrame(
            {"source_index": ["a"], "value": [10]},
            index=pd.RangeIndex(1, name="plan_id"),
        ),
        pf.ValueField(name="value", dtype=int),
    ).replace_state(couplings=pf.CouplingSet((pf.Materialize("value"),)))

    with pytest.warns(UserWarning, match="plan couplings are ignored"):
        result = pf.explode(source, plan)

    assert result.table["value"].tolist() == [10]
