"""Tests for MergedField-based composition (B2)."""

from __future__ import annotations

import pandas as pd

import patchframe as pf


def _ds(index_name: str, *cols: str) -> pf.Dataset:
    table = pd.DataFrame(
        {col: [1, 2] for col in cols},
        index=pd.Index(["a", "b"], name=index_name),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=index_name),
            *(pf.ValueField(name=col, dtype=int) for col in cols),
        )
    )
    return pf.make_from_dataframe(table, schema)


def test_concat_columns_collision_prunes_losing_input_couplings():
    # both inputs declare "tag"; each carries a distinct coupling referencing it
    left = pf.bind_slice(_ds("left_id", "tag", "lval"), "tag", "lval")
    right = pf.bind_slice(_ds("right_id", "tag", "rval"), "tag", "rval")

    # collision side defaults to "left": left's "tag" wins, right's loses
    result = pf.concat_columns(left, right, collision="keep")

    # the MergedField resolved away — no MergedField escapes into the schema
    assert isinstance(result.schema.get("tag"), pf.ValueField)

    # right's coupling referenced the superseded "tag" and was pruned;
    # the winner's coupling survives
    couplings = result.couplings.couplings
    assert len(couplings) == 1
    assert couplings[0].data_field.name == "lval"


def test_concat_columns_without_collision_unions_couplings():
    left = pf.bind_slice(_ds("left_id", "sa", "da"), "sa", "da")
    right = pf.bind_slice(_ds("right_id", "sb", "db"), "sb", "db")

    # no shared field names — no collision, both couplings survive
    result = pf.concat_columns(left, right)

    assert len(result.couplings.couplings) == 2


def test_merge_collision_prunes_losing_input_couplings():
    left = pf.bind_slice(_ds("item_id", "tag", "lval"), "tag", "lval")
    right = pf.bind_slice(_ds("item_id", "tag", "rval"), "tag", "rval")
    plan = pf.join(left, right)

    # "tag" collides; collision side defaults left, so right's "tag" loses
    result = pf.merge(left, right, plan, collision="keep")

    couplings = result.couplings.couplings
    assert len(couplings) == 1
    assert couplings[0].data_field.name == "lval"


def _row_ds(index_labels: list[str]) -> pf.Dataset:
    table = pd.DataFrame(
        {"value": list(range(len(index_labels)))},
        index=pd.Index(index_labels, name="item_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="item_id"),
            pf.ValueField(name="value", dtype=int),
        )
    )
    return pf.make_from_dataframe(table, schema)


def test_concat_rows_preserves_shared_field_identity():
    base = _row_ds(["a", "b", "c", "d"])
    # both chunks descend from `base` via where (schema preserved)
    chunk_a = pf.where(base, base.table["value"] < 2)
    chunk_b = pf.where(base, base.table["value"] >= 2)

    result = pf.concat_rows(chunk_a, chunk_b)

    assert (
        result.schema.get("value").field_identity
        == base.schema.get("value").field_identity
    )


def test_concat_rows_mints_field_identity_when_inputs_diverge():
    a = _row_ds(["a", "b"])
    b = _row_ds(["c", "d"])

    result = pf.concat_rows(a, b)

    merged = result.schema.get("value").field_identity
    assert merged != a.schema.get("value").field_identity
    assert merged != b.schema.get("value").field_identity
