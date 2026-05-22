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
