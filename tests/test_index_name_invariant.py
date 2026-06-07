"""The table index must be named after its IndexField (self-describing identity).

Covers IndexField.validate_column enforcement, make_from_dataframe naming the
index, and rename keeping the index axis consistent with a renamed IndexField.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _schema() -> pf.Schema:
    return pf.Schema(
        fields=(pf.IndexField(name="item_id"), pf.ValueField(name="v", dtype=int))
    )


def test_index_field_validate_column_requires_matching_index_name():
    field = pf.IndexField(name="item_id")

    field.validate_column(pd.Series(["a", "b"], name="item_id"))  # matches: ok

    with pytest.raises(ValueError, match="must be named 'item_id'"):
        field.validate_column(pd.Series(["a", "b"], name="wrong"))
    with pytest.raises(ValueError, match="must be named 'item_id'"):
        field.validate_column(pd.Series(["a", "b"], name=None))


def test_schema_validate_table_rejects_unnamed_index():
    bad = pd.DataFrame({"v": [1]}, index=pd.Index(["a"]))  # unnamed index
    with pytest.raises(ValueError, match="must be named 'item_id'"):
        _schema().validate_table(bad)


def test_make_from_dataframe_names_unnamed_index_from_field():
    ds = pf.make_from_dataframe(
        pd.DataFrame({"v": [1, 2]}, index=["a", "b"]),  # unnamed input index
        _schema(),
    )
    assert ds.table.index.name == "item_id"


def test_make_from_dataframe_renames_mismatched_index_to_field():
    ds = pf.make_from_dataframe(
        pd.DataFrame({"v": [1]}, index=pd.Index(["a"], name="foo")),
        _schema(),
    )
    assert ds.table.index.name == "item_id"


def test_rename_renames_index_axis_consistently():
    ds = pf.make_from_dataframe(pd.DataFrame({"v": [1, 2]}, index=["a", "b"]), _schema())

    renamed = pf.rename(ds, {"item_id": "row_id"})

    assert renamed.schema.has("row_id")
    assert renamed.table.index.name == "row_id"  # axis renamed to match the field
    # And the result re-validates cleanly under the invariant.
    renamed.schema.validate_table(renamed.table)
