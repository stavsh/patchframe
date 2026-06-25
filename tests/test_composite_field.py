"""Tests for CompositeField — the column-grouping field (Stage A).

A ``CompositeField`` groups N native columns under one logical schema field via
an index-less sub-schema; the columns are named ``{composite}.{subfield}``. This
stage is purely additive (a new field type + a ``validate_table`` branch) — no
operator or index changes — so it covers construction/validation, the dotted
column mapping, the fail-loud guards, and that single-field datasets are
unaffected. The index variant (``CompositeIndexField``) is Stage B.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.fields import CompositeField


def _location_sub_schema() -> pf.Schema:
    return pf.Schema(
        fields=(
            pf.ValueField(name="lat", dtype=float),
            pf.ValueField(name="lon", dtype=float),
        )
    )


def _located_schema() -> pf.Schema:
    return pf.Schema(
        fields=(
            pf.IndexField(name="id"),
            pf.CompositeField(name="location", sub_schema=_location_sub_schema()),
            pf.ValueField(name="label", dtype=str),
        )
    )


def _located_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location.lat": [1.0, 2.0, 3.0],
            "location.lon": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        },
        index=pd.Index([0, 1, 2], name="id"),
    )


def _located() -> pf.Dataset:
    return pf.make_from_dataframe(_located_table(), _located_schema())


class TestCompositeField:
    def test_builds_and_validates(self):
        ds = _located()
        assert ds.schema.has("location")
        assert isinstance(ds.schema.get("location"), CompositeField)
        # The dotted columns are real, native pandas columns.
        assert list(ds.table["location.lat"]) == [1.0, 2.0, 3.0]

    def test_column_names(self):
        field = pf.CompositeField(name="location", sub_schema=_location_sub_schema())
        assert field.column_names() == ("location.lat", "location.lon")

    def test_missing_dotted_column_fails_loud(self):
        table = _located_table().drop(columns=["location.lon"])
        with pytest.raises(ValueError, match=r"missing.*location\.lon"):
            pf.make_from_dataframe(table, _located_schema())

    def test_wrong_subfield_dtype_fails_loud(self):
        table = _located_table()
        table["location.lat"] = ["x", "y", "z"]  # str, not float
        with pytest.raises(ValueError, match="lat"):
            pf.make_from_dataframe(table, _located_schema())

    def test_index_in_sub_schema_rejected(self):
        with pytest.raises(ValueError, match="index-less"):
            pf.CompositeField(
                name="bad",
                sub_schema=pf.Schema(fields=(pf.IndexField(name="k"),)),
            )

    def test_empty_sub_schema_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            pf.CompositeField(name="bad", sub_schema=pf.Schema(fields=()))

    def test_validate_column_on_composite_raises(self):
        field = pf.CompositeField(name="location", sub_schema=_location_sub_schema())
        with pytest.raises(TypeError, match="spans multiple columns"):
            field.validate_column(pd.Series([1.0, 2.0]))

    def test_native_access_is_unchanged(self):
        # The whole point: the grouping is schema-level; the table is native.
        ds = _located()
        assert ds.table["location.lat"].tolist() == [1.0, 2.0, 3.0]
        assert ds.table["label"].tolist() == ["a", "b", "c"]

    def test_coexists_with_plain_fields(self):
        # A missing *plain* field is still reported alongside composites.
        table = _located_table().drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            pf.make_from_dataframe(table, _located_schema())


class TestAdditive:
    def test_single_field_dataset_unaffected(self):
        # Regression: a dataset with no CompositeField behaves exactly as before.
        df = pd.DataFrame({"v": [1, 2, 3]}, index=pd.Index([0, 1, 2], name="id"))
        schema = pf.Schema(
            fields=(pf.IndexField(name="id"), pf.ValueField(name="v", dtype=int))
        )
        ds = pf.make_from_dataframe(df, schema)
        assert ds.schema.names() == ("id", "v")
        assert ds.table["v"].tolist() == [1, 2, 3]

    def test_explicit_index_attempt_rejected(self):
        with pytest.raises(ValueError, match="cannot be primary"):
            pf.CompositeField(
                name="location", sub_schema=_location_sub_schema(), primary=True
            )


def _located2() -> pf.Dataset:
    table = _located_table()
    table.index = pd.Index([3, 4, 5], name="id")  # disjoint, for row-stacking
    return pf.make_from_dataframe(table, _located_schema())


def _to_upper(value: str) -> str:
    return str(value).upper()


class TestComposition:
    # Composition maps field->column 1:1 by name; a CompositeField spans N dotted
    # columns, so composition must fail loud (never the silent phantom-column
    # corruption) until the 1->N mapping lands.
    def test_concat_rows_fails_loud(self):
        with pytest.raises(NotImplementedError, match="composition"):
            pf.concat_rows(_located(), _located2())

    def test_concat_columns_fails_loud(self):
        with pytest.raises(NotImplementedError, match="composition"):
            pf.concat_columns(_located(), _located())


class TestCoupling:
    # A composite is opaque to couplings in v1: its name is not a real column and
    # its sub-columns are not top-level fields.
    def test_map_fields_rejects_composite_input(self):
        with pytest.raises(TypeError, match="CompositeField"):
            pf.map_fields(_located(), ["location"], _to_upper, out="x")

    def test_map_fields_rejects_subcolumn_ref(self):
        # "location.lat" is a real column but not a top-level schema field.
        with pytest.raises(ValueError, match="not in schema"):
            pf.map_fields(_located(), ["location.lat"], _to_upper, out="x")

    def test_coupling_on_plain_column_unaffected(self):
        # A composite in the schema does not break ordinary couplings on plain
        # columns (additive).
        out = pf.map_fields(_located(), ["label"], _to_upper, out="upper")
        assert out.table["upper"].tolist() == ["A", "B", "C"]
        assert out.schema.has("location")  # composite carried through untouched


class TestAtomicOps:
    # Generic column ops treat the composite as one unit, via the field's
    # table_columns()/rename_table_columns() — no isinstance in the operators.
    def test_keep_keeps_all_composite_columns(self):
        out = pf.keep(_located(), ["id", "location"])
        assert out.schema.names() == ("id", "location")
        assert "location.lat" in out.table.columns
        assert "location.lon" in out.table.columns
        assert "label" not in out.table.columns

    def test_drop_removes_all_composite_columns(self):
        out = pf.drop(_located(), ["location"])
        assert out.schema.names() == ("id", "label")
        assert "location.lat" not in out.table.columns
        assert "location.lon" not in out.table.columns
        assert "label" in out.table.columns

    def test_rename_reprefixes_composite_columns(self):
        out = pf.rename(_located(), {"location": "geo"})
        field = out.schema.get("geo")
        assert isinstance(field, CompositeField)
        assert field.column_names() == ("geo.lat", "geo.lon")
        assert out.table["geo.lat"].tolist() == [1.0, 2.0, 3.0]
        assert "location.lat" not in out.table.columns

    def test_keep_rejects_subcolumn(self):
        with pytest.raises(ValueError, match="not in schema"):
            pf.keep(_located(), ["id", "location.lat"])

    def test_drop_rejects_subcolumn(self):
        with pytest.raises(ValueError, match="not in schema"):
            pf.drop(_located(), ["location.lat"])

    def test_rename_rejects_subfield(self):
        with pytest.raises(ValueError, match="not in schema"):
            pf.rename(_located(), {"location.lat": "x"})


class TestOperatorAudit:
    # A composite must never be silently corrupted by ops that assume 1:1
    # field<->column. Capability ("spans one column") is asked of the field via
    # table_columns(), not isinstance'd.
    def test_assign_to_composite_rejected(self):
        # Was silent: assign created a phantom "location" column.
        with pytest.raises(ValueError, match="spans multiple table columns"):
            _located().assign(location=[9, 9, 9])

    def test_assign_subcolumn_name_rejected_by_schema(self):
        # Was silent: a 2nd field claiming location.lat. The schema invariant
        # (no overlapping claimed columns) catches it.
        with pytest.raises(ValueError, match="overlapping table columns"):
            _located().assign(**{"location.lat": [9.0, 9.0, 9.0]})

    def test_partition_by_composite_rejected(self):
        with pytest.raises(TypeError, match="spans .* table columns"):
            pf.partition(_located(), "location")

    def test_set_index_composite_rejected(self):
        with pytest.raises(TypeError, match="spans multiple table columns"):
            pf.set_index(_located(), "location")

    def test_schema_rejects_overlapping_claimed_columns(self):
        # A stray field claiming a composite's dotted column is a schema error.
        with pytest.raises(ValueError, match="overlapping table columns"):
            pf.Schema(
                fields=(
                    pf.IndexField(name="id"),
                    pf.CompositeField(name="location", sub_schema=_location_sub_schema()),
                    pf.ValueField(name="location.lat", dtype=float),
                )
            )

    def test_where_carries_composite_through(self):
        # Row filter is column-agnostic — the composite survives untouched (safe).
        out = pf.where(_located(), _located().table["label"] == "a")
        assert out.schema.has("location")
        assert list(out.table.index) == [0]
