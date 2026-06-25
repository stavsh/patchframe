"""Tests for CompositeIndexField — the composite row identity (Stage B).

A ``CompositeIndexField`` is the index-axis counterpart of ``CompositeField``: a
separate type subclassing ``IndexField`` whose index-less ``sub_schema`` fields
are the pandas ``MultiIndex`` levels. It carries one composite ``IndexIdentity``;
the row identity is the tuple, unique. These cover construction/validation, the
identity-helper compatibility (why subclassing works), the inviolable uniqueness,
and the *critical* fail-loud composition (not a silent MRO fallback to the
single-level ``IndexFieldCompositionPolicy``).
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.fields import CompositeIndexField, IndexField
from patchframe.dataset.identity import primary_index_field, primary_index_identity


def _cell_sub_schema() -> pf.Schema:
    return pf.Schema(
        fields=(
            pf.ValueField(name="creative_id", dtype=str),
            pf.ValueField(name="segment_id", dtype=str),
        )
    )


def _cell_schema() -> pf.Schema:
    return pf.Schema(
        fields=(
            pf.CompositeIndexField(name="cell", sub_schema=_cell_sub_schema()),
            pf.ValueField(name="conversions", dtype=int),
        )
    )


def _cell_table() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [("cr0", "s1"), ("cr0", "s2"), ("cr1", "s1")],
        names=["creative_id", "segment_id"],
    )
    return pd.DataFrame({"conversions": [3, 5, 2]}, index=idx)


def _celled() -> pf.Dataset:
    return pf.make_from_dataframe(_cell_table(), _cell_schema())


class TestCompositeIndexField:
    def test_builds_native_multiindex(self):
        ds = _celled()
        assert isinstance(ds.table.index, pd.MultiIndex)
        assert list(ds.table.index.names) == ["creative_id", "segment_id"]
        assert list(ds.table["conversions"]) == [3, 5, 2]

    def test_is_the_one_index_field(self):
        # Subclassing IndexField is why the single-index machinery is untouched.
        ds = _celled()
        field = primary_index_field(ds.schema)
        assert isinstance(field, CompositeIndexField)
        assert isinstance(field, IndexField)
        assert field.level_names() == ("creative_id", "segment_id")

    def test_occupies_no_columns(self):
        field = pf.CompositeIndexField(name="cell", sub_schema=_cell_sub_schema())
        assert field.table_columns() == ()  # it is the index, not columns

    def test_identity_minted(self):
        ds = _celled()
        assert primary_index_identity(ds.state) is not None

    def test_tuple_uniqueness_enforced(self):
        # The one inviolable invariant — a duplicate tuple is rejected.
        dup = pd.DataFrame(
            {"conversions": [1, 2]},
            index=pd.MultiIndex.from_tuples(
                [("a", "b"), ("a", "b")], names=["creative_id", "segment_id"]
            ),
        )
        with pytest.raises(ValueError):
            pf.make_from_dataframe(dup, _cell_schema())

    def test_composition_fails_loud_not_silent(self):
        # CRITICAL: must hit CompositeIndexFieldCompositionPolicy, not fall back
        # via MRO to the single-level IndexFieldCompositionPolicy.
        with pytest.raises(NotImplementedError, match="composite index"):
            pf.concat_rows(_celled(), _celled())

    def test_partition_by_composite_index_rejected(self):
        # It is the primary index, so it cannot be a partition key.
        with pytest.raises(TypeError, match="primary index"):
            pf.partition(_celled(), "cell")


class TestValidation:
    def test_rejects_single_level_index(self):
        field = pf.CompositeIndexField(name="cell", sub_schema=_cell_sub_schema())
        single = pd.DataFrame({"conversions": [1]}, index=pd.Index(["x"], name="cell"))
        with pytest.raises(ValueError, match="MultiIndex"):
            field.validate_in_table(single)

    def test_rejects_wrong_level_names(self):
        field = pf.CompositeIndexField(name="cell", sub_schema=_cell_sub_schema())
        wrong = pd.DataFrame(
            {"conversions": [1]},
            index=pd.MultiIndex.from_tuples([("a", "b")], names=["x", "y"]),
        )
        with pytest.raises(ValueError, match="do not match the sub-schema"):
            field.validate_in_table(wrong)

    def test_index_in_sub_schema_rejected(self):
        with pytest.raises(ValueError, match="index-less"):
            pf.CompositeIndexField(
                name="cell",
                sub_schema=pf.Schema(fields=(pf.IndexField(name="k"),)),
            )

    def test_empty_sub_schema_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            pf.CompositeIndexField(name="cell", sub_schema=pf.Schema(fields=()))


class TestForeignLevel:
    def test_foreign_index_field_level(self):
        # A level may be a ForeignIndexField (the per-component reference that
        # the per-level-identity choice buys).
        creatives = pf.new_index_identity()
        sub = pf.Schema(
            fields=(
                pf.ForeignIndexField(name="creative_id", dtype=str, index_identity=creatives),
                pf.ValueField(name="segment_id", dtype=str),
            )
        )
        schema = pf.Schema(
            fields=(
                pf.CompositeIndexField(name="cell", sub_schema=sub),
                pf.ValueField(name="conversions", dtype=int),
            )
        )
        ds = pf.make_from_dataframe(_cell_table(), schema)
        field = primary_index_field(ds.schema)
        assert field.level_names() == ("creative_id", "segment_id")
