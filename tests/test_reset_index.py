"""Tests for reset_index (decompose) and set_index's composite handling.

Index<->column conversion is field-owned (``to_data_fields``), so ``reset_index``
and ``set_index``'s demote are polymorphic — no ``isinstance`` on the index, no
single-level assumption. ``reset_index`` is the sanctioned way *out* of a
composite index, and a ``ForeignIndexField`` level survives the decompose so a
subsequent rollup re-aligns by identity (the Stage D pattern).
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.fields import ForeignIndexField, IndexColumnField, IndexField
from patchframe.dataset.identity import primary_index_field, primary_index_identity


def _single() -> pf.Dataset:
    df = pd.DataFrame({"v": [1, 2, 3]}, index=pd.Index([10, 11, 12], name="id"))
    return pf.make_from_dataframe(
        df, pf.Schema(fields=(pf.IndexField(name="id"), pf.ValueField(name="v", dtype=int)))
    )


def _composite(creatives=None) -> tuple[pf.Dataset, object]:
    creatives = creatives or pf.new_index_identity()
    idx = pd.MultiIndex.from_tuples(
        [("cr0", "s1"), ("cr0", "s2"), ("cr1", "s1")], names=["creative_id", "segment_id"]
    )
    df = pd.DataFrame({"conversions": [3, 5, 2]}, index=idx)
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
    return pf.make_from_dataframe(df, schema), creatives


class TestResetIndex:
    def test_single_index_to_column(self):
        out = pf.reset_index(_single())
        assert out.schema.names() == ("index", "id", "v")
        assert isinstance(out.schema.get("id"), IndexColumnField)
        assert isinstance(primary_index_field(out.schema), IndexField)
        assert out.table["id"].tolist() == [10, 11, 12]

    def test_composite_decomposes_to_level_columns(self):
        ds, _ = _composite()
        out = pf.reset_index(ds)
        assert out.schema.names() == ("index", "creative_id", "segment_id", "conversions")
        assert not isinstance(out.table.index, pd.MultiIndex)
        assert out.table["creative_id"].tolist() == ["cr0", "cr0", "cr1"]

    def test_foreign_level_reference_survives(self):
        ds, _ = _composite()
        out = pf.reset_index(ds)
        # The per-level reference is preserved — the basis for the rollup.
        assert isinstance(out.schema.get("creative_id"), ForeignIndexField)

    def test_reset_then_rollup_realigns_by_identity(self):
        # The Stage D pattern: composite -> reset -> partition by a level column,
        # which re-inherits the level's foreign identity.
        ds, creatives = _composite()
        rolled = pf.partition(pf.reset_index(ds), "creative_id")
        assert sorted(rolled.table.index) == ["cr0", "cr1"]
        assert primary_index_identity(rolled.state) == creatives

    def test_custom_index_name(self):
        out = pf.reset_index(_single(), index_name="row")
        assert primary_index_field(out.schema).name == "row"

    def test_name_collision_rejected(self):
        with pytest.raises(ValueError, match="collides"):
            pf.reset_index(_single(), index_name="v")


class TestSetIndexComposite:
    def test_set_index_on_composite_current_rejected(self):
        ds, _ = _composite()
        with pytest.raises(TypeError, match="composite.*reset_index"):
            pf.set_index(ds, "conversions")

    def test_set_index_single_still_works(self):
        # Regression: the polymorphic demote keeps the single-index path intact.
        df = pd.DataFrame(
            {"k": ["a", "b", "c"], "v": [1, 2, 3]},
            index=pd.Index([10, 11, 12], name="id"),
        )
        ds = pf.make_from_dataframe(
            df,
            pf.Schema(
                fields=(
                    pf.IndexField(name="id"),
                    pf.ValueField(name="k", dtype=str),
                    pf.ValueField(name="v", dtype=int),
                )
            ),
        )
        out = pf.set_index(ds, "k")
        assert primary_index_field(out.schema).name == "k"
        assert isinstance(out.schema.get("id"), IndexColumnField)  # old index demoted
