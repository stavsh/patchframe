"""Tests for composite-key partition/reduce and the null_keys policy (Stage C).

``partition(ds, by=[...])`` / ``reduce(ds, by=[...])`` produce a composite-indexed
dataset (a ``CompositeIndexField`` over a native ``MultiIndex``), the levels being
the key columns' own fields. With ``reset_index`` this closes the rollup loop
(composite grain -> decompose -> roll up by a level).
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.fields import CompositeIndexField, ForeignIndexField, ValueField
from patchframe.dataset.identity import primary_index_field, primary_index_identity


def _events(creative_field=None) -> pf.Dataset:
    creative_field = creative_field or pf.ValueField(name="creative_id", dtype=str)
    df = pd.DataFrame(
        {
            "creative_id": ["cr0", "cr0", "cr1", "cr1", "cr1"],
            "segment_id": ["s1", "s2", "s1", "s1", "s2"],
            "conv": [1, 2, 3, 4, 5],
        },
        index=pd.Index(range(5), name="row"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="row"),
            creative_field,
            pf.ValueField(name="segment_id", dtype=str),
            pf.ValueField(name="conv", dtype=int),
        )
    )
    return pf.make_from_dataframe(df, schema)


class TestCompositePartition:
    def test_produces_composite_index(self):
        parts = pf.partition(_events(), ["creative_id", "segment_id"])
        field = primary_index_field(parts.schema)
        assert isinstance(field, CompositeIndexField)
        assert field.level_names() == ("creative_id", "segment_id")
        assert isinstance(parts.table.index, pd.MultiIndex)
        assert len(parts.table) == 4  # (cr0,s1),(cr0,s2),(cr1,s1),(cr1,s2)

    def test_levels_are_the_source_fields(self):
        # A ForeignIndexField key column stays a ForeignIndexField level (so a
        # later reset + rollup re-aligns by identity).
        creatives = pf.new_index_identity()
        ds = _events(pf.ForeignIndexField(name="creative_id", dtype=str, index_identity=creatives))
        field = primary_index_field(pf.partition(ds, ["creative_id", "segment_id"]).schema)
        level = field.sub_schema.get("creative_id")
        assert isinstance(level, ForeignIndexField)
        assert level.index_identity == creatives

    def test_into_collides_with_level(self):
        with pytest.raises(ValueError, match="collides"):
            pf.partition(_events(), ["creative_id", "segment_id"], into="creative_id")

    def test_composite_domain_rejected(self):
        with pytest.raises(TypeError, match="domain.* composite"):
            pf.partition(_events(), ["creative_id", "segment_id"], domain=_events())


class TestCompositeReduce:
    def test_aggregates_per_cell(self):
        out = pf.reduce(
            _events(),
            ["creative_id", "segment_id"],
            aggs={"total": pf.Sum.on("conv"), "n": pf.Count.on()},
        )
        assert isinstance(out.table.index, pd.MultiIndex)
        table = out.table.sort_index()
        assert table.loc[("cr1", "s1"), "total"] == 7.0  # 3 + 4
        assert table.loc[("cr1", "s1"), "n"] == 2
        assert table.loc[("cr0", "s2"), "total"] == 2.0

    def test_reduce_reset_rollup(self):
        # The Stage D loop: composite grain -> decompose -> roll up by a level.
        cells = pf.reduce(_events(), ["creative_id", "segment_id"], aggs={"cells": pf.Sum.on("conv")})
        flat = pf.reset_index(cells)
        rolled = pf.reduce(flat, "creative_id", aggs={"total": pf.Sum.on("cells")})
        totals = rolled.table.sort_index()["total"]
        assert totals.loc["cr0"] == 3.0   # 1 + 2
        assert totals.loc["cr1"] == 12.0  # 7 + 5


class TestNullKeys:
    def _with_null(self) -> pf.Dataset:
        ds = _events()
        table = ds.table.copy()
        table.loc[0, "segment_id"] = None
        return ds.replace_state(table=table)

    def test_error_is_default(self):
        with pytest.raises(ValueError, match="null value"):
            pf.partition(self._with_null(), ["creative_id", "segment_id"])

    def test_drop(self):
        parts = pf.partition(self._with_null(), ["creative_id", "segment_id"], null_keys="drop")
        # Row 0 (cr0, <null>) is dropped; (cr0,s2),(cr1,s1),(cr1,s2) remain.
        assert len(parts.table) == 3

    def test_group_not_implemented(self):
        with pytest.raises(NotImplementedError, match="group"):
            pf.partition(self._with_null(), ["creative_id", "segment_id"], null_keys="group")

    def test_single_key_drop_still_works(self):
        ds = self._with_null()
        # Single-key null drop (segment_id has one null).
        parts = pf.partition(ds, "segment_id", null_keys="drop")
        assert sorted(parts.table.index) == ["s1", "s2"]
