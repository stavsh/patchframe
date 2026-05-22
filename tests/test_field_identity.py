"""Tests for semantic field identity propagation."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

import patchframe as pf
from patchframe.dataset.identity import FieldIdentity


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


def test_field_carries_identity_after_construction():
    field = pf.ValueField(name="x", dtype=int)
    assert isinstance(field.field_identity, FieldIdentity)


def test_field_identity_excluded_from_structural_equality():
    a = pf.ValueField(name="x", dtype=int)
    b = pf.ValueField(name="x", dtype=int)
    # independently constructed fields get distinct lineage identities
    assert a.field_identity != b.field_identity
    # but structural equality ignores lineage
    assert a == b


def test_replace_preserves_field_identity():
    field = pf.ValueField(name="x", dtype=int)
    assert replace(field, name="y").field_identity == field.field_identity


def test_rename_preserves_field_identity():
    ds = _dataset(["a", "b"], values=[1, 2])
    original = ds.schema.get("value").field_identity
    result = pf.rename(ds, {"value": "val"})
    assert result.schema.get("val").field_identity == original


def test_drop_preserves_surviving_field_identities():
    ds = _dataset(["a", "b"], values=[1, 2])
    original = ds.schema.get("item_id").field_identity
    result = pf.drop(ds, ["value"])
    assert result.schema.get("item_id").field_identity == original


def test_where_preserves_field_identities():
    ds = _dataset(["a", "b"], values=[1, 2])
    original = ds.schema.get("value").field_identity
    result = pf.where(ds, ds.table["value"] > 1)
    assert result.schema.get("value").field_identity == original


def test_set_index_preserves_field_lineage_through_rewrite():
    ds = _dataset(["a", "b"], values=[1, 2])
    value_identity = ds.schema.get("value").field_identity
    index_identity = ds.schema.get("item_id").field_identity

    result = pf.set_index(ds, "value")

    # the promoted column keeps its field identity through the rewrite
    assert result.schema.get("value").field_identity == value_identity
    # the downgraded index column keeps its field identity too
    assert result.schema.get("item_id").field_identity == index_identity
