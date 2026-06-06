"""Tests for the Dataset -> handle entry bridge (Phase 2).

``ds.field(...)`` / ``ds.fields([...])`` are the entry bridge from the eager
surface to the handle surface (lazy-and-bundle.md §1). Repeated calls on one
dataset share a context so several handles can feed a single operator.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.context import resolve_field_selectors


def _dataset() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {
                "value": pd.array([1, 2], dtype="Int64"),
                "value2": pd.array([3, 4], dtype="Int64"),
            }
        ),
        pf.Schema(
            fields=(
                pf.ValueField(name="value", dtype=int),
                pf.ValueField(name="value2", dtype=int),
            )
        ),
    )


def test_field_returns_resolving_handle():
    ds = _dataset()
    handle = ds.field("value")

    assert isinstance(handle, pf.FieldHandle)
    assert handle.name == "value"
    assert handle.resolve().name == "value"


def test_repeated_field_calls_share_one_context():
    ds = _dataset()

    assert ds.field("value").dataset_context is ds.field("value2").dataset_context


def test_field_handles_feed_a_multi_field_operator():
    ds = _dataset()

    call = pf.bind_slice.instance().normalize_call(ds.field("value"), ds.field("value2"))

    assert call.args == ("value", "value2")
    assert call.datasets == (ds,)
    assert len(call.reference_contexts) == 1


def test_fields_returns_a_selection():
    ds = _dataset()
    selection = ds.fields(["value", "value2"])

    assert isinstance(selection, pf.FieldSelection)
    assert len(selection) == 2
    assert selection.names() == ("value", "value2")
    assert [handle.name for handle in selection] == ["value", "value2"]
    assert selection[0].name == "value"
    assert tuple(field.name for field in selection.resolve()) == ("value", "value2")


def test_selection_shares_the_dataset_context():
    ds = _dataset()
    selection = ds.fields(["value", "value2"])

    assert selection.dataset_context is ds.field("value").dataset_context


def test_empty_selection_has_no_context():
    ds = _dataset()
    selection = ds.fields([])

    assert len(selection) == 0
    assert selection.dataset_context is None


def test_selection_rejects_handles_from_different_datasets():
    left = _dataset()
    right = _dataset()

    with pytest.raises(ValueError, match="share one DatasetContext"):
        pf.FieldSelection((left.field("value"), right.field("value")))


def test_field_prefers_an_ambient_context_pointing_at_this_dataset():
    ds = _dataset()
    ctx = ds.context()

    with ctx:
        handle = ds.field("value")

    assert handle.dataset_context is ctx


def test_selection_resolves_through_field_selector_helper():
    ds = _dataset()
    selection = ds.fields(["value", "value2"])

    resolved = resolve_field_selectors(selection, ds.schema, op_name="test")

    assert resolved == ("value", "value2")
