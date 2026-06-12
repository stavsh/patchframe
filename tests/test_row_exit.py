"""Row access is the exit point: evaluate, then exit to plain Python.

ds[item_id] evaluates the row's pending couplings (ephemerally) and converts
every value through its field's exit (Field.exit_value, with the
register_field_exit registry for field types you do not own): a BundleField
fiber leaves as a list of records, recursively. The storage surface
(ds.table / ds["col"]) keeps framework objects — lazy fiber navigation lives
there. rows() is the positional, DataLoader-pluggable view over exited rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.fields import _FIELD_EXITS


def _double(a):
    return a * 2


def _clips() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"duration": [10.0, 20.0]},
            index=pd.Index(["a", "b"], name="clip_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.ValueField(name="duration", dtype=float),
            )
        ),
    )


def _segments() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"clip_id": ["a", "b", "a"], "text": ["t0", "t1", "t2"]},
            index=pd.Index(["s0", "s1", "s2"], name="seg_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="seg_id"),
                pf.ValueField(name="clip_id", dtype=str),
                pf.ValueField(name="text", dtype=str),
            )
        ),
    )


def _grouped() -> pf.Dataset:
    clips = _clips()
    groups = pf.partition(
        pf.link(_segments(), clips, "clip_id"),
        "clip_id",
        domain=clips,
        into="segments",
    )
    return pf.concat_columns(clips, pf.keep(groups, ["segments"]))


# -- the exit pass -------------------------------------------------------------


def test_row_access_exits_fiber_as_records():
    clips = _grouped()

    row = clips["a"]

    assert row["segments"] == [
        {"seg_id": "s0", "clip_id": "a", "text": "t0"},
        {"seg_id": "s2", "clip_id": "a", "text": "t2"},
    ]
    assert row["duration"] == 10.0


def test_storage_surface_keeps_the_fiber_dataset():
    clips = _grouped()

    # Lazy fiber navigation lives on the storage surface, not the exit point.
    fiber = clips.table.loc["a", "segments"]
    assert isinstance(fiber, pf.Dataset)
    column = clips["segments"]
    assert isinstance(column.iloc[0], pf.Dataset)


def test_exit_composes_with_evaluation_inside_the_fiber():
    # A pending coupling inside the fiber is evaluated by the fiber's own row
    # access during export — evaluation and exit compose recursively.
    members = pf.make_from_dataframe(
        pd.DataFrame(
            {"clip": ["a", "a"], "v": [1, 2]},
            index=pd.Index(["m0", "m1"], name="m"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="m"),
                pf.ValueField(name="clip", dtype=str),
                pf.ValueField(name="v", dtype=int),
            )
        ),
    )
    handle = pf.map_fields(members.fields(["v"]), _double, out="w")
    carrier = handle.dataset_context.dataset  # "w" pending

    groups = pf.partition(carrier, "clip", into="members")
    records = groups["a"]["members"]

    assert [record["w"] for record in records] == [2, 4]
    # ...and nothing was consumed anywhere (evaluation is ephemeral).
    assert len(carrier.couplings.couplings) == 1


def test_default_exit_is_identity():
    ds = pf.make_from_dataframe(
        pd.DataFrame({"tags": [("x", "y")]}, index=pd.Index(["r"], name="i")),
        pf.Schema(fields=(pf.IndexField(name="i"), pf.ValueField(name="tags"))),
    )

    assert ds["r"]["tags"] == ("x", "y")


def test_register_field_exit_takes_precedence():
    ds = pf.make_from_dataframe(
        pd.DataFrame({"tags": [("x", "y")]}, index=pd.Index(["r"], name="i")),
        pf.Schema(fields=(pf.IndexField(name="i"), pf.ValueField(name="tags"))),
    )

    pf.register_field_exit(pf.ValueField, lambda field_def, value: list(value))
    try:
        assert ds["r"]["tags"] == ["x", "y"]
    finally:
        _FIELD_EXITS.pop(pf.ValueField)
    assert ds["r"]["tags"] == ("x", "y")


# -- rows(): the positional, DataLoader-pluggable view ---------------------------


def test_rows_view_is_a_map_style_dataset():
    clips = _grouped()
    view = clips.rows()

    assert len(view) == 2
    assert view[0] == clips["a"]
    assert view[1] == clips["b"]
    assert view[-1] == clips["b"]
    assert [row["clip_id"] for row in view] == ["a", "b"]  # legacy iter protocol


def test_rows_view_field_selection():
    clips = _grouped()

    assert clips.rows("duration")[0] == 10.0
    assert clips.rows(["clip_id", "duration"])[1] == {
        "clip_id": "b",
        "duration": 20.0,
    }
    with pytest.raises(ValueError, match="not in the schema"):
        clips.rows("missing")


def test_rows_batched_fetch_bulk_evaluates_without_consuming_the_source():
    ds = pf.make_from_dataframe(
        pd.DataFrame(
            {"v": [1, 2, 3]},
            index=pd.Index(["x", "y", "z"], name="i"),
        ),
        pf.Schema(fields=(pf.IndexField(name="i"), pf.ValueField(name="v", dtype=int))),
    )
    handle = pf.map_fields(ds.fields(["v"]), _double, out="w")
    carrier = handle.dataset_context.dataset
    view = carrier.rows("w")

    # Order preserved; duplicates allowed (samplers may draw with replacement).
    assert view.__getitems__([2, 0, 2]) == [6, 2, 6]
    # The source dataset's pending work is untouched (the transient consumed).
    assert len(carrier.couplings.couplings) == 1
    assert carrier.table["w"].isna().all()


def test_dataset_int_positional_fallback_is_deprecated():
    clips = _clips()

    with pytest.warns(DeprecationWarning, match="positional fallback"):
        row = clips[0]
    assert row["clip_id"] == "a"
