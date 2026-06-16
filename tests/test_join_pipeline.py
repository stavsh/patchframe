"""End-to-end: match → partition for matched sets per row.

The design culmination (join-dimensions-identity.md §6, partition-aggregate.md
§6): a dimensional join emits a correspondence plan, and partition reads it as
groups — matched segments per window. The correspondence's left_index is a
ForeignIndexField into the windows identity, so partition's identity arm with
`domain=windows` makes the grouping total (an empty fiber for a window that
matched nothing). This validates the whole machinery composes before the
fusion example is rewritten onto it.
"""

from __future__ import annotations

import pandas as pd

import patchframe as pf


def _span(start: float, stop: float) -> pf.DimensionedSlice:
    return pf.DimensionedSlice(dims={"time": slice(start, stop)})


def _windows() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "b"],
            "span": [_span(0, 2), _span(5, 7), _span(0, 2)],
        },
        index=pd.Index(["w0", "w1", "w2"], name="window_id"),
    )
    return pf.make_from_dataframe(
        table,
        pf.Schema(
            fields=(
                pf.IndexField(name="window_id"),
                pf.ValueField(name="clip", dtype=str),
                pf.DimensionedSliceField(name="span"),
            )
        ),
    )


def _segments() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "clip": ["a", "b"],
            "seg_span": [_span(0.5, 1.2), _span(0.1, 0.4)],
            "text": ["hello", "world"],
        },
        index=pd.Index(["s0", "s1"], name="segment_id"),
    )
    return pf.make_from_dataframe(
        table,
        pf.Schema(
            fields=(
                pf.IndexField(name="segment_id"),
                pf.ValueField(name="clip", dtype=str),
                pf.DimensionedSliceField(name="seg_span"),
                pf.ValueField(name="text", dtype=str),
            )
        ),
    )


def test_match_then_partition_gives_matched_sets_per_window():
    windows, segments = _windows(), _segments()

    correspondence = pf.match(
        windows, segments, on="clip", predicates={"time": pf.overlap()}
    )
    # w0(a,[0,2)) ∩ s0(a,[.5,1.2)); w2(b,[0,2)) ∩ s1(b,[.1,.4)); w1(a,[5,7)) none.
    assert set(zip(correspondence.table["left_index"], correspondence.table["right_index"])) == {
        ("w0", "s0"),
        ("w2", "s1"),
    }

    groups = pf.partition(
        correspondence, "left_index", domain=windows, into="matched"
    )

    # Total over the windows domain: every window present, in domain order.
    assert list(groups.table.index) == ["w0", "w1", "w2"]
    # The base inherits the windows identity (alignment-ready).
    assert pf.primary_index_identity(groups) == pf.primary_index_identity(windows)

    matched = {
        window: set(groups.table.loc[window, "matched"].table["right_index"])
        for window in groups.table.index
    }
    assert matched == {"w0": {"s0"}, "w1": set(), "w2": {"s1"}}
    # The unmatched window is an empty fiber (schema intact, zero rows).
    assert len(groups.table.loc["w1", "matched"].table) == 0


def test_matched_sets_attach_to_windows_by_alignment():
    # The grouping inherits the windows identity, so it attaches with no
    # collision strategy — the matched correspondence rides alongside windows.
    windows, segments = _windows(), _segments()
    correspondence = pf.match(
        windows, segments, on="clip", predicates={"time": pf.overlap()}
    )
    groups = pf.partition(correspondence, "left_index", domain=windows, into="matched")

    attached = pf.concat_columns(windows, pf.keep(groups, ["matched"]))

    assert list(attached.table.index) == ["w0", "w1", "w2"]
    assert set(attached.table.loc["w0", "matched"].table["right_index"]) == {"s0"}
    assert len(attached.table.loc["w1", "matched"].table) == 0
