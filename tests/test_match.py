"""match: declarative dimensional correspondence (dimension-join-execution.md).

match sequences a dimension_join chain: `on` keys are categorical equals
partitioners, `predicates[dimension]` are interval/spatial terms, run in stage
order (partitioners first) and threaded via candidates. Sugar over the
compositional chain; coexists with the strategy-based join.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _span(start: float, stop: float) -> pf.DimensionedSlice:
    return pf.DimensionedSlice(dims={"time": slice(start, stop)})


def _windows() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "b"],
            "span": [_span(0, 2), _span(1, 3), _span(0, 2)],
        },
        index=pd.Index(["w0", "w1", "w2"], name="window_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="window_id"),
            pf.ValueField(name="clip", dtype=str),
            pf.DimensionedSliceField(name="span"),
        )
    )
    return pf.make_from_dataframe(table, schema)


def _segments() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "b"],
            "seg_span": [_span(0.5, 1.2), _span(2.5, 2.8), _span(0.1, 0.4)],
        },
        index=pd.Index(["s0", "s1", "s2"], name="segment_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="segment_id"),
            pf.ValueField(name="clip", dtype=str),
            pf.DimensionedSliceField(name="seg_span"),
        )
    )
    return pf.make_from_dataframe(table, schema)


def _pairs(plan: pf.Dataset) -> set:
    return set(zip(plan.table["left_index"], plan.table["right_index"]))


def test_match_on_plus_overlap_predicate():
    windows, segments = _windows(), _segments()

    plan = pf.match(
        windows, segments, on="clip", predicates={"time": pf.overlap()}
    )

    # clip-scoped overlap: same clip AND overlapping in time.
    assert _pairs(plan) == {("w0", "s0"), ("w1", "s0"), ("w1", "s1"), ("w2", "s2")}
    assert isinstance(plan.schema.get("left_index"), pf.ForeignIndexField)
    assert plan.schema.get("left_index").target_identity == pf.primary_index_identity(windows)
    assert plan.schema.get("right_index").target_identity == pf.primary_index_identity(segments)


def test_match_equals_only():
    windows, segments = _windows(), _segments()

    plan = pf.match(windows, segments, on="clip")

    assert _pairs(plan) == {
        ("w0", "s0"), ("w0", "s1"), ("w1", "s0"), ("w1", "s1"), ("w2", "s2")
    }


def test_match_predicate_only_is_unscoped():
    windows, segments = _windows(), _segments()

    plan = pf.match(windows, segments, predicates={"time": pf.overlap()})

    # No `on` → overlap across clips too.
    assert ("w2", "s0") in _pairs(plan)  # clip b window overlaps clip a segment


def test_match_pad_reaches_across_the_gap():
    windows, segments = _windows(), _segments()

    plain = pf.match(windows, segments, on="clip", predicates={"time": pf.overlap()})
    padded = pf.match(windows, segments, on="clip", predicates={"time": pf.overlap(pad=0.3)})

    # w0[0,2) and s1[2.5,2.8) (both clip a) are 0.5s apart — pad 0.3 still misses,
    # but w1[1,3) already overlaps s1, so check pad only adds, never removes.
    assert _pairs(plain) <= _pairs(padded)


def test_match_requires_a_term():
    windows, segments = _windows(), _segments()
    with pytest.raises(ValueError, match="at least one"):
        pf.match(windows, segments)


def test_match_ambiguous_dimension_errors():
    # Two slice fields both carrying "time" → cannot resolve uniquely.
    windows = _windows()
    table = windows.table.copy()
    table["span2"] = windows.table["span"]
    schema = pf.Schema(fields=(*windows.schema.fields, pf.DimensionedSliceField(name="span2")))
    ambiguous = pf.make_from_dataframe(table, schema)

    with pytest.raises(ValueError, match="unique slice field"):
        pf.match(ambiguous, _segments(), predicates={"time": pf.overlap()})
