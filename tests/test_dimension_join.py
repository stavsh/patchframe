"""dimension_join: one single-dimension predicate → a correspondence plan.

The atom of the dimensional join (dimension-join-execution.md). Eager produces
a left_index/right_index ForeignIndex correspondence; `candidates=` narrows a
prior correspondence (the compositional chain). Not coupling-able (builds a new
plan schema), so its lazy arm lifts onto a BundleField carrier — free from the
signature, like merge/join.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _windows() -> pf.Dataset:
    x = pf.IndexDimension(name="time")
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "b"],
            "span": [x.spec(0, 2), x.spec(1, 3), x.spec(0, 2)],
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
    return pf.make_from_dataframe(
        pd.DataFrame(
            {
                "clip": ["a", "a", "b"],
                "start": [0.5, 2.5, 0.1],
                "end": [1.2, 2.8, 0.4],
            },
            index=pd.Index(["s0", "s1", "s2"], name="segment_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="segment_id"),
                pf.ValueField(name="clip", dtype=str),
                pf.ValueField(name="start", dtype=float),
                pf.ValueField(name="end", dtype=float),
            )
        ),
    )


def _pairs(plan: pf.Dataset) -> set:
    return {
        (left, right)
        for left, right in zip(plan.table["left_index"], plan.table["right_index"])
    }


# -- eager: a single predicate -------------------------------------------------


def test_equals_produces_same_clip_correspondence():
    windows, segments = _windows(), _segments()

    plan = pf.dimension_join(
        windows, segments, predicate=pf.equals(), left_on="clip", right_on="clip"
    )

    assert _pairs(plan) == {("w0", "s0"), ("w0", "s1"), ("w1", "s0"), ("w1", "s1"), ("w2", "s2")}
    assert isinstance(plan.schema.get("left_index"), pf.ForeignIndexField)
    assert plan.schema.get("left_index").target_identity == pf.primary_index_identity(windows)
    assert plan.schema.get("right_index").target_identity == pf.primary_index_identity(segments)


def test_overlap_slice_vs_start_end_pair():
    windows, segments = _windows(), _segments()

    # left is a composed slice (dimension "time"); right is a (start, end) pair.
    plan = pf.dimension_join(
        windows,
        segments,
        predicate=pf.overlap(),
        left_on="span",
        right_on=("start", "end"),
        dimension="time",
    )

    # Overlap alone does NOT scope by clip — every interval pair that overlaps:
    #   w0[0,2): s0[.5,1.2), s2[.1,.4);  w1[1,3): s0, s1[2.5,2.8);  w2[0,2): s0, s2.
    assert _pairs(plan) == {
        ("w0", "s0"), ("w0", "s2"),
        ("w1", "s0"), ("w1", "s1"),
        ("w2", "s0"), ("w2", "s2"),
    }


# -- eager: the chain (candidates=) --------------------------------------------


def test_chain_equals_then_overlap():
    windows, segments = _windows(), _segments()

    blocks = pf.dimension_join(
        windows, segments, predicate=pf.equals(), left_on="clip", right_on="clip"
    )
    matches = pf.dimension_join(
        windows,
        segments,
        blocks,
        predicate=pf.overlap(),
        left_on="span",
        right_on=("start", "end"),
        dimension="time",
    )

    # clip-scoped overlap: same-clip pairs that also overlap in time. w1[1,3)
    # genuinely overlaps s1[2.5,2.8) (both clip a), so it survives.
    assert _pairs(matches) == {("w0", "s0"), ("w1", "s0"), ("w1", "s1"), ("w2", "s2")}


# -- lazy arm: free from the signature (bundle carrier) ------------------------


def test_lazy_arm_via_bundle_handles_matches_eager():
    windows, segments = _windows(), _segments()
    eager = pf.dimension_join(
        windows, segments, predicate=pf.equals(), left_on="clip", right_on="clip"
    )

    b = pf.bundle(left=windows, right=segments)
    handle = pf.dimension_join(
        b.field("left"),
        b.field("right"),
        predicate=pf.equals(),
        left_on="clip",
        right_on="clip",
        out="corr",
    )

    assert isinstance(handle, pf.FieldHandle)
    # Deferred: nothing computed until collect.
    carrier = handle.dataset_context.dataset
    assert carrier.table["corr"].isna().all()
    assert _pairs(handle.collect()) == _pairs(eager)


def test_regular_field_handle_is_rejected():
    # Not coupling-able → its lazy form needs bundle cells; a plain field handle
    # (eager intent) is rejected, not silently resolved (the operand-dispatch law).
    windows, segments = _windows(), _segments()
    with pytest.raises(TypeError, match="bundle FieldHandles"):
        pf.dimension_join(
            windows.field("clip"),
            segments,
            predicate=pf.equals(),
            left_on="clip",
            right_on="clip",
            out="corr",
        )


def test_missing_field_errors():
    windows, segments = _windows(), _segments()
    with pytest.raises(ValueError, match="not present"):
        pf.dimension_join(
            windows, segments, predicate=pf.equals(), left_on="missing", right_on="clip"
        )


def test_candidates_identity_mismatch_is_rejected():
    # The open-seam safety check: a correspondence whose left_index references a
    # *different* windows dataset (matching labels, fresh identity) is rejected.
    windows, segments = _windows(), _segments()
    other_windows = _windows()  # same labels, independently minted identity
    blocks = pf.dimension_join(
        other_windows, segments, predicate=pf.equals(), left_on="clip", right_on="clip"
    )

    with pytest.raises(ValueError, match="different"):
        pf.dimension_join(
            windows,
            segments,
            blocks,
            predicate=pf.overlap(),
            left_on="span",
            right_on=("start", "end"),
            dimension="time",
        )
