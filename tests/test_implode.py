"""implode: the plan-driven collapse — explode's cardinality dual.

implode(members, plan) gathers the member rows a membership plan references
and collapses them into per-group fibers (N→1), where explode expands (1→N).
A member mapped to several groups replicates into each (the reason it is more
than a column partition); `domain` makes the grouping total (empty fiber for a
group that matched nothing). A match/join correspondence is the common plan.
Not coupling-able (builds a tall bundle), so its lazy arm lifts onto a
BundleField carrier — free from the signature, like merge/join/dimension_join.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _span(a: float, b: float) -> pf.DimensionedSlice:
    return pf.DimensionedSlice(dims={"time": slice(a, b)})


def _windows() -> pf.Dataset:
    table = pd.DataFrame(
        {
            "clip": ["a", "a", "b", "a"],
            "span": [_span(0, 2), _span(1, 3), _span(0, 2), _span(10, 12)],
        },
        index=pd.Index(["w0", "w1", "w2", "w3"], name="window_id"),
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
            "seg_span": [_span(0.5, 2.5), _span(0.1, 0.4)],  # s0 spans w0/w1
            "text": ["spanning", "world"],
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


def _correspondence() -> tuple[pf.Dataset, pf.Dataset, pf.Dataset]:
    windows, segments = _windows(), _segments()
    plan = pf.match(windows, segments, on="clip", predicates={"time": pf.overlap()})
    return windows, segments, plan


def _texts(groups: pf.Dataset, into: str) -> dict:
    return {
        g: list(groups.table.loc[g, into].table["text"]) for g in groups.table.index
    }


def test_implode_replicates_and_totals_over_domain():
    windows, segments, plan = _correspondence()

    groups = pf.implode(segments, plan, windows, into="seg")

    assert list(groups.table.index) == ["w0", "w1", "w2", "w3"]
    assert pf.primary_index_identity(groups) == pf.primary_index_identity(windows)
    assert _texts(groups, "seg") == {
        "w0": ["spanning"],
        "w1": ["spanning"],  # the member mapped to two groups replicated
        "w2": ["world"],
        "w3": [],  # collapsed nothing → empty fiber
    }
    # The fiber holds real member rows + the member's own label (right_index).
    w0 = groups.table.loc["w0", "seg"]
    assert "text" in w0.schema.names() and "seg_span" in w0.schema.names()
    assert list(w0.table["right_index"]) == ["s0"]


def test_implode_without_domain_covers_observed_groups():
    _, segments, plan = _correspondence()

    groups = pf.implode(segments, plan, into="seg")

    assert set(groups.table.index) == {"w0", "w1", "w2"}  # w3 absent (no domain)


def test_implode_attaches_to_domain_by_alignment():
    windows, segments, plan = _correspondence()
    groups = pf.implode(segments, plan, windows, into="seg")

    attached = pf.concat_columns(windows, pf.keep(groups, ["seg"]))

    assert list(attached.table.index) == ["w0", "w1", "w2", "w3"]
    assert list(attached.table.loc["w1", "seg"].table["text"]) == ["spanning"]
    assert len(attached.table.loc["w3", "seg"].table) == 0


def test_implode_lazy_arm_via_bundle_handles_matches_eager():
    windows, segments, plan = _correspondence()
    eager = pf.implode(segments, plan, windows, into="seg")

    b = pf.bundle(members=segments, plan=plan, domain=windows)
    handle = pf.implode(
        b.field("members"), b.field("plan"), b.field("domain"), into="seg", out="grouped"
    )

    assert isinstance(handle, pf.FieldHandle)
    carrier = handle.dataset_context.dataset
    assert carrier.table["grouped"].isna().all()  # deferred until collect
    collected = handle.collect()
    assert _texts(collected, "seg") == _texts(eager, "seg")


def test_implode_regular_field_handle_is_rejected():
    # Not coupling-able → its lazy form needs bundle cells; a plain field handle
    # is rejected, not silently resolved (the operand-dispatch law).
    windows, segments, plan = _correspondence()
    with pytest.raises(TypeError, match="bundle FieldHandles"):
        pf.implode(
            segments.field("text"), plan, windows, into="seg", out="grouped"
        )
