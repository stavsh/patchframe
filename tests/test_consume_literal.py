"""Literal consume: a coupling is pending work — completed and discharged.

consume/collect produce a state with the run couplings removed, so consuming
the chain twice is work-idempotent (the second call finds nothing producing
the column). Discharge travels with the *output* state — consume is a pure
function of its input snapshot. Row access is *evaluation*, not consumption:
pending work is computed ephemerally per row; nothing is persisted or
discharged (read paths cannot consume). See lazy-duality-plan.md, 2026-06-11.
"""

from __future__ import annotations

import pandas as pd

import patchframe as pf
from patchframe.dataset.context import FieldSelection

CALLS = {"n": 0}


def _counted_double(a):
    CALLS["n"] += 1
    return a * 2


def _double(a):
    return a * 2


def _base() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.Index(["x", "y", "z"], name="i"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="i"),
                pf.ValueField(name="a", dtype=int),
            )
        ),
    )


def test_consume_discharges_and_chained_consume_is_free():
    CALLS["n"] = 0
    handle = pf.map_fields(_base().fields(["a"]), _counted_double, out="c")
    carrier = handle.dataset_context.dataset

    consumed = pf.consume(carrier, "c")

    assert CALLS["n"] == 3
    assert consumed.table["c"].tolist() == [2, 4, 6]
    assert len(consumed.couplings.couplings) == 0

    again = pf.consume(consumed, "c")  # no producer left: idempotent branch

    assert CALLS["n"] == 3  # work-idempotent on the discharged state
    assert again.table["c"].tolist() == [2, 4, 6]


def test_discharge_travels_with_the_output_state():
    # consume is a pure function: the *input* snapshot keeps its pending work,
    # so consuming the same snapshot twice recomputes twice.
    CALLS["n"] = 0
    handle = pf.map_fields(_base().fields(["a"]), _counted_double, out="c")
    carrier = handle.dataset_context.dataset

    pf.consume(carrier, "c")
    pf.consume(carrier, "c")

    assert CALLS["n"] == 6
    assert len(carrier.couplings.couplings) == 1


def test_collect_advances_the_cursor_to_the_consumed_snapshot():
    CALLS["n"] = 0
    handle = pf.map_fields(_base().fields(["a"]), _counted_double, out="c")

    first = handle.collect()

    assert CALLS["n"] == 3
    assert handle.dataset_context.dataset is first
    assert len(first.couplings.couplings) == 0

    second = handle.collect()  # already consumed: nothing re-runs

    assert CALLS["n"] == 3
    assert second.table["c"].tolist() == [2, 4, 6]


def test_row_access_evaluates_ephemerally():
    CALLS["n"] = 0
    handle = pf.map_fields(_base().fields(["a"]), _counted_double, out="c")

    assert handle.loc["y"] == 4
    assert handle.loc["y"] == 4

    # Evaluation, not consumption: each access recomputes; nothing persisted,
    # nothing discharged.
    assert CALLS["n"] == 2
    carrier = handle.dataset_context.dataset
    assert carrier.table["c"].isna().all()
    assert len(carrier.couplings.couplings) == 1


def test_annotation_after_consume_reads_from_storage():
    # The annotation flow in one gesture: consume completes-and-detaches, so
    # assignment afterwards is safe by construction and row access returns it.
    CALLS["n"] = 0
    handle = pf.map_fields(_base().fields(["a"]), _counted_double, out="c")

    handle.collect()
    handle.loc["y"] = 123

    assert handle.loc["y"] == 123
    assert CALLS["n"] == 3  # no recompute: the coupling was discharged


def test_partial_consume_discharges_only_its_chain():
    h_c = pf.map_fields(_base().fields(["a"]), _double, out="c")
    h_d = pf.map_fields(h_c, _double, out="d")
    carrier = h_d.dataset_context.dataset
    c_coupling = next(
        c for c in carrier.couplings.couplings if c.output_field() == "c"
    )

    partial = pf.consume(carrier, c_coupling)

    assert partial.table["c"].tolist() == [2, 4, 6]
    assert partial.table["d"].isna().all()
    assert [c.output_field() for c in partial.couplings.couplings] == ["d"]


def test_field_selection_collect_returns_the_filled_snapshot():
    # Regression: each handle.collect() advances the shared cursor, so the
    # selection terminal returns the consumed snapshot (and chain couplings
    # shared between fields run once).
    h_c = pf.map_fields(_base().fields(["a"]), _double, out="c")
    h_d = pf.map_fields(h_c, _double, out="d")

    out = FieldSelection((h_c, h_d)).collect()

    assert out.table["c"].tolist() == [2, 4, 6]
    assert out.table["d"].tolist() == [4, 8, 12]
    assert len(out.couplings.couplings) == 0
