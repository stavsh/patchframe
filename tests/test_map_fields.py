"""map_fields: the per-row map operator + FieldHandle.loc getter.

map_fields is a *computation* (not a binding), so it follows the operand-
dispatch law: the eager ``map_fields(ds, [...], fn, out=)`` records a
MapCoupling and consumes it immediately — the column is filled now and the
coupling stays as the recorded recipe. The handle form records same-level
without running and returns a chaining FieldHandle; consume / row access /
collect runs it later. The function must be picklable (module-level),
mirroring the bundle arm.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import patchframe as pf


def _sum(a, b):
    return a + b


def _pair(a, b):
    return (a, b)


def _base() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["x", "y", "z"]),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="a", dtype=int),
                pf.ValueField(name="b", dtype=int),
            )
        ),
    )


def test_eager_arm_computes_now_and_discharges_the_coupling():
    ds = _base()

    out = pf.map_fields(ds, ["a", "b"], _sum, out="c")

    assert isinstance(out, pf.Dataset)
    assert "c" in out.schema.names()
    # The law: a Dataset operand means the work happens now.
    assert out.table["c"].tolist() == [11, 22, 33]
    # Consume is literal: the coupling is discharged; the column is the product.
    assert len(out.couplings.couplings) == 0
    # A second consume is the idempotent no-producer branch.
    assert pf.consume(out, "c").table["c"].tolist() == [11, 22, 33]


def test_lazy_arm_does_not_compute_until_collect():
    ds = _base()

    handle = pf.map_fields(ds.fields(["a", "b"]), _sum, out="c")

    # Deferral is opt-in via the handle arm: the carrier column is still null.
    carrier = handle.dataset_context.dataset
    assert carrier.table["c"].isna().all()
    assert handle.collect().table["c"].tolist() == [11, 22, 33]


def test_lazy_arm_returns_handle_and_collects():
    ds = _base()

    handle = pf.map_fields(ds.fields(["a", "b"]), _sum, out="c")

    assert isinstance(handle, pf.FieldHandle)
    assert handle.name == "c"
    assert handle.collect().table["c"].tolist() == [11, 22, 33]


def test_lazy_handle_items_yield_per_row():
    ds = _base()

    handle = pf.map_fields(ds.fields(["a", "b"]), _pair, out="c")

    assert [value for _, value in handle.items()] == [(1, 10), (2, 20), (3, 30)]


def test_field_handle_loc_runs_coupling_for_one_row():
    ds = _base()

    handle = pf.map_fields(ds.fields(["a", "b"]), _pair, out="c")

    assert handle.loc["y"] == (2, 20)


def test_lambda_fn_warns_at_record_time():
    ds = _base()

    # The warning fires when the coupling is recorded (here), not later at collect.
    with pytest.warns(pf.UnpicklableCallWarning, match="cannot be pickled"):
        pf.map_fields(ds, ["a", "b"], lambda a, b: a + b, out="c")


def test_lambda_fn_can_be_escalated_to_an_error():
    ds = _base()

    with warnings.catch_warnings():
        warnings.simplefilter("error", pf.UnpicklableCallWarning)
        with pytest.raises(pf.UnpicklableCallWarning):
            pf.map_fields(ds, ["a", "b"], lambda a, b: a + b, out="c")


def test_module_level_fn_does_not_warn():
    ds = _base()

    with warnings.catch_warnings():
        warnings.simplefilter("error", pf.UnpicklableCallWarning)
        out = pf.map_fields(ds, ["a", "b"], _sum, out="c")

    assert "c" in out.schema.names()


def test_output_name_collision_errors():
    ds = _base()

    with pytest.raises(ValueError, match="already exists"):
        pf.map_fields(ds, ["a", "b"], _sum, out="a")


def test_unknown_input_field_errors():
    ds = _base()

    with pytest.raises(ValueError, match="not in schema"):
        pf.map_fields(ds, ["a", "missing"], _sum, out="c")
