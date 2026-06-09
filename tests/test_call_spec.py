"""CallSpec serializability and the early UnpicklableCallWarning.

A deferred operator call is recorded as an ``ApplyOperator`` carrying a
``CallSpec`` (operator + args/kwargs + variant). The coupling must stay
pickle-friendly (design-constraints §7) — it references its cells by *name*, so
it pickles independently of the (unpicklable-or-not) cell datasets. When the call
carries something unpicklable (a lambda predicate), the problem is surfaced at
*defer* time via ``UnpicklableCallWarning``, not later at ``collect``/save.
"""

from __future__ import annotations

import pickle
import warnings

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.couplings import ApplyOperator, CallSpec, CouplingSet


def _merge_inputs() -> tuple[pf.Dataset, pf.Dataset, pf.Dataset]:
    left = pf.make_from_dataframe(
        pd.DataFrame({"score": [1, 2]}, index=["a", "b"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="score", dtype=int))),
    )
    right = pf.make_from_dataframe(
        pd.DataFrame({"label": [3, 4]}, index=["a", "b"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="label", dtype=int))),
    )
    return left, right, pf.join(left, right, how="inner")


def _deferred_merge() -> ApplyOperator:
    left, right, plan = _merge_inputs()
    b = pf.bundle(left=left, right=right, plan=plan)
    merged = pf.merge(
        b.field("left"),
        b.field("right"),
        b.field("plan"),
        out="merged",
        collision=pf.ColumnCollisionStrategy(mode="keep", side="left"),
    )
    carrier = merged.dataset_context.dataset
    return next(c for c in carrier.couplings.couplings if isinstance(c, ApplyOperator))


def test_apply_operator_coupling_pickle_round_trips():
    apply = _deferred_merge()

    restored = pickle.loads(pickle.dumps(apply))

    assert isinstance(restored, ApplyOperator)
    assert restored.operator is pf.merge  # class re-imported by reference
    assert restored.params == {"collision": pf.ColumnCollisionStrategy(mode="keep", side="left")}
    assert restored.input_fields() == ("left", "right", "plan")
    assert restored.output_field() == "merged"


def test_coupling_set_with_apply_operator_pickle_round_trips():
    # The persisted unit: a CouplingSet carrying the deferred call. It references
    # its cells by name, so it pickles without the (separate) cell datasets.
    couplings = CouplingSet((_deferred_merge(),))

    restored = pickle.loads(pickle.dumps(couplings))

    assert isinstance(restored.couplings[0], ApplyOperator)
    assert restored.couplings[0].operator is pf.merge


def test_callspec_replay_equals_eager():
    left, right, plan = _merge_inputs()

    result = CallSpec(operator=pf.merge).replay(left, right, plan)

    pd.testing.assert_frame_equal(result.table, pf.merge(left, right, plan).table)


def test_operator_call_spec_drops_runtime_fields():
    left, right, plan = _merge_inputs()
    call = pf.merge.instance().normalize_call(left, right, plan)

    spec = call.spec()

    assert isinstance(spec, CallSpec)
    # The serializable core normalizes the operator to its class — the same
    # by-reference handle the bundle-defer path records. (call.operator is the
    # configured instance.)
    assert spec.operator is type(call.operator) is pf.merge
    # No datasets/states/contexts ride along: it pickles by reference.
    assert pickle.loads(pickle.dumps(spec)).operator is pf.merge


def _filter_ds() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"]),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name="v", dtype=int))),
    )


def test_unpicklable_lambda_warns_at_defer_time_and_still_replays():
    b = pf.bundle(_filter_ds())

    # The warning fires HERE, when the coupling is recorded — not later at collect.
    with pytest.warns(pf.UnpicklableCallWarning, match="where"):
        handle = pf.where(b.field("cell_0"), lambda df: df["v"] > 1, out="kept")

    # It still replays in-process.
    assert handle.collect().table["v"].tolist() == [2, 3]


def test_unpicklable_call_can_be_escalated_to_an_error():
    b = pf.bundle(_filter_ds())

    with warnings.catch_warnings():
        warnings.simplefilter("error", pf.UnpicklableCallWarning)
        with pytest.raises(pf.UnpicklableCallWarning):
            pf.where(b.field("cell_0"), lambda df: df["v"] > 1, out="kept")


def test_picklable_predicate_does_not_warn():
    b = pf.bundle(_filter_ds())

    with warnings.catch_warnings():
        warnings.simplefilter("error", pf.UnpicklableCallWarning)
        handle = pf.where(b.field("cell_0"), _keep_v_above_one, out="kept")  # module-level fn

    assert handle.collect().table["v"].tolist() == [2, 3]


def _keep_v_above_one(df: pd.DataFrame) -> pd.Series:
    return df["v"] > 1
