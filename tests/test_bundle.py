"""Tests for the eager wide-record Bundle: the deferred form of a blocking op.

Covers the first proof of the lazy<->eager duality (lazy-and-bundle.md §3, §6):
a blocking operator deferred as a one-row BundleField record + an ApplyOperator
coupling, with the ``FieldHandle.collect()`` terminal (internally ``_collect``)
running it and extracting the result.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

import patchframe as pf
from patchframe.dataset.couplings import ApplyOperator
from patchframe.ops.bundle import BUNDLE_INDEX_NAME, _collect, build_apply_bundle


def _dataset(table: pd.DataFrame, *fields: pf.Field) -> pf.Dataset:
    return pf.make_from_dataframe(
        table,
        pf.Schema(fields=(pf.IndexField(name="item_id"), *fields)),
    )


def _merge_inputs() -> tuple[pf.Dataset, pf.Dataset, pf.Dataset]:
    left = _dataset(
        pd.DataFrame({"score": [10, 20]}, index=["a", "b"]),
        pf.ValueField(name="score", dtype=int),
    )
    right = _dataset(
        pd.DataFrame({"label": [200, 300]}, index=["b", "c"]),
        pf.ValueField(name="label", dtype=int),
    )
    plan = pf.join(left, right, how="outer")
    return left, right, plan


def _merge_bundle() -> tuple[pf.Dataset, pf.Dataset, pf.Dataset, pf.Dataset]:
    left, right, plan = _merge_inputs()
    bundle = build_apply_bundle(
        pf.merge,
        inputs={"left": left, "right": right, "join_plan": plan},
    )
    return bundle, left, right, plan


def test_build_apply_bundle_is_a_one_row_wide_record():
    bundle, left, right, plan = _merge_bundle()

    assert len(bundle) == 1
    assert bundle.schema.names() == (
        BUNDLE_INDEX_NAME,
        "left",
        "right",
        "join_plan",
        "result",
    )
    assert isinstance(bundle.schema.get("left"), pf.BundleField)
    assert isinstance(bundle.schema.get("result"), pf.BundleField)

    # Cells hold the actual input datasets (a reference to immutable state).
    assert bundle.table.at[0, "left"] is left
    assert bundle.table.at[0, "right"] is right
    assert bundle.table.at[0, "join_plan"] is plan
    # The output cell is declared but unmaterialized.
    assert pd.isna(bundle.table.at[0, "result"])


def test_bundle_records_a_single_apply_operator_coupling():
    bundle, _, _, _ = _merge_bundle()

    couplings = bundle.couplings.couplings
    assert len(couplings) == 1
    coupling = couplings[0]
    assert isinstance(coupling, ApplyOperator)
    assert coupling.input_fields() == ("left", "right", "join_plan")
    assert coupling.output_field() == "result"
    assert coupling.operator is pf.merge


def test_collect_equals_eager_merge():
    bundle, left, right, plan = _merge_bundle()

    collected = bundle.field("result").collect()
    eager = pf.merge(left, right, plan)

    assert isinstance(collected, pf.Dataset)
    assert collected.schema.names() == eager.schema.names()
    pd.testing.assert_frame_equal(collected.table, eager.table)


def test_collect_result_inherits_real_op_identity_not_bundle_scaffold():
    bundle, left, right, plan = _merge_bundle()

    collected = bundle.field("result").collect()
    eager = pf.merge(left, right, plan)

    # merge inherits the join-plan's row identity (index=2); the bundle's own
    # throwaway base identity must not leak into the result.
    assert pf.primary_index_identity(collected) == pf.primary_index_identity(eager)
    assert pf.primary_index_identity(collected) == pf.primary_index_identity(plan)


def test_consume_fills_output_cell_and_preserves_input_cells():
    bundle, left, right, plan = _merge_bundle()

    filled = pf.consume(bundle, "result")

    assert isinstance(filled.table.at[0, "result"], pf.Dataset)
    # Inputs are pinned, not consumed or copied.
    assert filled.table.at[0, "left"] is left
    assert filled.table.at[0, "right"] is right
    assert filled.table.at[0, "join_plan"] is plan


def test_collect_terminal_paths_agree():
    bundle, _, _, _ = _merge_bundle()
    # The handle terminal targets the field it points at.
    via_handle = bundle.field("result").collect()
    # The internal terminal infers the single ApplyOperator output.
    via_internal = _collect(bundle)

    assert isinstance(via_handle, pf.Dataset)
    pd.testing.assert_frame_equal(via_handle.table, via_internal.table)


def test_bundlefield_validate_accepts_dataset_cells_and_nulls():
    left, _, _ = _merge_inputs()
    field = pf.BundleField(name="cell")

    field.validate_column(pd.Series([left], dtype=object))
    field.validate_column(pd.Series([None], dtype=object))


def test_apply_operator_coupling_is_pickle_friendly():
    bundle, _, _, _ = _merge_bundle()
    coupling = bundle.couplings.couplings[0]

    restored = pickle.loads(pickle.dumps(coupling))

    assert restored.input_fields() == ("left", "right", "join_plan")
    assert restored.output_field() == "result"
    assert restored.operator is pf.merge


def test_build_apply_bundle_rejects_empty_inputs_and_output_collision():
    left, right, _ = _merge_inputs()
    with pytest.raises(ValueError, match="at least one input"):
        build_apply_bundle(pf.merge, inputs={})
    with pytest.raises(ValueError, match="collides with an input"):
        build_apply_bundle(pf.merge, inputs={"left": left}, output="left")
