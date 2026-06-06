"""Tests for the bundle / extract / flatten operators and consume idempotency.

These are the flat<->bundle morphisms as first-class operators (Phase 3). The
notable one is ``extract``: the first real ``FieldHandle``->``BundleField``
operand, which gets eager/lazy dispatch and context propagation for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

import patchframe as pf
from patchframe.ops.bundle import BUNDLE_INDEX_NAME


def _dataset(values: list[int], index: list[str], name: str = "v") -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({name: values}, index=index),
        pf.Schema(fields=(pf.IndexField(name="item_id"), pf.ValueField(name=name, dtype=int))),
    )


@dataclass(frozen=True, slots=True)
class _Doubler(pf.Coupling):
    """A trivial coupling: ``target = source * 2`` (non-BundleField output)."""

    source: pf.FieldRef
    target: pf.FieldRef

    def input_fields(self) -> tuple[str, ...]:
        return (self.source.name,)

    def output_field(self) -> str:
        return self.target.name

    def compute(self, state) -> pd.Series:
        return state.table[self.source.name] * 2


def test_bundle_named_cells_build_a_wide_record_carrier():
    left = _dataset([1], ["a"])
    right = _dataset([2], ["b"])

    b = pf.bundle(left=left, right=right)

    assert isinstance(b, pf.Dataset)
    assert len(b) == 1
    assert b.schema.names() == (BUNDLE_INDEX_NAME, "left", "right")
    assert isinstance(b.schema.get("left"), pf.BundleField)
    assert b.table.at[0, "left"] is left
    assert b.table.at[0, "right"] is right
    assert b.couplings.couplings == ()


def test_bundle_positional_cells_are_auto_named():
    left = _dataset([1], ["a"])
    right = _dataset([2], ["b"])

    b = pf.bundle(left, right)

    assert b.schema.names() == (BUNDLE_INDEX_NAME, "cell_0", "cell_1")
    assert b.table.at[0, "cell_0"] is left


def test_bundle_rejects_non_datasets_and_empty():
    left = _dataset([1], ["a"])
    with pytest.raises(TypeError, match="must be a Dataset"):
        pf.bundle(left=left, right=object())
    with pytest.raises(ValueError, match="at least one dataset"):
        pf.bundle()


def test_bundle_is_an_eager_sibling_and_does_not_advance_a_cursor():
    left = _dataset([1], ["a"])
    right = _dataset([2], ["b"])
    ctx = left.context()

    with ctx:
        pf.bundle(left=left, right=right)

    assert ctx.dataset is left  # construction is a sibling, not a cursor advance


def test_extract_pulls_a_cell_out_eagerly():
    left = _dataset([1], ["a"])
    right = _dataset([2], ["b"])
    b = pf.bundle(left=left, right=right)

    assert pf.extract(b, "left") is left
    assert pf.extract(b, "right") is right


def test_extract_accepts_a_bundlefield_handle_and_propagates_context():
    left = _dataset([1], ["a"])
    right = _dataset([2], ["b"])
    b = pf.bundle(left=left, right=right)
    context = b.field("left").dataset_context

    result = pf.extract(b.field("left"))

    assert result is left
    # handle operand -> lazy -> the context propagated to the extracted cell
    assert context.dataset is left


def test_extract_infers_single_bundle_field():
    only = _dataset([1], ["a"])
    b = pf.bundle(cell=only)

    assert pf.extract(b) is only


def test_extract_rejects_a_non_bundle_field():
    left = _dataset([1], ["a"])
    b = pf.bundle(left=left)

    with pytest.raises(TypeError, match="not a BundleField"):
        pf.extract(b, BUNDLE_INDEX_NAME)


def test_flatten_is_concat_rows_of_cells():
    a = _dataset([1], ["a"])
    c = _dataset([2], ["c"])

    flat = pf.flatten(pf.bundle(a, c))
    eager = pf.concat_rows(a, c)

    pd.testing.assert_frame_equal(flat.table, eager.table)


def test_flatten_single_cell_returns_it():
    only = _dataset([1], ["a"])

    assert pf.flatten(pf.bundle(only)) is only


def test_consume_is_idempotent_on_a_materialized_field():
    ds = _dataset([1, 2], ["a", "b"])

    # "v" exists and no coupling produces it: consume returns it unchanged.
    result = pf.consume(ds, "v")

    pd.testing.assert_frame_equal(result.table, ds.table)


def test_collect_on_a_non_bundle_field_returns_the_container():
    # Extraction is BundleField-specific; collecting a regular field just
    # materializes it and returns the container dataset, not a cell.
    ds = _dataset([1, 2], ["a", "b"])

    result = ds.field("v").collect()

    assert isinstance(result, pf.Dataset)
    assert result.schema.names() == ds.schema.names()
    pd.testing.assert_frame_equal(result.table, ds.table)


def test_collect_on_a_bundle_field_still_extracts_the_cell():
    only = _dataset([1], ["a"])
    b = pf.bundle(cell=only)

    # The BundleField path still extracts the inner dataset.
    assert b.field("cell").collect() is only


def test_collect_runs_a_pending_coupling_on_a_non_bundle_field():
    ds = pf.make_from_dataframe(
        pd.DataFrame(
            {
                "v": pd.array([1, 2], dtype="Int64"),
                "doubled": pd.array([pd.NA, pd.NA], dtype="Int64"),
            },
            index=["a", "b"],
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="v", dtype=int),
                pf.ValueField(name="doubled", dtype=int),
            )
        ),
        couplings=pf.CouplingSet((_Doubler(pf.FieldRef("v"), pf.FieldRef("doubled")),)),
    )

    result = ds.field("doubled").collect()

    assert isinstance(result, pf.Dataset)
    # Non-BundleField: the pending coupling runs and the container is returned
    # with the field materialized — not extracted to a cell.
    assert result.schema.names() == ds.schema.names()
    assert result.table["doubled"].tolist() == [2, 4]
    assert result.table["v"].tolist() == [1, 2]
