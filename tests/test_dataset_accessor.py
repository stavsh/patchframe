"""Phase 1 of docs/design/dataset-accessor.md: the DatasetAccessor / DatasetSource
mirror, the no-materialize length/shape primitives, and the SourceManager
genericity audit (the manager resolves a DatasetAccessor with zero changes)."""

from __future__ import annotations

import pickle

import pandas as pd

from patchframe.data.dataset_accessor import DatasetAccessor
from patchframe.data.dimensions import IndexDimension
from patchframe.data.manager import SourceManager
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.sources.memory.dataset_source import ROW_DIMENSION, MemoryDatasetSource
from patchframe.testing import assert_dataset_source_contract

_ROW = IndexDimension(name=ROW_DIMENSION)


def _sample_state(n: int = 5) -> DatasetState:
    table = pd.DataFrame(
        {"x": list(range(n))},
        index=pd.Index([f"r{i}" for i in range(n)], name="id"),
    )
    schema = Schema(fields=(IndexField(name="id"), ValueField(name="x")))
    return DatasetState(schema=schema, table=table)


def test_materialize_full() -> None:
    src = MemoryDatasetSource(_sample_state())
    out = src.materialize(DatasetAccessor(source_desc_id=1))
    assert isinstance(out, Dataset)
    assert len(out) == 5


def test_create_slice_materialize() -> None:
    src = MemoryDatasetSource(_sample_state())
    acc = DatasetAccessor(source_desc_id=1).slice(_ROW.spec(1, 3))
    out = src.materialize(acc)
    assert list(out.table.index) == ["r1", "r2"]
    assert list(out.table["x"]) == [1, 2]


def test_length_and_shape_do_not_materialize() -> None:
    class _NoReadSource(MemoryDatasetSource):
        def read_full(self):  # type: ignore[override]
            raise AssertionError("length()/shape() must not materialize")

    src = _NoReadSource(_sample_state())
    assert src.length() == 5
    assert src.shape(DatasetAccessor(source_desc_id=1)) == (5,)
    assert src.shape(DatasetAccessor(source_desc_id=1).slice(_ROW.spec(1, 4))) == (3,)


def test_contract() -> None:
    src = MemoryDatasetSource(_sample_state())
    assert_dataset_source_contract(src, index_slice=_ROW.spec(0, 3))


def test_manager_genericity_roundtrip() -> None:
    # The audit: SourceManager resolves a DatasetAccessor with NO changes.
    mgr = SourceManager()
    desc_id = mgr.register_source(MemoryDatasetSource(_sample_state()))
    acc = DatasetAccessor(source_desc_id=desc_id, manager_hint=mgr)
    assert len(acc.materialize()) == 5          # via manager_hint
    assert len(acc.materialize(mgr)) == 5        # via explicit manager


def test_manager_reopen_via_descriptor() -> None:
    # register_descriptor (no live cache) forces open() through the source type.
    mgr = SourceManager()
    mgr.register_source_type(MemoryDatasetSource)
    desc_id = mgr.register_descriptor(MemoryDatasetSource(_sample_state()).describe())
    out = mgr.get_source_by_descriptor_id(desc_id).materialize(
        DatasetAccessor(source_desc_id=desc_id)
    )
    assert len(out) == 5


def test_accessor_pickles() -> None:
    acc = DatasetAccessor(source_desc_id=3).slice(_ROW.spec(0, 2))
    assert pickle.loads(pickle.dumps(acc)) == acc
