"""``offload`` (docs/design/dataset-accessor.md §5): persist a dataset into a
``DatasetStore`` and stream it back as a lazy chunk bundle. Grounds the
``resolve_fiber_cell`` resolver at the bundle read sites (``rows()`` exit,
``flatten``, ``extract``) over the in-memory backend."""

from __future__ import annotations

import pandas as pd

import patchframe as pf
from patchframe.data.manager import SourceManager
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.bundle import extract, flatten
from patchframe.sources.memory.dataset_source import MemoryDatasetSource, MemoryDatasetStore


def _ds(n: int = 6) -> Dataset:
    table = pd.DataFrame(
        {"x": list(range(n))},
        index=pd.Index([f"r{i}" for i in range(n)], name="id"),
    )
    schema = Schema(fields=(IndexField(name="id"), ValueField(name="x")))
    return Dataset(state=DatasetState(schema=schema, table=table))


def test_offload_chunks_and_streams() -> None:
    store = MemoryDatasetStore(manager=SourceManager())
    bundle = pf.offload(_ds(6), store=store, chunk_size=2)
    assert len(bundle.table) == 3  # ceil(6 / 2) chunks
    chunk1 = bundle.rows()[1]["fiber"]  # exits via resolve_fiber_cell -> records
    assert [row["x"] for row in chunk1] == [2, 3]


def test_offload_streams_only_touched_chunks() -> None:
    reads: list[slice] = []

    class _SpySource(MemoryDatasetSource):
        def read_partial(self, index_slice):
            reads.append(index_slice.dims["row"])
            return super().read_partial(index_slice)

    class _SpyStore(MemoryDatasetStore):
        def put(self, ds):
            return self.manager.register_source(_SpySource(ds.state))

    store = _SpyStore(manager=SourceManager())
    bundle = pf.offload(_ds(6), store=store, chunk_size=2)
    _ = bundle.rows()[1]["fiber"]  # touch only chunk 1
    assert reads == [slice(2, 4)]


def test_flatten_round_trips() -> None:
    store = MemoryDatasetStore(manager=SourceManager())
    ds = _ds(6)
    flat = flatten(pf.offload(ds, store=store, chunk_size=2))
    pd.testing.assert_frame_equal(flat.table[["x"]], ds.table[["x"]])
    assert list(flat.table.index) == list(ds.table.index)


def test_offload_whole_then_extract() -> None:
    store = MemoryDatasetStore(manager=SourceManager())
    ds = _ds(4)
    bundle = pf.offload(ds, store=store)  # chunk_size=None -> one whole fiber
    assert len(bundle.table) == 1
    whole = extract(bundle)
    pd.testing.assert_frame_equal(whole.table[["x"]], ds.table[["x"]])


def test_offload_requires_store_and_dataset() -> None:
    import pytest

    with pytest.raises(TypeError):
        pf.offload(_ds(3))  # no store
    with pytest.raises(TypeError):
        pf.offload(object(), store=MemoryDatasetStore(manager=SourceManager()))
