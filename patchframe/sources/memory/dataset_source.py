"""
patchframe.sources.memory.dataset_source

In-memory ``DatasetSource``: holds one ``Dataset``'s state and serves it — whole,
or an index-range row-subset — to ``DatasetAccessor`` materialization. The
dataset-valued sibling of ``MemoryDataSource``.

In-process: ``reopenable`` (``open(describe())`` works in the same process,
carrying the live ``DatasetState`` in the descriptor) but **not** ``portable``
(the held state may reference live runtime objects). This is the pandas-backed
memory store of docs/design/dataset-accessor.md §5 — object-cell payloads are
shared by reference under iloc, so an index-slice gather copies only the pointer/
scalar columns, never the payloads. ``length()`` reads the row count off the
table without materializing payloads.
"""

from __future__ import annotations

import uuid

from patchframe.data.dataset_source import DatasetSource
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions, IndexDimension
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.state import DatasetState

#: Conventional name of the row axis (axis 0) of a dataset source (§6).
ROW_DIMENSION = "row"


class MemoryDatasetSource(DatasetSource):
    """Serve one in-memory ``Dataset``, sliced by row-range, to ``DatasetAccessor``s."""

    source_type = "memory_dataset"
    runtime = True
    reopenable = True
    portable = False
    supports_partial_read = True
    thread_safe = True

    def __init__(self, state: DatasetState, *, source_id: str | None = None) -> None:
        self._state = state
        self._source_id = source_id or str(uuid.uuid4())
        self.dimensions = Dimensions((IndexDimension(name=ROW_DIMENSION),))

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "MemoryDatasetSource":
        return cls(state=descriptor.open_config["state"], source_id=descriptor.source_id)

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_type=self.source_type,
            source_id=self._source_id,
            open_config={"state": self._state},
            capabilities={"dimensions": self.dimensions},
        )

    def length(self) -> int:
        return len(self._state.table)

    def read_full(self) -> Dataset:
        return Dataset(state=self._state)

    def read_partial(self, index_slice: DimensionedSlice) -> Dataset:
        resolved = self.dimensions.resolve(index_slice)
        row_value = resolved[0].value if resolved else slice(None)
        return Dataset(state=self._state).replace_state(
            table=self._state.table.iloc[row_value]
        )
