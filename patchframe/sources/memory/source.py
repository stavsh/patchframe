"""
patchframe.sources.memory.source

In-memory array store and data source for patchframe.

MemoryArrayStore  — persistence layer: holds arrays with their extents,
                    knows dimensions, produces a SourceDescriptor.
MemoryDataSource  — runtime layer: opened from a SourceDescriptor,
                    materializes DataAccessors, applies lazy DimensionedSlices.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions
from patchframe.data.source import DataSource
from patchframe.storage.array_store import ArrayStore


@dataclass(frozen=True, slots=True)
class _MemoryArrayEntry:
    """Internal storage record for MemoryArrayStore."""

    array: np.ndarray
    extent: DimensionedSlice


@dataclass(slots=True)
class MemoryArrayStore(ArrayStore):
    """In-memory array store backed by a nested dict of _MemoryArrayEntry objects."""

    dimensions: Dimensions
    _entries: dict[Any, dict[str, _MemoryArrayEntry]] = field(default_factory=dict)
    _asset_names: dict[int, str] = field(default_factory=lambda: {0: "data"})
    _source_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def write(self, item_id: Any, asset_name: str, array: np.ndarray, extent: DimensionedSlice) -> None:
        self._entries.setdefault(item_id, {})[asset_name] = _MemoryArrayEntry(array=array, extent=extent)

    def append(self, item_id: Any, asset_name: str, array: np.ndarray, extent: DimensionedSlice) -> None:
        if asset_name in self._entries.get(item_id, {}):
            raise ValueError(f"Entry ({item_id!r}, {asset_name!r}) already exists; use write() to overwrite.")
        self._entries.setdefault(item_id, {})[asset_name] = _MemoryArrayEntry(array=array, extent=extent)

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_type="memory",
            source_id=self._source_id,
            open_config={"entries": self._entries},
            capabilities={
                "dimensions": self.dimensions,
                "asset_names": self._asset_names,
            },
        )


@dataclass(slots=True)
class MemoryDataSource(DataSource):
    """Runtime data source over in-memory numpy arrays.

    Opened from a SourceDescriptor produced by MemoryArrayStore.describe().
    Dimensions and asset_names are read from descriptor.capabilities.
    """

    source_type: str = "memory"
    thread_safe: bool = True
    fork_safe: bool = False
    dimensions: Dimensions = field(default_factory=Dimensions)
    _entries: dict[Any, dict[str, _MemoryArrayEntry]] = field(default_factory=dict)
    _asset_names: dict[int, str] = field(default_factory=lambda: {0: "data"})
    _source_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "MemoryDataSource":
        return cls(
            dimensions=descriptor.capabilities.get("dimensions", Dimensions()),
            _entries=descriptor.open_config.get("entries", {}),
            _asset_names=descriptor.capabilities.get("asset_names", {0: "data"}),
            _source_id=descriptor.source_id,
        )

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_type="memory",
            source_id=self._source_id,
            open_config={"entries": self._entries},
            capabilities={
                "dimensions": self.dimensions,
                "asset_names": self._asset_names,
            },
        )

    def materialize(self, accessor: DataAccessor) -> Any:
        asset_name = self._asset_names[accessor.asset_id]
        entry = self._entries[accessor.item_id][asset_name]
        if accessor.dimensioned_slice is not None:
            resolved = self.dimensions.resolve(accessor.dimensioned_slice)
            return entry.array[tuple(di.value for di in resolved)]
        return entry.array

    def inspect(self, accessor: DataAccessor) -> Mapping[str, Any]:
        asset_name = self._asset_names[accessor.asset_id]
        entry = self._entries[accessor.item_id][asset_name]
        return {
            "shape": tuple(entry.array.shape),
            "dtype": str(entry.array.dtype),
            "extent": entry.extent,
            "asset_id": accessor.asset_id,
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def extent_for(self, item_id: Any) -> DimensionedSlice | None:
        item_entries = self._entries.get(item_id)
        if not item_entries:
            return None
        return next(iter(item_entries.values())).extent

    def slice_accessor(self, accessor: DataAccessor, dim_slice: DimensionedSlice) -> DataAccessor:
        unknown = set(dim_slice.dims) - set(self.dimensions.names())
        if unknown:
            raise ValueError(f"DimensionedSlice references unknown dimensions: {sorted(unknown)}")
        return accessor.slice(dim_slice)
