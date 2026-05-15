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
from typing import Any

import numpy as np

from patchframe.data.accessor import DataAccessor
from patchframe.data.array_source import ArrayDataSource
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions
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

    def write(
        self,
        item_id: Any,
        asset_name: str,
        array: np.ndarray,
        extent: DimensionedSlice,
    ) -> None:
        self._entries.setdefault(item_id, {})[asset_name] = _MemoryArrayEntry(
            array=array,
            extent=extent,
        )

    def append(
        self,
        item_id: Any,
        asset_name: str,
        array: np.ndarray,
        extent: DimensionedSlice,
    ) -> None:
        if asset_name in self._entries.get(item_id, {}):
            raise ValueError(
                f"Entry ({item_id!r}, {asset_name!r}) already exists; "
                "use write() to overwrite."
            )
        self._entries.setdefault(item_id, {})[asset_name] = _MemoryArrayEntry(
            array=array,
            extent=extent,
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


class MemoryDataSource(ArrayDataSource):
    """Runtime data source over in-memory numpy arrays.

    Opened from a SourceDescriptor produced by MemoryArrayStore.describe().
    Uses ArrayDataSource for descriptor roundtrip, dimension validation, and
    full-read-plus-slice materialization.
    """

    source_type = "memory"
    thread_safe: bool = True
    fork_safe: bool = False
    config_fields = ("entries", "asset_names")

    def __init__(
        self,
        *,
        dimensions: Dimensions | None = None,
        source_id: str | None = None,
        entries: dict[Any, dict[str, _MemoryArrayEntry]] | None = None,
        asset_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions,
            source_id=source_id or str(uuid.uuid4()),
            entries=entries or {},
            asset_names=asset_names or {0: "data"},
        )

    def read_full(self, item_id: Any, accessor: DataAccessor) -> Any:
        asset_name = self.asset_names[accessor.asset_id]
        entry = self.entries[item_id][asset_name]
        return entry.array

    def inspect(self, accessor: DataAccessor) -> dict[str, Any]:
        asset_name = self.asset_names[accessor.asset_id]
        entry = self.entries[accessor.item_id][asset_name]
        return {
            "shape": tuple(entry.array.shape),
            "dtype": str(entry.array.dtype),
            "extent": entry.extent,
            "asset_id": accessor.asset_id,
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def extent_for(self, item_id: Any) -> DimensionedSlice | None:
        item_entries = self.entries.get(item_id)
        if not item_entries:
            return None
        return next(iter(item_entries.values())).extent
