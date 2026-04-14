"""
patchframe.sources.memory.source

In-memory reference data source for patchframe.

``MemoryDataSource`` is the first concrete backend intended for the core
package. It provides a simple runtime implementation backed by in-memory numpy
arrays and is suitable for initial end-to-end testing.

This source should eventually support:
- multiple named assets per logical item
- lazy view/slice application
- inspection without materialization where possible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.source import DataSource
from patchframe.data.slices import SliceSpec


@dataclass(slots=True)
class MemoryDataSource(DataSource):
    """Simple in-memory data source backed by numpy arrays."""

    source_type: str = "memory"
    thread_safe: bool = True
    fork_safe: bool = False
    assets_by_item: dict[Any, dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "MemoryDataSource":
        """Open a memory-backed data source from a descriptor.

        This stub assumes a future implementation will resolve a process-local
        in-memory asset store from ``descriptor.open_config``.
        """
        raise NotImplementedError("MemoryDataSource.open is not implemented yet.")

    def materialize(self, accessor: DataAccessor) -> Any:
        """Materialize one accessor into a numpy array."""
        raise NotImplementedError("MemoryDataSource.materialize is not implemented yet.")

    def inspect(self, accessor: DataAccessor) -> Mapping[str, Any]:
        """Return lightweight metadata for one accessor."""
        raise NotImplementedError("MemoryDataSource.inspect is not implemented yet.")

    def slice_accessor(self, accessor: DataAccessor, spec: SliceSpec) -> DataAccessor:
        """Return a new accessor representing a sliced view."""
        raise NotImplementedError("MemoryDataSource.slice_accessor is not implemented yet.")