"""
patchframe.data.source

Runtime data-source interface for patchframe.

A DataSource interprets and materializes DataAccessor instances. It is
opened from a SourceDescriptor by SourceManager and may be backed by an
ArrayStore or another runtime data provider.

Dimensions are owned by the SourceDescriptor (populated by ArrayStore.describe()
or DataSource.describe()) and read into the DataSource instance during open().
"""

from __future__ import annotations

from typing import Any

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions


class DataSource:
    """Base runtime data-source interface."""

    source_type: str = "base"
    thread_safe: bool = False
    fork_safe: bool = False
    dimensions: Dimensions  # set from SourceDescriptor.capabilities during open()

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "DataSource":
        """Open a live data source from a source descriptor."""
        raise NotImplementedError

    def describe(self) -> SourceDescriptor:
        """Return a SourceDescriptor that can reopen this source.

        Must be consistent with open(): open(source.describe()) should produce
        an equivalent source. Used by SourceManager.register_source() to
        register a live source into the manager.
        """
        raise NotImplementedError

    def materialize(self, accessor: DataAccessor) -> Any:
        """Materialize the given accessor into an in-memory object."""
        raise NotImplementedError

    def inspect(self, accessor: DataAccessor) -> dict[str, Any]:
        """Return lightweight metadata about the given accessor."""
        raise NotImplementedError

    def slice_accessor(self, accessor: DataAccessor, dim_slice: DimensionedSlice) -> DataAccessor:
        """Return a new accessor with the given slice attached.

        The base implementation delegates to accessor.slice(). Subclasses
        may override to validate the slice against their dimensions first.
        """
        return accessor.slice(dim_slice)

    def extent_for(self, item_id: Any) -> "DimensionedSlice | None":
        """Return the full-array extent for the given item, or None if unknown.

        Subclasses override this to expose per-item extents in natural units.
        Used by callers that need to know the shape of an item without
        materializing it.
        """
        return None

    def close(self) -> None:
        """Close any live resources associated with this source."""
        return None
