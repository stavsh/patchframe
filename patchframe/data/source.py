"""
patchframe.data.source

Runtime data-source interface for patchframe.

A ``DataSource`` interprets and materializes ``DataAccessor`` instances. It is
a live process-local object managed by ``SourceManager`` and may be backed by
an ``ArrayStore`` or by some other runtime data provider.
"""

from __future__ import annotations

from typing import Any

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.slices import SliceSpec


class DataSource:
    """Base runtime data-source interface."""

    source_type: str = "base"
    thread_safe: bool = False
    fork_safe: bool = False

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "DataSource":
        """Open a live data source from a source descriptor."""
        raise NotImplementedError

    def materialize(self, accessor: DataAccessor) -> Any:
        """Materialize the given accessor into an in-memory object."""
        raise NotImplementedError

    def inspect(self, accessor: DataAccessor) -> dict[str, Any]:
        """Return lightweight metadata about the given accessor."""
        raise NotImplementedError

    def slice_accessor(self, accessor: DataAccessor, spec: SliceSpec) -> DataAccessor:
        """Return a new accessor representing a sliced view."""
        raise NotImplementedError

    def close(self) -> None:
        """Close any live resources associated with this source."""
        return None