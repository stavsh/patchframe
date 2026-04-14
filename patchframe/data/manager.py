"""
patchframe.data.manager

Process-local source management for patchframe.

``SourceManager`` owns live ``DataSource`` instances and is responsible for
opening, reusing, leasing, and closing them. It should be treated as a runtime
facility rather than durable dataset state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.source import DataSource


@dataclass(frozen=True, slots=True)
class SourceLease:
    """Runtime lease token for a live source."""

    descriptor_key: tuple[Any, ...]
    owner: str | None = None


class SourceManager:
    """Process-local manager for live data sources."""

    def __init__(self) -> None:
        self._sources: dict[tuple[Any, ...], DataSource] = {}
        self._descriptor_by_id: dict[int, SourceDescriptor] = {}
        self._id_by_key: dict[tuple[Any, ...], int] = {}
        self._next_descriptor_id: int = 1

    def normalize_key(self, descriptor: SourceDescriptor) -> tuple[Any, ...]:
        """Return a stable cache key for the given descriptor."""
        return (
            descriptor.source_type,
            descriptor.source_id,
            tuple(sorted(descriptor.open_config.items())),
        )

    def register_descriptor(self, descriptor: SourceDescriptor) -> int:
        """Register a descriptor and return its integer id."""
        key = self.normalize_key(descriptor)
        if key in self._id_by_key:
            return self._id_by_key[key]
        desc_id = self._next_descriptor_id
        self._next_descriptor_id += 1
        self._id_by_key[key] = desc_id
        self._descriptor_by_id[desc_id] = descriptor
        return desc_id

    def descriptor_for_id(self, descriptor_id: int) -> SourceDescriptor:
        """Return the descriptor associated with the given id."""
        return self._descriptor_by_id[descriptor_id]

    def get_source(self, descriptor: SourceDescriptor) -> DataSource:
        """Open or reuse a live source for the given descriptor."""
        key = self.normalize_key(descriptor)
        if key not in self._sources:
            # TODO: replace this with registry-based dispatch.
            raise NotImplementedError("Source factory dispatch is not implemented yet.")
        return self._sources[key]

    def get_source_by_descriptor_id(self, descriptor_id: int) -> DataSource:
        """Resolve a descriptor id and return the corresponding live source."""
        descriptor = self.descriptor_for_id(descriptor_id)
        return self.get_source(descriptor)

    def acquire(self, descriptor: SourceDescriptor, owner: str | None = None) -> SourceLease:
        """Acquire a lease for the given descriptor."""
        key = self.normalize_key(descriptor)
        _ = self.get_source(descriptor)
        return SourceLease(descriptor_key=key, owner=owner)

    def release(self, lease: SourceLease) -> None:
        """Release a source lease.

        This stub does not yet implement refcounting.
        """
        return None

    def close_all(self) -> None:
        """Close all live sources."""
        for source in self._sources.values():
            source.close()
        self._sources.clear()