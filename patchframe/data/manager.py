"""
patchframe.data.manager

Process-local source management for patchframe.

``SourceManager`` owns live ``DataSource`` instances and is responsible for
opening, reusing, leasing, and closing them. It should be treated as a runtime
facility rather than durable dataset state.

A module-level default manager (``get_default_manager()``) is used automatically
by ``DataAccessor.materialize()`` and ``CreationOperator`` when no explicit
manager is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from patchframe.data.descriptor import SourceDescriptor

if TYPE_CHECKING:
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
        self._source_types: dict[str, type[DataSource]] = {}

    def register_source_type(self, source_cls: type[DataSource]) -> None:
        """Register a ``DataSource`` class by its ``source_type``."""
        self._source_types[source_cls.source_type] = source_cls

    def normalize_key(self, descriptor: SourceDescriptor) -> tuple[Any, ...]:
        """Return a stable cache key for the given descriptor."""
        return (descriptor.source_type, descriptor.source_id)

    def register_descriptor(self, descriptor: SourceDescriptor) -> int:
        """Register a descriptor and return its integer id (idempotent)."""
        key = self.normalize_key(descriptor)
        if key in self._id_by_key:
            return self._id_by_key[key]
        desc_id = self._next_descriptor_id
        self._next_descriptor_id += 1
        self._id_by_key[key] = desc_id
        self._descriptor_by_id[desc_id] = descriptor
        return desc_id

    def register_source(self, source: DataSource) -> int:
        """Register a live DataSource and its descriptor. Returns source_desc_id.

        Calls source.describe() to obtain the descriptor, registers the source
        type and descriptor, and stores the live instance so it is reused on
        subsequent get_source() calls. Idempotent: registering the same source
        a second time returns the existing id.
        """
        descriptor = source.describe()
        self._source_types[descriptor.source_type] = type(source)
        desc_id = self.register_descriptor(descriptor)
        key = self.normalize_key(descriptor)
        self._sources[key] = source
        return desc_id

    def descriptor_for_id(self, descriptor_id: int) -> SourceDescriptor:
        """Return the descriptor associated with the given id."""
        return self._descriptor_by_id[descriptor_id]

    def get_source(self, descriptor: SourceDescriptor) -> DataSource:
        """Return the live source for the given descriptor, opening it if needed."""
        key = self.normalize_key(descriptor)
        if key not in self._sources:
            source_cls = self._source_types.get(descriptor.source_type)
            if source_cls is None:
                raise KeyError(f"Unknown source type: {descriptor.source_type!r}")
            self._sources[key] = source_cls.open(descriptor)
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
        """Release a source lease (refcounting not yet implemented)."""
        return None

    def close_all(self) -> None:
        """Close all live sources."""
        for source in self._sources.values():
            source.close()
        self._sources.clear()


# ---------------------------------------------------------------------------
# Module-level default manager
# ---------------------------------------------------------------------------

_default_manager: SourceManager | None = None


def get_default_manager() -> SourceManager:
    """Return the process-wide default SourceManager, creating it on first call."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SourceManager()
    return _default_manager


def reset_default_manager() -> None:
    """Replace the default manager with a fresh instance.

    Useful in tests to ensure source isolation between test cases.
    """
    global _default_manager
    _default_manager = SourceManager()
