"""
patchframe.data.accessor

Tiny lazy data-access handle for patchframe.

DataAccessor stores only identity information. All materialization and
slicing is deferred to the DataSource associated with source_desc_id.

Manager resolution order for materialize() / inspect():
  1. Explicit ``manager`` argument
  2. ``manager_hint`` stamped on the accessor at creation time
  3. Process-wide default SourceManager
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patchframe.data.dimensioned_slice import DimensionedSlice
    from patchframe.data.manager import SourceManager


@dataclass(frozen=True, slots=True)
class DataAccessor:
    """Tiny immutable lazy accessor."""

    source_desc_id: int
    item_id: Any
    asset_id: int = 0
    view_id: int = 0
    dimensioned_slice: "DimensionedSlice | None" = None
    manager_hint: "SourceManager | None" = None

    def slice(self, dim_slice: "DimensionedSlice") -> "DataAccessor":
        """Return a new accessor with the given slice attached (lazy)."""
        return replace(self, dimensioned_slice=dim_slice)

    def materialize(self, manager: "SourceManager | None" = None) -> Any:
        """Materialize this accessor into an in-memory object."""
        return self._resolve_manager(manager).get_source_by_descriptor_id(self.source_desc_id).materialize(self)

    def inspect(self, manager: "SourceManager | None" = None) -> dict[str, Any]:
        """Return lightweight metadata about this accessor."""
        return self._resolve_manager(manager).get_source_by_descriptor_id(self.source_desc_id).inspect(self)

    def _resolve_manager(self, manager: "SourceManager | None") -> "SourceManager":
        if manager is not None:
            return manager
        if self.manager_hint is not None:
            return self.manager_hint
        from patchframe.data.manager import get_default_manager
        return get_default_manager()
