"""
patchframe.data.accessor

Tiny lazy data-access handle for patchframe.

Phase 1 stores small immutable ``DataAccessor`` objects directly in dataframe
object columns. They are intentionally minimal and primarily carry ids into
shared descriptor / asset / view tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patchframe.data.manager import SourceManager
    from patchframe.data.slices import SliceSpec


@dataclass(frozen=True, slots=True)
class DataAccessor:
    """Tiny immutable lazy accessor.

    Parameters
    ----------
    source_desc_id:
        Integer id into a shared source-descriptor table.
    item_id:
        Logical item identifier inside the source.
    asset_id:
        Integer id into a shared asset dictionary.
    view_id:
        Integer id into a shared view/slice table.
    manager_hint:
        Optional process-local source manager hint. This must not be relied on
        for correctness across serialization boundaries.
    """

    source_desc_id: int
    item_id: Any
    asset_id: int = 0
    view_id: int = 0
    manager_hint: "SourceManager | None" = None

    def slice(self, spec: "SliceSpec") -> "DataAccessor":
        """Return a new accessor with a new view.

        This stub does not yet compile ``SliceSpec`` into a shared view table.
        A later implementation should delegate that responsibility to a dataset
        state object or a source-aware view compiler.
        """
        raise NotImplementedError("Slice compilation is not implemented yet.")

    def materialize(self, manager: "SourceManager | None" = None) -> Any:
        """Materialize this accessor using a source manager."""
        mgr = manager or self.manager_hint
        if mgr is None:
            raise RuntimeError("No SourceManager available for materialization.")
        source = mgr.get_source_by_descriptor_id(self.source_desc_id)
        return source.materialize(self)

    def inspect(self, manager: "SourceManager | None" = None) -> dict[str, Any]:
        """Inspect this accessor using a source manager."""
        mgr = manager or self.manager_hint
        if mgr is None:
            raise RuntimeError("No SourceManager available for inspection.")
        source = mgr.get_source_by_descriptor_id(self.source_desc_id)
        return source.inspect(self)