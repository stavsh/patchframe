"""
patchframe.dataset.provenance

Provenance models for patchframe datasets.

The core provenance layer tracks where a dataset came from and how it was
derived. It is intentionally separate from runtime source objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DatasetSourceInfo:
    """Serializable source provenance record."""

    source_uri: str
    source_type: str
    source_name: str = "unknown"
    source_backend: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LineageEntry:
    """Single lineage record describing one derivation step."""

    op_name: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Dataset provenance consisting of source infos and lineage entries."""

    sources: tuple[DatasetSourceInfo, ...] = field(default_factory=tuple)
    lineage: tuple[LineageEntry, ...] = field(default_factory=tuple)

    def add_source(self, source: DatasetSourceInfo) -> "DatasetProvenance":
        """Return a new provenance with one additional source."""
        return DatasetProvenance(sources=self.sources + (source,), lineage=self.lineage)

    def add_lineage(self, entry: LineageEntry) -> "DatasetProvenance":
        """Return a new provenance with one additional lineage entry."""
        return DatasetProvenance(sources=self.sources, lineage=self.lineage + (entry,))