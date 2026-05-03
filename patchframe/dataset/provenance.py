"""
patchframe.dataset.provenance

Source provenance records for patchframe datasets.

DatasetSourceInfo describes where a dataset was created from. It is serializable,
carried through operations by default, and intended for source-tracking workflows
such as upsert-back patterns and multi-source labeling UIs.
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
