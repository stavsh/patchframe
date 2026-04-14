"""
patchframe.data.slices

Generic slice-spec types for patchframe.

The core package does not assume geometry. Slice specs are opaque typed values
that can later be interpreted by a source or extension package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """Typed per-row slice request."""

    kind: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)