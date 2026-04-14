"""
patchframe.ops.effects

Effect metadata for patchframe operations.

Operations in patchframe are defined over multiple dataset aspects, not just the
table. Effect metadata makes those transformations explicit and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AspectEffect:
    """Single aspect-level effect declaration."""

    mode: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OperationEffects:
    """Declared effects of an operation across dataset aspects."""

    schema: AspectEffect = field(default_factory=lambda: AspectEffect("preserve"))
    table: AspectEffect = field(default_factory=lambda: AspectEffect("preserve"))
    bindings: AspectEffect = field(default_factory=lambda: AspectEffect("preserve"))
    provenance: AspectEffect = field(default_factory=lambda: AspectEffect("preserve"))
    accessors: AspectEffect = field(default_factory=lambda: AspectEffect("preserve"))