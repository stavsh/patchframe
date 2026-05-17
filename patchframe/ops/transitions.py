"""
patchframe.ops.transitions

Transition metadata for patchframe operators.

Operators in patchframe are defined over multiple dataset aspects, not just the
row table. Transition metadata makes those structural effects explicit and
inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AspectTransition:
    """Single aspect-level transition declaration."""

    mode: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    """Declared transitions of an operator across dataset aspects."""

    schema: AspectTransition = field(default_factory=lambda: AspectTransition("preserve"))
    table: AspectTransition = field(default_factory=lambda: AspectTransition("preserve"))
    couplings: AspectTransition = field(default_factory=lambda: AspectTransition("preserve"))
    sources: AspectTransition = field(default_factory=lambda: AspectTransition("inherit"))
    accessors: AspectTransition = field(default_factory=lambda: AspectTransition("preserve"))
    index_identity: AspectTransition = field(
        default_factory=lambda: AspectTransition("preserve")
    )
