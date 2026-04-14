"""
patchframe.dataset.bindings

Serializable binding declarations for patchframe datasets.

Bindings describe relationships between fields, but they do not implement the
runtime logic for those relationships. Runtime validation and interpretation are
handled by ``BindEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BindingSpec:
    """Single binding declaration.

    Parameters
    ----------
    kind:
        Binding kind identifier, for example ``data_slicing``.
    subject:
        Primary field or object being bound.
    object:
        Secondary field or object referenced by the binding.
    payload:
        Optional binding-specific metadata.
    """

    kind: str
    subject: str
    object: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BindingSpecSet:
    """Collection of binding declarations."""

    specs: tuple[BindingSpec, ...] = field(default_factory=tuple)

    def add(self, *specs: BindingSpec) -> "BindingSpecSet":
        """Return a new binding set with additional specs."""
        return BindingSpecSet(specs=self.specs + tuple(specs))

    def filter_by_kind(self, kind: str) -> tuple[BindingSpec, ...]:
        """Return all binding specs of the given kind."""
        return tuple(spec for spec in self.specs if spec.kind == kind)

    def rewrite_field_names(self, mapping: dict[str, str]) -> "BindingSpecSet":
        """Return a new binding set with field references rewritten."""
        rewritten = []
        for spec in self.specs:
            rewritten.append(
                BindingSpec(
                    kind=spec.kind,
                    subject=mapping.get(spec.subject, spec.subject),
                    object=mapping.get(spec.object, spec.object),
                    payload=spec.payload,
                )
            )
        return BindingSpecSet(specs=tuple(rewritten))