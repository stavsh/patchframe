"""
patchframe.dataset.bind_engine

Compiled binding interpretation layer for patchframe.

``BindEngine`` validates binding specs against a schema and exposes convenient
lookup helpers for binding-aware operations. It should remain small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass

from patchframe.dataset.bindings import BindingSpecSet
from patchframe.dataset.schema import Schema


@dataclass(frozen=True, slots=True)
class BindEngine:
    """Compiled binding view over a schema and binding spec set."""

    schema: Schema
    bindings: BindingSpecSet

    def validate(self) -> None:
        """Validate bindings against the schema."""
        field_names = set(self.schema.names())
        for spec in self.bindings.specs:
            if spec.subject not in field_names:
                raise ValueError(f"Binding subject does not exist in schema: {spec.subject}")
            if spec.object not in field_names:
                raise ValueError(f"Binding object does not exist in schema: {spec.object}")

    def slice_field_for_data_field(self, data_field_name: str) -> str | None:
        """Return the bound slice-spec field for the given data field, if any."""
        for spec in self.bindings.filter_by_kind("data_slicing"):
            if spec.subject == data_field_name:
                return spec.object
        return None

    def default_data_field(self) -> str | None:
        """Return the default data field declared in bindings, if any."""
        for spec in self.bindings.filter_by_kind("default_data_field"):
            return spec.subject
        return None