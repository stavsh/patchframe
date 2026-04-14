"""
patchframe.dataset.schema

Schema container for patchframe datasets.

A schema is an ordered collection of fields with lookup and structural update
helpers. It owns no executable binding logic and no runtime source state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import pandas as pd

from patchframe.dataset.schema import Field


@dataclass(frozen=True, slots=True)
class Schema:
    """Ordered collection of dataset fields."""

    fields: tuple[Field, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"Schema contains duplicate field names: {names}")

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def names(self) -> tuple[str, ...]:
        """Return field names in schema order."""
        return tuple(field.name for field in self.fields)

    def get(self, name: str) -> Field:
        """Return the field with the given name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def has(self, name: str) -> bool:
        """Return whether the schema contains a field with the given name."""
        return any(field.name == name for field in self.fields)

    def add(self, *new_fields: Field) -> "Schema":
        """Return a new schema with fields appended."""
        return Schema(fields=self.fields + tuple(new_fields))

    def drop(self, *names: str) -> "Schema":
        """Return a new schema without the named fields."""
        to_drop = set(names)
        return Schema(fields=tuple(field for field in self.fields if field.name not in to_drop))

    def rename(self, mapping: dict[str, str]) -> "Schema":
        """Return a new schema with renamed fields."""
        renamed: list[Field] = []
        for field in self.fields:
            if field.name in mapping:
                renamed.append(
                    type(field)(
                        name=mapping[field.name],
                        logical_type=field.logical_type,
                        nullable=field.nullable,
                        metadata=field.metadata,
                    )
                )
            else:
                renamed.append(field)
        return Schema(fields=tuple(renamed))

    def retype(self, name: str, replacement: Field) -> "Schema":
        """Return a new schema with one field replaced."""
        replaced = []
        found = False
        for field in self.fields:
            if field.name == name:
                replaced.append(replacement)
                found = True
            else:
                replaced.append(field)
        if not found:
            raise KeyError(name)
        return Schema(fields=tuple(replaced))

    def validate_table(self, table: pd.DataFrame) -> None:
        """Validate that all schema fields exist in the given table."""
        missing = [field.name for field in self.fields if field.name not in table.columns and field.logical_type != "index"]
        if missing:
            raise ValueError(f"Table is missing schema fields: {missing}")

    @classmethod
    def from_fields(cls, fields: Iterable[Field]) -> "Schema":
        """Construct a schema from an iterable of fields."""
        return cls(fields=tuple(fields))