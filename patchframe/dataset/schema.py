"""
patchframe.dataset.schema

Schema container for patchframe datasets.

A schema is an ordered collection of fields. It owns no runtime source state
and no executable coupling logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

import pandas as pd

from patchframe.dataset.fields import Field


@dataclass(frozen=True, slots=True)
class Schema:
    """Ordered collection of dataset fields."""

    fields: tuple[Field, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"Schema contains duplicate field names: {names}")

        primary_counts: dict[type, int] = defaultdict(int)
        for f in self.fields:
            if f.primary:
                primary_counts[type(f)] += 1
        violations = [cls.__name__ for cls, count in primary_counts.items() if count > 1]
        if violations:
            raise ValueError(f"Schema has multiple primary fields for types: {violations}")

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def names(self) -> tuple[str, ...]:
        """Return field names in schema order."""
        return tuple(f.name for f in self.fields)

    def has(self, name: str) -> bool:
        """Return whether the schema contains a field with the given name."""
        return any(f.name == name for f in self.fields)

    def get(self, name: str) -> Field:
        """Return the field with the given name."""
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def add(self, *new_fields: Field) -> "Schema":
        """Return a new schema with fields appended."""
        return Schema(fields=self.fields + tuple(new_fields))

    def drop(self, *names: str) -> "Schema":
        """Return a new schema without the named fields."""
        to_drop = set(names)
        return Schema(fields=tuple(f for f in self.fields if f.name not in to_drop))

    def rename(self, mapping: dict[str, str]) -> "Schema":
        """Return a new schema with renamed fields."""
        return Schema(fields=tuple(
            replace(f, name=mapping[f.name]) if f.name in mapping else f
            for f in self.fields
        ))

    def retype(self, name: str, replacement: Field) -> "Schema":
        """Return a new schema with one field replaced."""
        replaced = []
        found = False
        for f in self.fields:
            if f.name == name:
                replaced.append(replacement)
                found = True
            else:
                replaced.append(f)
        if not found:
            raise KeyError(name)
        return Schema(fields=tuple(replaced))

    def validate_table(self, table: pd.DataFrame) -> None:
        """Validate that all schema fields are present in the table with compatible dtypes."""
        missing = [
            f.name for f in self.fields
            if f.logical_type != "index" and f.name not in table.columns
        ]
        if missing:
            raise ValueError(f"Table is missing schema fields: {missing}")

        for f in self.fields:
            if f.logical_type == "index":
                f.validate_column(table.index.to_series())
            else:
                f.validate_column(table[f.name])

    @classmethod
    def from_fields(cls, fields: Iterable[Field]) -> "Schema":
        """Construct a schema from an iterable of fields."""
        return cls(fields=tuple(fields))
