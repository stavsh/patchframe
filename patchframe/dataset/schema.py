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

        # No two fields may claim the same table column. With 1:1 fields this is
        # implied by unique names, but a CompositeField spans dotted columns
        # (location.lat), so a stray field named "location.lat" — or two
        # composites — could collide. Fields own their columns; overlap is a bug.
        claimed = [col for f in self.fields for col in f.table_columns()]
        if len(claimed) != len(set(claimed)):
            seen: set[str] = set()
            dupes: set[str] = set()
            for col in claimed:
                (dupes if col in seen else seen).add(col)
            raise ValueError(
                f"Schema fields claim overlapping table columns: {sorted(dupes)}"
            )

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

    def table_column_renames(self, mapping: dict[str, str]) -> dict[str, str]:
        """Map old table-column names to new for a field-rename ``mapping``.

        Delegates to each renamed field: a normal field renames its one column, a
        ``CompositeField`` re-prefixes its dotted columns, an index field
        contributes none (its axis renames separately).
        """
        renames: dict[str, str] = {}
        for old, new in mapping.items():
            renames.update(self.get(old).rename_table_columns(new))
        return renames

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
        """Validate that all schema fields are present in the table with compatible dtypes.

        A ``CompositeField`` occupies its N dotted columns (``{name}.{sub}``)
        rather than one column named after it, so it is checked against those.
        """
        missing: list[str] = []
        for f in self.fields:
            for col in f.table_columns():  # () for the index; dotted for a composite
                if col not in table.columns:
                    missing.append(col)
        if missing:
            raise ValueError(f"Table is missing schema fields: {missing}")

        for f in self.fields:
            f.validate_in_table(table)

    @classmethod
    def from_fields(cls, fields: Iterable[Field]) -> "Schema":
        """Construct a schema from an iterable of fields."""
        return cls(fields=tuple(fields))
