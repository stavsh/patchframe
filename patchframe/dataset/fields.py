"""
patchframe.dataset.fields

Core field definitions for patchframe schemas.

Fields describe what kind of values a column holds, but they do not encode
cross-field relationships. Relationships between fields belong in bindings,
not in field definitions.

The core package intentionally keeps the field model shallow and non-geometric.
Extension packages may introduce richer field types later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Field:
    """Base field definition.

    Parameters
    ----------
    name:
        Column name in the dataset table.
    logical_type:
        Logical type identifier for the field, such as ``index``, ``value``,
        ``data``, or ``slice_spec``.
    nullable:
        Whether null values are allowed in the column.
    metadata:
        Optional field-level metadata. This must not contain executable logic.
    """

    name: str
    logical_type: str
    nullable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexField(Field):
    """Field representing the dataset index column."""

    logical_type: str = "index"
    nullable: bool = False


@dataclass(frozen=True, slots=True)
class ValueField(Field):
    """Field representing a regular scalar/object value column."""

    logical_type: str = "value"


@dataclass(frozen=True, slots=True)
class DataField(Field):
    """Field representing a lazy data-access column.

    Values in a data field are expected to be ``DataAccessor`` instances.
    """

    logical_type: str = "data"


@dataclass(frozen=True, slots=True)
class SliceSpecField(Field):
    """Field representing a column of ``SliceSpec`` values."""

    logical_type: str = "slice_spec"