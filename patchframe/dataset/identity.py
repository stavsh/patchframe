"""Semantic identity helpers for dataset indexes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class IndexIdentity:
    """Stable semantic namespace for labels in a dataset index."""

    id: str


def new_index_identity() -> IndexIdentity:
    """Return a fresh semantic identity for a dataset index namespace."""

    return IndexIdentity(id=str(uuid4()))


@dataclass(frozen=True, slots=True)
class FieldIdentity:
    """Stable semantic identity for a schema field across operator transitions.

    Minted at field construction, preserved when a field is renamed, retyped,
    or otherwise transformed via ``dataclasses.replace``, and freshly minted
    for genuinely new fields.

    Excluded from structural field equality (``Field`` declares it
    ``compare=False``): two fields with the same shape but different lineage
    are still ``==``. Compare ``field_identity`` explicitly where lineage
    matters. Also intended as the stable token a future field-reference layer
    resolves against.
    """

    id: str


def new_field_identity() -> FieldIdentity:
    """Return a fresh semantic identity for a schema field."""

    return FieldIdentity(id=str(uuid4()))


def primary_index_field(schema: Any) -> Any:
    """Return the primary ``IndexField`` from a schema-like object."""

    from patchframe.dataset.fields import IndexField

    fields = [field for field in schema if isinstance(field, IndexField)]
    if len(fields) != 1:
        raise ValueError(f"Expected exactly one IndexField, found {len(fields)}.")
    return fields[0]


def maybe_primary_index_field(schema: Any) -> Any | None:
    """Return the primary ``IndexField`` when present."""

    from patchframe.dataset.fields import IndexField

    fields = [field for field in schema if isinstance(field, IndexField)]
    if not fields:
        return None
    if len(fields) != 1:
        raise ValueError(f"Expected at most one IndexField, found {len(fields)}.")
    return fields[0]


def primary_index_identity(state_or_schema: Any) -> IndexIdentity:
    """Return the primary index identity from a state- or schema-like object."""

    schema = getattr(state_or_schema, "schema", state_or_schema)
    field = primary_index_field(schema)
    if field.identity is None:
        raise ValueError(f"IndexField {field.name!r} does not have an identity.")
    return field.identity


def with_primary_index_identity(schema: Any, identity: IndexIdentity) -> Any:
    """Return ``schema`` with its primary ``IndexField`` assigned ``identity``."""

    from patchframe.dataset.fields import IndexField
    from patchframe.dataset.schema import Schema

    fields = tuple(
        replace(field, identity=identity) if isinstance(field, IndexField) else field
        for field in schema
    )
    return Schema(fields=fields)


def ensure_primary_index_identity(schema: Any) -> Any:
    """Mint a primary index identity if the schema has an anonymous IndexField."""

    field = maybe_primary_index_field(schema)
    if field is None or field.identity is not None:
        return schema
    return with_primary_index_identity(schema, new_index_identity())


def mint_primary_index_identity(schema: Any) -> Any:
    """Return ``schema`` with a fresh primary index identity when it has an index."""

    if maybe_primary_index_field(schema) is None:
        return schema
    return with_primary_index_identity(schema, new_index_identity())


def foreign_index_fields(
    schema: Any,
    target_identity: IndexIdentity | None = None,
) -> tuple[Any, ...]:
    """Return foreign index fields in ``schema``, optionally filtered by target."""

    from patchframe.dataset.fields import ForeignIndexField

    fields = tuple(field for field in schema if isinstance(field, ForeignIndexField))
    if target_identity is None:
        return fields
    return tuple(field for field in fields if field.target_identity == target_identity)


def resolve_foreign_index_field(
    schema: Any,
    target_identity: IndexIdentity,
    *,
    field_name: str | None = None,
    op_name: str = "foreign index resolver",
) -> Any:
    """Resolve a foreign index field targeting ``target_identity``.

    If ``field_name`` is omitted, the schema must contain exactly one matching
    foreign index field.
    """

    from patchframe.dataset.fields import ForeignIndexField

    if field_name is not None:
        field = schema.get(field_name)
        if not isinstance(field, ForeignIndexField):
            raise TypeError(f"{op_name}: field {field_name!r} is not a ForeignIndexField.")
        if field.target_identity != target_identity:
            raise ValueError(
                f"{op_name}: ForeignIndexField {field_name!r} targets a different "
                "index identity."
            )
        return field

    matches = foreign_index_fields(schema, target_identity)
    if len(matches) != 1:
        raise ValueError(
            f"{op_name}: expected exactly one ForeignIndexField targeting the "
            f"requested index identity, found {len(matches)}."
        )
    return matches[0]
