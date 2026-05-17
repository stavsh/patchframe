"""Field composition policies used by concat/merge-style operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from patchframe.dataset.fields import (
    DimensionField,
    Field,
    IndexColumnField,
    IndexField,
    ValueField,
)
from patchframe.dataset.identity import new_index_identity

CompositionRole = Literal["row_stack", "column_add", "key_coalesce", "collision"]
CollisionMode = Literal["error", "keep", "update_missing", "coalesce", "rename"]
CollisionSide = Literal["left", "right"]
ConflictMode = Literal["keep_chosen", "raise"]


@dataclass(frozen=True, slots=True)
class CompositionContext:
    """Operational context passed to registered field policies."""

    role: CompositionRole
    op: str | None = None
    side: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnCollisionStrategy:
    """Policy for resolving table columns that map to the same output field."""

    mode: CollisionMode = "error"
    side: CollisionSide = "left"
    on_conflict: ConflictMode = "keep_chosen"


def _normal_field(field: Field) -> Field:
    if isinstance(field, IndexField) or field.nullable:
        return field
    return replace(field, nullable=True)


class FieldCompositionPolicy:
    """Base policy for composing fields of the same concrete type."""

    def check_row_compatible(
        self,
        field: Field,
        others: tuple[Field, ...],
        context: CompositionContext,
    ) -> None:
        for other in others:
            if type(field) is not type(other):
                raise TypeError(
                    f"Cannot compose {type(field).__name__} with {type(other).__name__}."
                )
            if field.dtype != other.dtype:
                raise TypeError(
                    f"Cannot compose field {field.name!r}: dtype {field.dtype!r} "
                    f"does not match {other.dtype!r}."
                )

    def compose_rows(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        field = _first_field(fields)
        self.check_row_compatible(field, fields[1:], context)
        return _normal_field(field)

    def compose_column(
        self,
        field: Field,
        existing_fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        result = _normal_field(field)
        if result.primary and any(
            type(existing) is type(result) and existing.primary for existing in existing_fields
        ):
            result = replace(result, primary=False)
        return result

    def compose_key(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        return self.compose_rows(fields, context)

    def normalize_column(
        self,
        series: pd.Series,
        field: Field,
        context: CompositionContext,
    ) -> pd.Series:
        if field.dtype is None:
            return series
        try:
            field.validate_column(series)
        except ValueError:
            return series.astype(field.dtype)
        return series


class ValueFieldCompositionPolicy(FieldCompositionPolicy):
    """ValueFields row-compose by logical field type; pandas owns dtype widening."""

    def check_row_compatible(
        self,
        field: Field,
        others: tuple[Field, ...],
        context: CompositionContext,
    ) -> None:
        for other in others:
            if type(field) is not type(other):
                raise TypeError(
                    f"Cannot compose {type(field).__name__} with {type(other).__name__}."
                )

    def compose_rows(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        field = _first_field(fields)
        self.check_row_compatible(field, fields[1:], context)
        dtypes = {f.dtype for f in fields}
        dtype = field.dtype if len(dtypes) == 1 else None
        return replace(_normal_field(field), dtype=dtype)


class DimensionFieldCompositionPolicy(FieldCompositionPolicy):
    """DimensionFields require matching dimension objects and dtypes."""

    def check_row_compatible(
        self,
        field: Field,
        others: tuple[Field, ...],
        context: CompositionContext,
    ) -> None:
        super().check_row_compatible(field, others, context)
        if not isinstance(field, DimensionField):
            return
        for other in others:
            if not isinstance(other, DimensionField):
                continue
            if field.dimension != other.dimension:
                raise TypeError(
                    f"Cannot compose DimensionField {field.name!r}: dimension "
                    f"{field.dimension!r} does not match {other.dimension!r}."
                )


class IndexFieldCompositionPolicy(FieldCompositionPolicy):
    """Index fields remain a special identity field in column composition."""

    def compose_rows(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        self.check_row_compatible(_first_field(fields), fields[1:], context)
        identities = {
            field.identity
            for field in fields
            if isinstance(field, IndexField) and field.identity is not None
        }
        result = _first_field(fields)
        if len(identities) > 1:
            return replace(result, identity=new_index_identity())
        if len(identities) == 1:
            return replace(result, identity=next(iter(identities)))
        return result

    def compose_column(
        self,
        field: Field,
        existing_fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        if any(isinstance(existing, IndexField) for existing in existing_fields):
            return IndexColumnField(
                name=field.name,
                dtype=field.dtype,
                metadata=field.metadata,
                index_identity=field.identity if isinstance(field, IndexField) else None,
            )
        return field


class IndexColumnFieldCompositionPolicy(FieldCompositionPolicy):
    """Index reference columns compose only when they target the same identity."""

    def compose_rows(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        self.check_row_compatible(_first_field(fields), fields[1:], context)
        identities = {
            field.index_identity
            for field in fields
            if isinstance(field, IndexColumnField) and field.index_identity is not None
        }
        if len(identities) > 1:
            raise TypeError("Cannot compose IndexColumnField values with different identities.")
        result = _normal_field(_first_field(fields))
        if identities and isinstance(result, IndexColumnField):
            return replace(result, index_identity=next(iter(identities)))
        return result

    def compose_key(
        self,
        fields: tuple[Field, ...],
        context: CompositionContext,
    ) -> Field:
        return self.compose_rows(fields, context)


_FIELD_POLICIES: dict[type[Field], FieldCompositionPolicy] = {}


def register_field_policy(field_type: type[Field], policy: FieldCompositionPolicy) -> None:
    """Register the policy used for a Field subclass."""
    _FIELD_POLICIES[field_type] = policy


def field_policy_for(field: Field) -> FieldCompositionPolicy:
    """Return the nearest registered policy for ``field``."""
    for cls in type(field).__mro__:
        policy = _FIELD_POLICIES.get(cls)
        if policy is not None:
            return policy
    raise TypeError(f"No field composition policy registered for {type(field).__name__}.")


def compose_rows(
    fields: tuple[Field, ...],
    context: CompositionContext | None = None,
) -> Field:
    """Return the output field for row-wise stacking of compatible fields."""
    field = _first_field(fields)
    return field_policy_for(field).compose_rows(
        fields,
        context or CompositionContext(role="row_stack"),
    )


def compose_column(
    field: Field,
    existing_fields: tuple[Field, ...],
    context: CompositionContext | None = None,
) -> Field:
    """Return ``field`` adjusted for column-wise composition into existing fields."""
    return field_policy_for(field).compose_column(
        field,
        existing_fields,
        context or CompositionContext(role="column_add"),
    )


def compose_key(
    fields: tuple[Field, ...],
    context: CompositionContext | None = None,
) -> Field:
    """Return the output field for coalesced merge keys."""
    field = _first_field(fields)
    return field_policy_for(field).compose_key(
        fields,
        context or CompositionContext(role="key_coalesce"),
    )


def resolve_column_collision(
    left: pd.Series,
    right: pd.Series,
    strategy: ColumnCollisionStrategy,
) -> pd.Series:
    """Resolve two table columns according to a value collision strategy."""
    if strategy.mode == "rename":
        raise ValueError("rename collisions must be resolved before value composition.")
    if strategy.mode == "error":
        raise ValueError("Column collision is not resolvable with mode='error'.")

    chosen, other = (left, right) if strategy.side == "left" else (right, left)
    _check_value_conflicts(chosen, other, strategy)

    if strategy.mode == "keep":
        return chosen.copy()
    if strategy.mode in {"update_missing", "coalesce"}:
        return chosen.where(chosen.notna(), other)

    raise ValueError(f"Unknown collision strategy: {strategy.mode!r}.")


def normalize_column(
    series: pd.Series,
    field: Field,
    context: CompositionContext | None = None,
) -> pd.Series:
    """Return a table column normalized through the registered field policy."""
    return field_policy_for(field).normalize_column(
        series,
        field,
        context or CompositionContext(role="column_add"),
    )


def _check_value_conflicts(
    chosen: pd.Series,
    other: pd.Series,
    strategy: ColumnCollisionStrategy,
) -> None:
    if strategy.on_conflict != "raise":
        return
    both_present = chosen.notna() & other.notna()
    conflicts = both_present & (chosen != other)
    if conflicts.fillna(False).any():
        raise ValueError("Column collision has conflicting non-null values.")


def _first_field(fields: tuple[Field, ...]) -> Field:
    if not fields:
        raise ValueError("At least one field is required for composition.")
    return fields[0]


register_field_policy(Field, FieldCompositionPolicy())
register_field_policy(ValueField, ValueFieldCompositionPolicy())
register_field_policy(DimensionField, DimensionFieldCompositionPolicy())
register_field_policy(IndexField, IndexFieldCompositionPolicy())
register_field_policy(IndexColumnField, IndexColumnFieldCompositionPolicy())
