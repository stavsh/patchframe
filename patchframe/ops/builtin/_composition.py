"""Shared helpers for built-in composition operators."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    compose_column,
    compose_key,
    normalize_column,
    resolve_column_collision,
)
from patchframe.dataset.fields import Field
from patchframe.dataset.schema import Schema


def normalize_collision(
    collision: ColumnCollisionStrategy | str | None,
) -> ColumnCollisionStrategy:
    if collision is None:
        return ColumnCollisionStrategy()
    if isinstance(collision, str):
        return ColumnCollisionStrategy(mode=collision)
    return collision


def normalize_field_names(names: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def normalize_table_to_schema(
    table: pd.DataFrame,
    schema: Schema,
    context: CompositionContext,
) -> pd.DataFrame:
    result = table.copy()
    for field in schema:
        if field.name in result.columns:
            result[field.name] = normalize_column(result[field.name], field, context)
    return result


def compose_collision_field(
    existing: Field,
    incoming: Field,
    other_fields: tuple[Field, ...],
    strategy: ColumnCollisionStrategy,
    op_name: str,
) -> Field:
    if strategy.mode == "error":
        raise ValueError(f"{op_name}: field collision for {incoming.name!r}.")
    if strategy.mode == "rename":
        raise ValueError(f"{op_name}: rename collisions must be prepared before composition.")
    if strategy.mode == "keep":
        chosen = existing if strategy.side == "left" else incoming
        return compose_column(
            chosen,
            other_fields,
            CompositionContext(role="column_add", op=op_name),
        )

    field = compose_key(
        (existing, incoming),
        CompositionContext(role="key_coalesce", op=op_name),
    )
    return compose_column(
        replace(field, name=existing.name),
        other_fields,
        CompositionContext(role="column_add", op=op_name),
    )


def resolve_collision_column(
    name: str,
    existing: pd.Series,
    incoming: pd.Series,
    strategy: ColumnCollisionStrategy,
    op_name: str,
) -> pd.Series:
    try:
        return resolve_column_collision(existing, incoming, strategy)
    except ValueError as err:
        raise ValueError(f"{op_name}: column collision for {name!r}: {err}") from err


def union_couplings(*coupling_sets: CouplingSet) -> CouplingSet:
    couplings = []
    for coupling_set in coupling_sets:
        for coupling in coupling_set.couplings:
            if coupling not in couplings:
                couplings.append(coupling)
    return CouplingSet(couplings=tuple(couplings))


def preserve_row_couplings(*coupling_sets: CouplingSet, op_name: str) -> CouplingSet:
    if not coupling_sets:
        return CouplingSet()

    first = coupling_sets[0]
    if all(coupling_set == first for coupling_set in coupling_sets):
        return first

    if any(coupling_set.couplings for coupling_set in coupling_sets):
        raise ValueError(
            f"{op_name}: row concat cannot preserve different coupling sets. "
            "Consume coupled fields before concat and reapply couplings afterwards."
        )

    return CouplingSet()
