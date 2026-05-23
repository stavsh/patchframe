"""Shared helpers for built-in composition operators."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from patchframe.dataset.couplings import Coupling, CouplingSet
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    MergedField,
    normalize_column,
)
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState


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


def superseded_names_by_input(schema: Schema | None) -> dict[int, frozenset[str]]:
    """Map each input index to the field names where that input lost a collision."""
    if schema is None:
        return {}
    superseded: dict[int, set[str]] = {}
    for field in schema:
        if not isinstance(field, MergedField) or field.collision is None:
            continue
        winner = field.winning_parent().input_index
        for parent in field.parents:
            if parent.input_index != winner:
                superseded.setdefault(parent.input_index, set()).add(field.name)
    return {index: frozenset(names) for index, names in superseded.items()}


def derive_composed_couplings(
    states: tuple[DatasetState, ...],
    composed_schema: Schema | None,
) -> CouplingSet:
    """Union input couplings, dropping those an input lost to a collision.

    A coupling from an input that lost a name collision and references the
    superseded name is pruned; the rest are unioned (deduplicated).
    """
    superseded = superseded_names_by_input(composed_schema)
    result: list[Coupling] = []
    for input_index, state in enumerate(states):
        lost = superseded.get(input_index, frozenset())
        for coupling in state.couplings.couplings:
            touched = (coupling.output_field(), *coupling.input_fields())
            if lost and not lost.isdisjoint(touched):
                continue
            if coupling not in result:
                result.append(coupling)
    return CouplingSet(couplings=tuple(result))
