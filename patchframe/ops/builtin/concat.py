"""patchframe.ops.builtin.concat"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    compose_column,
    compose_rows,
)
from patchframe.dataset.fields import Field, IndexField
from patchframe.dataset.identity import (
    new_index_identity,
    primary_index_identity,
    with_primary_index_identity,
)
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator
from patchframe.ops.builtin._composition import (
    compose_collision_field,
    normalize_collision,
    normalize_table_to_schema,
    preserve_row_couplings,
    resolve_collision_column,
    union_couplings,
)
from patchframe.ops.transitions import AspectTransition, TransitionPlan


class concat_rows(CompositionOperator):
    """Stack datasets by rows."""

    transitions = TransitionPlan(
        schema=AspectTransition("derive"),
        table=AspectTransition("derive"),
        couplings=AspectTransition("derive"),
        sources=AspectTransition("union"),
        index_identity=AspectTransition("row_stack"),
    )

    def apply_schema(self, *states: DatasetState, **_: Any) -> Schema:
        _require_states(states, self.name)
        output_fields: list[Field] = []
        for name in _field_name_order(states):
            fields = tuple(state.schema.get(name) for state in states if state.schema.has(name))
            field = compose_rows(fields, CompositionContext(role="row_stack", op=self.name))
            output_fields.append(
                compose_column(
                    field,
                    tuple(output_fields),
                    CompositionContext(role="column_add", op=self.name),
                )
            )
        return _with_primary_identity(
            Schema(fields=tuple(output_fields)),
            _row_stack_index_identity(states),
        )

    def apply_table(self, *states: DatasetState, **_: Any) -> pd.DataFrame:
        _require_states(states, self.name)
        schema = self.apply_schema(*states)
        column_names = tuple(field.name for field in schema if field.logical_type != "index")
        tables = []
        for state in states:
            table = _table_for_output_schema(state, schema)
            for name in column_names:
                if name not in table.columns:
                    table[name] = pd.NA
            tables.append(table.loc[:, column_names])
        result = pd.concat(tables, axis=0)
        return normalize_table_to_schema(
            result,
            schema,
            CompositionContext(role="row_stack", op=self.name),
        )

    def apply_couplings(self, *states: DatasetState, **_: Any) -> CouplingSet:
        return preserve_row_couplings(
            *(state.couplings for state in states),
            op_name=self.name,
        )


class concat_columns(CompositionOperator):
    """Compose datasets by columns, aligning rows by pandas index."""

    transitions = TransitionPlan(
        schema=AspectTransition("derive"),
        table=AspectTransition("derive"),
        couplings=AspectTransition("derive"),
        sources=AspectTransition("union"),
        index_identity=AspectTransition("align_rows"),
    )

    def apply_schema(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> Schema:
        _require_states(states, self.name)
        strategy = normalize_collision(collision)
        output_fields: list[Field] = []
        positions: dict[str, int] = {}
        for state in states:
            for field in state.schema:
                if field.name not in positions:
                    positions[field.name] = len(output_fields)
                    output_fields.append(
                        compose_column(
                            field,
                            tuple(output_fields),
                            CompositionContext(role="column_add", op=self.name),
                        )
                    )
                    continue

                index = positions[field.name]
                existing = output_fields[index]
                output_fields[index] = compose_collision_field(
                    existing,
                    field,
                    tuple(f for i, f in enumerate(output_fields) if i != index),
                    strategy,
                    self.name,
                )
        return _with_primary_identity(
            Schema(fields=tuple(output_fields)),
            _aligned_index_identity(states),
        )

    def apply_table(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        _require_states(states, self.name)
        strategy = normalize_collision(collision)
        schema = self.apply_schema(*states, collision=strategy)
        result = pd.DataFrame(index=states[0].table.index)
        for state in states:
            incoming_table = _table_for_output_schema(state, schema)
            result, incoming = result.align(incoming_table, join="outer", axis=0)
            for name in incoming.columns:
                if name not in result.columns:
                    result[name] = incoming[name]
                    continue
                result[name] = resolve_collision_column(
                    name,
                    result[name],
                    incoming[name],
                    strategy,
                    self.name,
                )

        column_names = tuple(field.name for field in schema if field.name in result.columns)
        result = result.loc[:, column_names]
        return normalize_table_to_schema(
            result,
            schema,
            CompositionContext(role="column_add", op=self.name),
        )

    def apply_couplings(self, *states: DatasetState, **_: Any) -> CouplingSet:
        return union_couplings(*(state.couplings for state in states))


class concat(CompositionOperator):
    """Dispatch to ``concat_rows`` or ``concat_columns``."""

    transitions = TransitionPlan(
        schema=AspectTransition("dispatch"),
        table=AspectTransition("dispatch"),
        couplings=AspectTransition("dispatch"),
        sources=AspectTransition("dispatch"),
        index_identity=AspectTransition("dispatch"),
    )

    def __call__(self, *datasets: Dataset, axis: int = 0, **kwargs: Any) -> Dataset:
        if axis == 0:
            return concat_rows.instance()(*datasets, **kwargs)
        if axis == 1:
            return concat_columns.instance()(*datasets, **kwargs)
        raise ValueError("concat: axis must be 0 or 1.")

    def apply_schema(self, *states: DatasetState, **kwargs: Any) -> Schema:
        raise NotImplementedError("concat dispatches in __call__.")

    def apply_table(self, *states: DatasetState, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("concat dispatches in __call__.")

    def apply_couplings(self, *states: DatasetState, **kwargs: Any) -> CouplingSet:
        raise NotImplementedError("concat dispatches in __call__.")


def _field_name_order(states: tuple[DatasetState, ...]) -> tuple[str, ...]:
    names = []
    for state in states:
        for name in state.schema.names():
            if name not in names:
                names.append(name)
    return tuple(names)


def _table_for_output_schema(state: DatasetState, schema: Schema) -> pd.DataFrame:
    table = state.table.copy()
    for field in state.schema:
        if not isinstance(field, IndexField):
            continue
        if not schema.has(field.name):
            continue
        if isinstance(schema.get(field.name), IndexField):
            continue
        table[field.name] = pd.Series(state.table.index, index=state.table.index)
    return table


def _require_states(states: tuple[DatasetState, ...], op_name: str) -> None:
    if not states:
        raise ValueError(f"{op_name} requires at least one dataset.")


def _row_stack_index_identity(states: tuple[DatasetState, ...]):
    identities = {
        identity
        for identity in (_maybe_primary_index_identity(state) for state in states)
        if identity is not None
    }
    if not identities:
        return None
    if len(identities) == 1:
        return next(iter(identities))
    return new_index_identity()


def _aligned_index_identity(states: tuple[DatasetState, ...]):
    identities = {
        identity
        for identity in (_maybe_primary_index_identity(state) for state in states)
        if identity is not None
    }
    if not identities:
        return None
    if len(identities) == 1:
        return next(iter(identities))
    return new_index_identity()


def _maybe_primary_index_identity(state: DatasetState):
    try:
        return primary_index_identity(state)
    except ValueError:
        return None


def _with_primary_identity(schema: Schema, identity):
    if identity is None:
        return schema
    try:
        return with_primary_index_identity(schema, identity)
    except ValueError:
        return schema
