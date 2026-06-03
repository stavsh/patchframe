"""patchframe.ops.builtin.concat"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    FieldParent,
    MergedField,
    compose_column,
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
    derive_composed_couplings,
    normalize_collision,
    normalize_table_to_schema,
    preserve_row_couplings,
)
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)


class concat_rows(CompositionOperator):
    """Stack datasets by rows."""

    transitions = TransitionPlan(
        schema=SchemaTransition.compose(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.homogeneous(),
        sources=SourcesTransition.derive(),
        index_identity=IndexIdentityTransition.coalesce(),
    )
    cardinality = Cardinality.EXPAND

    def apply_schema(self, *states: DatasetState, **_: Any) -> Schema:
        _require_states(states, self.name)
        ctx = CompositionContext(role="row_stack", op=self.name)
        output_fields: list[Field] = []
        for name in _field_name_order(states):
            parents = tuple(
                FieldParent(input_index, state.schema.get(name))
                for input_index, state in enumerate(states)
                if state.schema.has(name)
            )
            if len(parents) == 1:
                output_fields.append(
                    compose_column(parents[0].field, tuple(output_fields), ctx)
                )
            else:
                # Row unification: a MergedField with no collision strategy.
                output_fields.append(MergedField.over(parents, context=ctx))
        return _with_primary_identity(
            Schema(fields=tuple(output_fields)),
            _row_stack_index_identity(states),
        )

    def apply_table(
        self,
        *states: DatasetState,
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        _require_states(states, self.name)
        schema = (
            composed_schema
            if composed_schema is not None
            else self.apply_schema(*states)
        )
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

    cardinality = Cardinality.PRESERVE

    def apply_schema(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> Schema:
        _require_states(states, self.name)
        strategy = normalize_collision(collision)
        ctx = CompositionContext(role="column_add", op=self.name)
        output_fields: list[Field] = []
        positions: dict[str, int] = {}
        first_parent: dict[str, FieldParent] = {}
        for input_index, state in enumerate(states):
            for field in state.schema:
                if field.name not in positions:
                    positions[field.name] = len(output_fields)
                    first_parent[field.name] = FieldParent(input_index, field)
                    output_fields.append(
                        compose_column(field, tuple(output_fields), ctx)
                    )
                    continue

                # Same-name collision: replace the slot with a MergedField that
                # carries the colliding parents. It is resolved only after every
                # aspect hook has read the lineage.
                slot = positions[field.name]
                existing = output_fields[slot]
                incoming = FieldParent(input_index, field)
                parents = (
                    existing.parents + (incoming,)
                    if isinstance(existing, MergedField)
                    else (first_parent[field.name], incoming)
                )
                output_fields[slot] = MergedField.over(
                    parents, collision=strategy, context=ctx
                )
        return _with_primary_identity(
            Schema(fields=tuple(output_fields)),
            _aligned_index_identity(states),
        )

    def apply_table(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        _require_states(states, self.name)
        schema = (
            composed_schema
            if composed_schema is not None
            else self.apply_schema(*states, collision=collision)
        )
        result = pd.DataFrame(index=states[0].table.index)
        for state in states:
            incoming_table = _table_for_output_schema(state, schema)
            result, incoming = result.align(incoming_table, join="outer", axis=0)
            for name in incoming.columns:
                if name not in result.columns:
                    result[name] = incoming[name]
                    continue
                # A table-column collision implies a MergedField in the schema;
                # the MergedField owns how the colliding columns resolve.
                merged = schema.get(name)
                assert isinstance(merged, MergedField)
                result[name] = merged.resolve_column([result[name], incoming[name]])

        column_names = tuple(field.name for field in schema if field.name in result.columns)
        result = result.loc[:, column_names]
        return normalize_table_to_schema(
            result,
            schema,
            CompositionContext(role="column_add", op=self.name),
        )

    def apply_couplings(
        self,
        *states: DatasetState,
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> CouplingSet:
        return derive_composed_couplings(states, composed_schema)


class concat(CompositionOperator):
    """Dispatch to ``concat_rows`` or ``concat_columns``."""

    def __call__(self, *datasets: Dataset, axis: int = 0, **kwargs: Any) -> Dataset:
        params = {}
        dataset_context = self.resolve_param("dataset_context")
        if dataset_context is not None:
            params["dataset_context"] = dataset_context
        if axis == 0:
            return concat_rows.instance(**params)(*datasets, **kwargs)
        if axis == 1:
            return concat_columns.instance(**params)(*datasets, **kwargs)
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
