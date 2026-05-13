"""patchframe.ops.builtin.merge"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    compose_column,
)
from patchframe.dataset.fields import Field, IndexField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator
from patchframe.ops.builtin._composition import (
    compose_collision_field,
    normalize_collision,
    normalize_table_to_schema,
    resolve_collision_column,
    union_couplings,
)

_LEFT_INDEX = "left_index"
_RIGHT_INDEX = "right_index"
_JOIN_MAPPING_NAMES = (_LEFT_INDEX, _RIGHT_INDEX)


class merge(CompositionOperator):
    """Materialize two datasets according to an explicit join-plan dataset."""

    def apply_schema(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> Schema:
        left, right, join_plan = _require_merge_states(states, self.name)
        _validate_join_plan(join_plan, self.name)
        _validate_reserved_input_names(left, "left", self.name)
        _validate_reserved_input_names(right, "right", self.name)

        strategy = normalize_collision(collision)
        output_fields = list(join_plan.schema.fields)
        positions = {field.name: index for index, field in enumerate(output_fields)}

        for state in (left, right):
            for field in _table_fields(state):
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
                output_fields[index] = compose_collision_field(
                    output_fields[index],
                    field,
                    tuple(f for i, f in enumerate(output_fields) if i != index),
                    strategy,
                    self.name,
                )

        return Schema(fields=tuple(output_fields))

    def apply_table(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        left, right, join_plan = _require_merge_states(states, self.name)
        _validate_join_plan(join_plan, self.name)
        _validate_mapping_labels(left, join_plan.table[_LEFT_INDEX], "left", self.name)
        _validate_mapping_labels(right, join_plan.table[_RIGHT_INDEX], "right", self.name)

        strategy = normalize_collision(collision)
        schema = self.apply_schema(*states, collision=strategy)
        result = join_plan.table.copy()

        for incoming in (
            _gather_table(left, join_plan.table[_LEFT_INDEX], join_plan.table.index, self.name),
            _gather_table(right, join_plan.table[_RIGHT_INDEX], join_plan.table.index, self.name),
        ):
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

        column_names = tuple(field.name for field in schema if field.logical_type != "index")
        result = result.loc[:, column_names]
        return normalize_table_to_schema(
            result,
            schema,
            CompositionContext(role="column_add", op=self.name),
        )

    def apply_couplings(self, *states: DatasetState, **_: Any) -> CouplingSet:
        _require_merge_states(states, self.name)
        return union_couplings(*(state.couplings for state in states))


def _require_merge_states(
    states: tuple[DatasetState, ...],
    op_name: str,
) -> tuple[DatasetState, DatasetState, DatasetState]:
    if len(states) != 3:
        raise ValueError(f"{op_name} requires left, right, and join-plan datasets.")
    return states[0], states[1], states[2]


def _validate_join_plan(join_plan: DatasetState, op_name: str) -> None:
    missing_fields = [name for name in _JOIN_MAPPING_NAMES if not join_plan.schema.has(name)]
    missing_columns = [name for name in _JOIN_MAPPING_NAMES if name not in join_plan.table.columns]
    missing = tuple(dict.fromkeys((*missing_fields, *missing_columns)))
    if missing:
        raise ValueError(
            f"{op_name}: join plan is missing required mapping columns: {list(missing)}"
        )

    index_fields = [
        name
        for name in _JOIN_MAPPING_NAMES
        if join_plan.schema.get(name).logical_type == "index"
    ]
    if index_fields:
        raise ValueError(
            f"{op_name}: join-plan mapping fields must be table-backed: {index_fields}"
        )


def _validate_reserved_input_names(
    state: DatasetState,
    side: str,
    op_name: str,
) -> None:
    reserved = [field.name for field in _table_fields(state) if field.name in _JOIN_MAPPING_NAMES]
    if reserved:
        raise ValueError(
            f"{op_name}: {side} dataset uses reserved join-plan mapping names: {reserved}"
        )


def _validate_mapping_labels(
    state: DatasetState,
    labels: pd.Series,
    side: str,
    op_name: str,
) -> None:
    index_values = set(state.table.index)
    missing = [
        label
        for label in labels
        if not _is_null_label(label) and label not in index_values
    ]
    if missing:
        raise ValueError(
            f"{op_name}: join plan references labels missing from {side} dataset: {missing}"
        )


def _gather_table(
    state: DatasetState,
    labels: pd.Series,
    output_index: pd.Index,
    op_name: str,
) -> pd.DataFrame:
    columns = [field.name for field in _table_fields(state)]
    if not columns:
        return pd.DataFrame(index=output_index)

    missing_columns = [name for name in columns if name not in state.table.columns]
    if missing_columns:
        raise ValueError(f"{op_name}: input table is missing schema columns: {missing_columns}")

    reindex_labels = [pd.NA if _is_null_label(label) else label for label in labels]
    result = state.table.reindex(reindex_labels).loc[:, columns]
    result.index = output_index
    return result


def _table_fields(state: DatasetState) -> tuple[Field, ...]:
    return tuple(field for field in state.schema if not isinstance(field, IndexField))


def _is_null_label(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    missing = pd.isna(value)
    return isinstance(missing, bool) and missing
