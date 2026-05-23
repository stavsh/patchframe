"""patchframe.ops.builtin.merge"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    FieldParent,
    MergedField,
    compose_column,
)
from patchframe.dataset.fields import Field, ForeignIndexField, IndexField
from patchframe.dataset.identity import primary_index_identity
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator
from patchframe.ops.builtin._composition import (
    derive_composed_couplings,
    normalize_collision,
    normalize_table_to_schema,
)
from patchframe.ops.transitions import (
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)

_LEFT_INDEX = "left_index"
_RIGHT_INDEX = "right_index"
_JOIN_MAPPING_NAMES = (_LEFT_INDEX, _RIGHT_INDEX)


class merge(CompositionOperator):
    """Materialize two datasets according to an explicit join-plan dataset."""

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.union(),
        sources=SourcesTransition.union(),
        index_identity=IndexIdentityTransition.inherit(input=2),
    )

    def apply_schema(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        **_: Any,
    ) -> Schema:
        left, right, join_plan = _require_merge_states(states, self.name)
        _validate_join_plan(join_plan, self.name)
        _validate_mapping_identity(left, join_plan, _LEFT_INDEX, "left", self.name)
        _validate_mapping_identity(right, join_plan, _RIGHT_INDEX, "right", self.name)
        _validate_reserved_input_names(left, "left", self.name)
        _validate_reserved_input_names(right, "right", self.name)

        strategy = normalize_collision(collision)
        ctx = CompositionContext(role="column_add", op=self.name)
        output_fields: list[Field] = list(join_plan.schema.fields)
        positions = {field.name: index for index, field in enumerate(output_fields)}
        first_parent: dict[str, FieldParent] = {
            field.name: FieldParent(2, field) for field in output_fields
        }

        for input_index, state in ((0, left), (1, right)):
            for field in _table_fields(state):
                if field.name not in positions:
                    positions[field.name] = len(output_fields)
                    first_parent[field.name] = FieldParent(input_index, field)
                    output_fields.append(
                        compose_column(field, tuple(output_fields), ctx)
                    )
                    continue

                # Same-name collision: replace the slot with a MergedField.
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

        return Schema(fields=tuple(output_fields))

    def apply_table(
        self,
        *states: DatasetState,
        collision: ColumnCollisionStrategy | str | None = None,
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        left, right, join_plan = _require_merge_states(states, self.name)
        _validate_join_plan(join_plan, self.name)
        _validate_mapping_identity(left, join_plan, _LEFT_INDEX, "left", self.name)
        _validate_mapping_identity(right, join_plan, _RIGHT_INDEX, "right", self.name)
        _validate_mapping_labels(left, join_plan.table[_LEFT_INDEX], "left", self.name)
        _validate_mapping_labels(right, join_plan.table[_RIGHT_INDEX], "right", self.name)

        schema = (
            composed_schema
            if composed_schema is not None
            else self.apply_schema(*states, collision=collision)
        )
        result = join_plan.table.copy()

        for incoming in (
            _gather_table(left, join_plan.table[_LEFT_INDEX], join_plan.table.index, self.name),
            _gather_table(right, join_plan.table[_RIGHT_INDEX], join_plan.table.index, self.name),
        ):
            for name in incoming.columns:
                if name not in result.columns:
                    result[name] = incoming[name]
                    continue
                # A table-column collision implies a MergedField in the schema;
                # the MergedField owns how the colliding columns resolve.
                merged = schema.get(name)
                assert isinstance(merged, MergedField)
                result[name] = merged.resolve_column([result[name], incoming[name]])

        column_names = tuple(field.name for field in schema if field.logical_type != "index")
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
        _require_merge_states(states, self.name)
        return derive_composed_couplings(states, composed_schema)


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

    foreign_fields = [
        name
        for name in _JOIN_MAPPING_NAMES
        if not isinstance(join_plan.schema.get(name), ForeignIndexField)
    ]
    if foreign_fields:
        raise TypeError(
            f"{op_name}: join-plan mapping fields must be ForeignIndexField: "
            f"{foreign_fields}"
        )


def _validate_mapping_identity(
    state: DatasetState,
    join_plan: DatasetState,
    field_name: str,
    side: str,
    op_name: str,
) -> None:
    field = join_plan.schema.get(field_name)
    if not isinstance(field, ForeignIndexField):
        return
    if field.target_identity != primary_index_identity(state):
        raise ValueError(
            f"{op_name}: join plan {field_name!r} does not reference the "
            f"{side} dataset index identity."
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
    null_mask = pd.isna(labels)
    non_null_labels = labels[~null_mask]
    missing = non_null_labels[~non_null_labels.isin(state.table.index)].tolist()
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

    reindex_labels = labels.astype(object).mask(pd.isna(labels), pd.NA).to_numpy(dtype=object)
    result = state.table.reindex(reindex_labels).loc[:, columns]
    result.index = output_index
    return result


def _table_fields(state: DatasetState) -> tuple[Field, ...]:
    return tuple(field for field in state.schema if not isinstance(field, IndexField))

