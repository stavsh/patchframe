"""Assign one or more table-backed fields to a dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling
from patchframe.dataset.field_composition import (
    CompositionContext,
    compose_column,
    normalize_column,
)
from patchframe.dataset.fields import Field, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import Cardinality, SchemaTransition, TransitionPlan


class assign(DatasetOperator):
    """Assign multiple table columns in one operation.

    Plain values infer a ``ValueField`` when the column is new. Pass
    ``(field_def, values)`` to add a specifically typed field:

        assign(
            ds,
            score=[0.1, 0.2],
            bbox=(DimensionedSliceField(name="bbox"), bboxes),
        )

    Existing columns may be assigned new values without changing their field
    definitions. Use a dedicated field-casting operator for schema role changes.
    """

    transitions = TransitionPlan(schema=SchemaTransition.extend())
    cardinality = Cardinality.PRESERVE

    def __call__(
        self,
        dataset,
        *,
        couplings: tuple[Coupling, ...] = (),
        **columns: Any,
    ):
        return self._apply(dataset, columns=columns, couplings=couplings)

    def apply_schema(
        self,
        state: DatasetState,
        *,
        columns: Mapping[str, Any],
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> Schema:
        schema = state.schema
        for name, value in columns.items():
            field_def, _ = _normalize_assignment(name, value)
            if schema.has(name):
                _validate_existing_field(schema.get(name), field_def, name=name)
                continue
            schema = schema.add(
                compose_column(
                    field_def,
                    schema.fields,
                    CompositionContext(role="column_add", op=self.name),
                )
            )
        return schema

    def apply_table(
        self,
        state: DatasetState,
        *,
        columns: Mapping[str, Any],
        couplings: tuple[Coupling, ...] = (),
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        schema = composed_schema or self.apply_schema(state, columns=columns)
        table = state.table.copy()
        for name, value in columns.items():
            _, values = _normalize_assignment(name, value)
            table[name] = values
            table[name] = normalize_column(
                table[name],
                schema.get(name),
                CompositionContext(role="column_add", op=self.name),
            )
        return table

    def new_couplings(
        self,
        state: DatasetState,
        *,
        columns: Mapping[str, Any],
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> tuple[Coupling, ...]:
        return tuple(couplings)


def _normalize_assignment(name: str, value: Any) -> tuple[Field, Any]:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], Field)
    ):
        field_def, values = value
        if field_def.name != name:
            raise ValueError(
                f"assign: field name {field_def.name!r} does not match "
                f"assigned column {name!r}."
            )
        return field_def, values
    return ValueField(name=name), value


def _validate_existing_field(existing: Field, requested: Field, *, name: str) -> None:
    if type(requested) is ValueField and requested.dtype is None:
        return
    if requested != existing:
        raise ValueError(
            f"assign: field {name!r} already exists with a different definition. "
            "Use a field-casting operator before assigning values."
        )
