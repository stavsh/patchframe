"""Assign one or more table-backed fields to a dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import (
    CompositionContext,
    compose_column,
    normalize_column,
)
from patchframe.dataset.fields import Field, ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, DatasetOperator
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


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
    per_row_independent = PerRowIndependence.INDEPENDENT

    def __call__(
        self,
        target: Any = MISSING,
        values: Any = MISSING,
        /,
        *,
        couplings: tuple[Coupling, ...] = (),
        **columns: Any,
    ) -> Any:
        # Two call shapes (an @overload-style split on the first operand):
        #   assign(dataset, a=..., b=...)   eager: a Dataset + column kwargs.
        #   assign([h_a, h_b], values)      handle: a selection of target field
        #     handles (typically from ds.new_field(...)) + a frame/mapping of
        #     values keyed by field name. ``target``/``values`` are positional-only
        #     so "target"/"values" stay usable as column names in the eager form.
        if _is_field_targets(target):
            return self._assign_to_fields(target, values, couplings)
        return self._dispatch(target, columns=columns, couplings=couplings)

    def _assign_to_fields(
        self,
        target: Any,
        values: Any,
        couplings: tuple[Coupling, ...],
    ) -> Any:
        from patchframe.dataset.context import FieldSelection

        if values is MISSING:
            raise TypeError(
                "assign(targets, values): the handle form needs a values frame or "
                "mapping keyed by field name."
            )
        selection = (
            target if isinstance(target, FieldSelection) else FieldSelection(tuple(target))
        )
        context = selection.dataset_context
        if context is None:
            raise ValueError("assign: empty target selection.")
        names = selection.names()
        columns = {name: values[name] for name in names}
        # Fill on the cursor's snapshot (no context effects), advance the shared
        # cursor, and hand back the selection of filled fields.
        result = self._apply(context.dataset, columns=columns, couplings=tuple(couplings))
        context.adopt(result)
        return FieldSelection(tuple(context.field(name) for name in names))

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


def _is_field_targets(target: Any) -> bool:
    """Whether ``target`` is the handle form: a selection / list of field handles."""

    from patchframe.dataset.context import FieldHandle, FieldSelection

    if isinstance(target, FieldSelection):
        return True
    return (
        isinstance(target, (list, tuple))
        and len(target) > 0
        and all(isinstance(item, FieldHandle) for item in target)
    )


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
