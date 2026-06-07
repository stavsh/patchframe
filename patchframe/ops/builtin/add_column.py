"""patchframe.ops.builtin.add_column"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling
from patchframe.dataset.field_composition import (
    CompositionContext,
    compose_column,
    normalize_column,
)
from patchframe.dataset.fields import Field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, DatasetOperator
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class add_column(DatasetOperator):
    """Add a column and its field declaration to a dataset.

    Parameters
    ----------
    field_def:
        Field definition for the new column. ``field_def.name`` is used as the
        column name.
    values:
        Column values. A ``pd.Series`` is aligned by index; a plain sequence is
        assigned by position.
    couplings:
        Optional Coupling instances to add alongside the new column.
    """

    transitions = TransitionPlan(schema=SchemaTransition.extend())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT

    def __call__(self, target: Any = MISSING, *args: Any, couplings: tuple[Coupling, ...] = (), **kwargs: Any) -> Any:
        # Handle form: add_column(field_handle, values) fills a field created with
        # new_field (the single-field counterpart to assign's handle form) and
        # returns the handle. Eager form: add_column(dataset, field_def, values).
        from patchframe.dataset.context import FieldHandle

        if isinstance(target, FieldHandle):
            values = args[0] if args else kwargs.get("values", MISSING)
            return self._fill_field(target, values, couplings)
        return DatasetOperator.__call__(self, target, *args, couplings=couplings, **kwargs)

    def _fill_field(
        self,
        handle: Any,
        values: Any,
        couplings: tuple[Coupling, ...],
    ) -> Any:
        from patchframe.ops.builtin.assign import assign

        if values is MISSING:
            raise TypeError("add_column(handle, values): the handle form needs values.")
        context = handle.dataset_context
        name = handle.name
        # The field already exists (null-filled via new_field); filling it is
        # assign-to-existing, so reuse assign's eager fill, then advance the cursor.
        result = assign.instance()._apply(
            context.dataset, columns={name: values}, couplings=tuple(couplings)
        )
        context.adopt(result)
        return context.field(name)

    def apply_schema(
        self,
        state: DatasetState,
        field_def: Field,
        values: Any,
        *,
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> Schema:
        return state.schema.add(self._compose_field(state, field_def))

    def apply_table(
        self,
        state: DatasetState,
        field_def: Field,
        values: Any,
        *,
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> pd.DataFrame:
        field_def = self._compose_field(state, field_def)
        df = state.table.copy()
        if isinstance(values, pd.Series):
            df[field_def.name] = values.values
        else:
            df[field_def.name] = values
        df[field_def.name] = normalize_column(
            df[field_def.name],
            field_def,
            CompositionContext(role="column_add", op=self.name),
        )
        return df

    def new_couplings(
        self,
        state: DatasetState,
        field_def: Field,
        values: Any,
        *,
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> tuple[Coupling, ...]:
        return tuple(couplings)

    def _compose_field(self, state: DatasetState, field_def: Field) -> Field:
        return compose_column(
            field_def,
            state.schema.fields,
            CompositionContext(role="column_add", op=self.name),
        )
