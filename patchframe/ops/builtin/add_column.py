"""patchframe.ops.builtin.add_column"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling, CouplingSet
from patchframe.dataset.field_composition import (
    CompositionContext,
    compose_column,
    normalize_column,
)
from patchframe.dataset.fields import Field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import AspectTransition, TransitionPlan


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

    transitions = TransitionPlan(
        schema    = AspectTransition("derive"),
        table     = AspectTransition("derive"),
        couplings = AspectTransition("derive"),
    )

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

    def apply_couplings(
        self,
        state: DatasetState,
        field_def: Field,
        values: Any,
        *,
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> CouplingSet:
        return state.couplings.add(*couplings)

    def _compose_field(self, state: DatasetState, field_def: Field) -> Field:
        return compose_column(
            field_def,
            state.schema.fields,
            CompositionContext(role="column_add", op=self.name),
        )
