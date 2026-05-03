"""patchframe.ops.builtin.add_column"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling, CouplingSet
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
        return state.schema.add(field_def)

    def apply_table(
        self,
        state: DatasetState,
        field_def: Field,
        values: Any,
        *,
        couplings: tuple[Coupling, ...] = (),
        **_: Any,
    ) -> pd.DataFrame:
        df = state.table.copy()
        if isinstance(values, pd.Series):
            df[field_def.name] = values.values
        else:
            df[field_def.name] = values
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
