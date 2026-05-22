"""patchframe.ops.builtin.drop"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import Cardinality, SchemaTransition, TransitionPlan


class drop(DatasetOperator):
    """Remove one or more fields from the dataset.

    Drops the fields from the schema and their corresponding table columns.
    Couplings that reference any dropped field (as input or output) are pruned.

    Usage
    -----
    drop(ds, ["col1", "col2"])
    drop.instance()(ds, ["col1", "col2"])
    """

    transitions = TransitionPlan(schema=SchemaTransition.narrow())
    cardinality = Cardinality.PRESERVE

    def apply_schema(self, state: DatasetState, fields: list[str], **_: Any) -> Schema:
        unknown = [f for f in fields if not state.schema.has(f)]
        if unknown:
            raise ValueError(f"drop: fields not in schema: {unknown}")
        return state.schema.drop(*fields)

    def apply_table(self, state: DatasetState, fields: list[str], **_: Any) -> pd.DataFrame:
        col_drops = [f for f in fields if f in state.table.columns]
        if not col_drops:
            return state.table
        return state.table.drop(columns=col_drops)
