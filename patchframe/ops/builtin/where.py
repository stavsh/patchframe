"""patchframe.ops.builtin.where"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import Cardinality, SchemaTransition, TransitionPlan


class where(DatasetOperator):
    """Filter rows by a predicate.

    Schema, couplings, and sources are preserved. Only the table changes.

    Usage
    -----
    where(ds, ds.table["col"] == "val")       # boolean Series
    where(ds, lambda df: df["col"] == "val")  # callable
    where.instance()(ds, mask)
    """

    transitions = TransitionPlan(schema=SchemaTransition.preserve())
    cardinality = Cardinality.FILTER

    def apply_table(
        self,
        state: DatasetState,
        predicate: pd.Series | Callable[[pd.DataFrame], pd.Series],
        **_,
    ) -> pd.DataFrame:
        mask = predicate(state.table) if callable(predicate) else predicate
        return state.table.loc[mask]
