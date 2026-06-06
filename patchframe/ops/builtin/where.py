"""patchframe.ops.builtin.where"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from patchframe.dataset.fields import BundleField
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import (
    DatasetInput,
    FieldOutput,
    FieldReturn,
    ParamInput,
)
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class where(DatasetOperator):
    """Filter rows by a predicate.

    Schema, couplings, and sources are preserved. Only the table changes.
    ``where`` is per-row-independent but cardinality-changing (``FILTER``), so it
    is *not* coupling-able: its lazy arm lifts onto a ``BundleField`` carrier.

    Usage
    -----
    where(ds, ds.table["col"] == "val")       # boolean Series, eager -> Dataset
    where(ds, lambda df: df["col"] == "val")  # callable, eager -> Dataset
    where(b.field("cell"), pred, out="kept")  # bundle handle, lazy -> FieldHandle
    """

    transitions = TransitionPlan(schema=SchemaTransition.preserve())
    cardinality = Cardinality.FILTER
    per_row_independent = PerRowIndependence.INDEPENDENT
    dataset = DatasetInput()
    predicate = ParamInput()
    out = FieldOutput(field_type=BundleField)
    returns = FieldReturn()

    def apply_table(
        self,
        state: DatasetState,
        predicate: pd.Series | Callable[[pd.DataFrame], pd.Series],
        **_,
    ) -> pd.DataFrame:
        mask = predicate(state.table) if callable(predicate) else predicate
        return state.table.loc[mask]
