"""patchframe.ops.builtin.keep"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn, ParamInput
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class keep(DatasetOperator):
    """Retain only the listed fields, dropping everything else.

    Complement of ``drop``. Schema and table are narrowed to the listed fields.
    Couplings that reference any removed field (as input or output) are pruned.

    Narrows the schema, so it is not coupling-able: its lazy arm lifts onto a
    ``BundleField`` carrier.

    Usage
    -----
    keep(ds, ["item_id", "data", "clip"])             # eager -> Dataset
    keep(b.field("cell"), ["item_id"], out="kept")     # lazy  -> FieldHandle
    """

    transitions = TransitionPlan(schema=SchemaTransition.narrow())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    dataset = DatasetInput()
    fields = ParamInput()
    out = FieldOutput()
    returns = FieldReturn()

    def apply_schema(self, state: DatasetState, fields: list[str], **_: Any) -> Schema:
        unknown = [f for f in fields if not state.schema.has(f)]
        if unknown:
            raise ValueError(f"keep: fields not in schema: {unknown}")
        kept_set = set(fields)
        return Schema(fields=tuple(f for f in state.schema.fields if f.name in kept_set))

    def apply_table(self, state: DatasetState, fields: list[str], **_: Any) -> pd.DataFrame:
        # Atomic: keeping a field keeps all of its table columns (a CompositeField
        # spans its dotted columns; an index field has none — preserved by pandas).
        kept_cols: set[str] = set()
        for name in fields:
            kept_cols.update(state.schema.get(name).table_columns())
        keep_cols = [c for c in state.table.columns if c in kept_cols]
        return state.table[keep_cols] if keep_cols else state.table[[]]
