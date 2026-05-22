"""patchframe.ops.builtin.keep"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import Cardinality, SchemaTransition, TransitionPlan


class keep(DatasetOperator):
    """Retain only the listed fields, dropping everything else.

    Complement of ``drop``. Schema and table are narrowed to the listed fields.
    Couplings that reference any removed field (as input or output) are pruned.

    Usage
    -----
    keep(ds, ["item_id", "data", "clip"])
    keep.instance()(ds, ["item_id", "data"])
    """

    transitions = TransitionPlan(schema=SchemaTransition.narrow())
    cardinality = Cardinality.PRESERVE

    def apply_schema(self, state: DatasetState, fields: list[str], **_: Any) -> Schema:
        unknown = [f for f in fields if not state.schema.has(f)]
        if unknown:
            raise ValueError(f"keep: fields not in schema: {unknown}")
        kept_set = set(fields)
        return Schema(fields=tuple(f for f in state.schema.fields if f.name in kept_set))

    def apply_table(self, state: DatasetState, fields: list[str], **_: Any) -> pd.DataFrame:
        kept_set = set(fields)
        keep_cols = [c for c in state.table.columns if c in kept_set]
        return state.table[keep_cols] if keep_cols else state.table[[]]
