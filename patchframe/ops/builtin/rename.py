"""patchframe.ops.builtin.rename"""

from __future__ import annotations

import pandas as pd

from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import Cardinality, SchemaTransition, TransitionPlan


class rename(DatasetOperator):
    """Rename one or more fields.

    Renames the corresponding schema fields, table columns, and any coupling
    specs that reference the old names. IndexField columns (the DataFrame index)
    are renamed in the schema only — the underlying index has no column name.

    Usage
    -----
    rename(ds, {"old": "new"})
    rename.instance()(ds, {"old": "new"})
    """

    transitions = TransitionPlan(schema=SchemaTransition.rewrite())
    cardinality = Cardinality.PRESERVE

    def resolve_transitions(self, state, mapping, **_):
        return self.transitions._with(schema=SchemaTransition.rewrite(mapping=mapping))

    def apply_schema(self, state: DatasetState, mapping: dict[str, str], **_) -> Schema:
        unknown = [k for k in mapping if not state.schema.has(k)]
        if unknown:
            raise ValueError(f"rename: fields not in schema: {unknown}")
        return state.schema.rename(mapping)

    def apply_table(self, state: DatasetState, mapping: dict[str, str], **_) -> pd.DataFrame:
        col_mapping = {k: v for k, v in mapping.items() if k in state.table.columns}
        return state.table.rename(columns=col_mapping)
