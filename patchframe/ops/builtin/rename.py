"""patchframe.ops.builtin.rename"""

from __future__ import annotations

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


class rename(DatasetOperator):
    """Rename one or more fields.

    Renames the corresponding schema fields, table columns, the DataFrame index
    axis (when the primary IndexField is renamed, so the index name stays
    consistent with its field), and any coupling specs that reference the old
    names.

    Rewrites the schema, so it is not coupling-able: its lazy arm (a bundle
    handle operand) lifts onto a ``BundleField`` carrier.

    Usage
    -----
    rename(ds, {"old": "new"})                       # eager -> Dataset
    rename(b.field("cell"), {"old": "new"}, out="r")  # lazy  -> FieldHandle
    """

    transitions = TransitionPlan(schema=SchemaTransition.rewrite())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    dataset = DatasetInput()
    mapping = ParamInput()
    out = FieldOutput()
    returns = FieldReturn()

    def apply_schema(self, state: DatasetState, mapping: dict[str, str], **_) -> Schema:
        unknown = [k for k in mapping if not state.schema.has(k)]
        if unknown:
            raise ValueError(f"rename: fields not in schema: {unknown}")
        return state.schema.rename(mapping)

    def apply_table(self, state: DatasetState, mapping: dict[str, str], **_) -> pd.DataFrame:
        col_mapping = {k: v for k, v in mapping.items() if k in state.table.columns}
        df = state.table.rename(columns=col_mapping)
        # The primary index is the DataFrame index, not a column; rename its axis
        # so the index name stays consistent with the renamed IndexField.
        if df.index.name in mapping:
            df = df.rename_axis(index=mapping[df.index.name])
        return df
