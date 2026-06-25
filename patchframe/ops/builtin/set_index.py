"""patchframe.ops.builtin.set_index"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.fields import IndexField
from patchframe.dataset.identity import maybe_primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn, ParamInput
from patchframe.ops.transitions import (
    Cardinality,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class set_index(DatasetOperator):
    """Set a table column as the dataset index.

    Rewrites the schema and mints a new index identity (and uniqueness is a
    global property), so it is not coupling-able: its lazy arm lifts onto a
    ``BundleField`` carrier.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.rewrite(),
        index_identity=IndexIdentityTransition.mint(),
    )
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.DEPENDENT  # index uniqueness is global
    dataset = DatasetInput()
    field = ParamInput()
    out = FieldOutput()
    returns = FieldReturn()

    def apply_schema(
        self,
        state: DatasetState,
        field: str,
        *,
        index_name: str | None = None,
        drop: bool = True,
        **_: Any,
    ) -> Schema:
        if not state.schema.has(field):
            raise ValueError(f"{self.name}: field {field!r} is not present in the schema.")
        if len(state.schema.get(field).table_columns()) > 1:
            raise TypeError(
                f"{self.name}: {field!r} spans multiple table columns (a CompositeField "
                "is atomic); a composite index is CompositeIndexField, not set_index."
            )
        old_index = maybe_primary_index_field(state.schema)
        if old_index is not None and len(old_index.level_names()) > 1:
            raise TypeError(
                f"{self.name}: the current index is composite ({old_index.level_names()}); "
                "decompose it with reset_index before set_index."
            )
        if not drop:
            raise NotImplementedError("set_index(drop=False) is not implemented yet.")

        target_name = index_name or field
        output_fields = []
        for field_def in state.schema:
            if field_def.name == field:
                # Promoting a column to the index is a rewrite: the field's
                # lineage identity carries through the representation change.
                output_fields.append(
                    IndexField(
                        name=target_name,
                        field_identity=field_def.field_identity,
                    )
                )
            elif field_def is old_index and field_def.name != target_name:
                # The index demotes to its data column(s) — the field's own
                # knowledge (single -> one IndexColumnField; composite -> levels),
                # not an isinstance branch here.
                output_fields.extend(field_def.to_data_fields())
            else:
                output_fields.append(field_def)
        return Schema(fields=tuple(output_fields))

    def apply_table(
        self,
        state: DatasetState,
        field: str,
        *,
        index_name: str | None = None,
        drop: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        if field not in state.table.columns:
            raise ValueError(f"{self.name}: field {field!r} is not present in the table.")
        if not drop:
            raise NotImplementedError("set_index(drop=False) is not implemented yet.")

        target_name = index_name or field
        df = state.table.copy()
        current_index_name = df.index.name
        if (
            current_index_name is not None
            and current_index_name != target_name
            and current_index_name not in df.columns
        ):
            df[current_index_name] = df.index
        df.index = df[field]
        df.index.name = target_name
        df = df.drop(columns=[field])
        return df
