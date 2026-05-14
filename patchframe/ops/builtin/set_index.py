"""patchframe.ops.builtin.set_index"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.fields import IndexColumnField, IndexField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import AspectTransition, TransitionPlan


class set_index(DatasetOperator):
    """Set a table column as the dataset index."""

    transitions = TransitionPlan(
        schema=AspectTransition("derive"),
        table=AspectTransition("derive"),
    )

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
        if not drop:
            raise NotImplementedError("set_index(drop=False) is not implemented yet.")

        target_name = index_name or field
        output_fields = []
        for field_def in state.schema:
            if field_def.name == field:
                output_fields.append(IndexField(name=target_name))
            elif field_def.primary and field_def.name != target_name:
                output_fields.append(
                    IndexColumnField(
                        name=field_def.name,
                        dtype=field_def.dtype,
                        nullable=True,
                        metadata=field_def.metadata,
                    )
                )
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

    def apply_couplings(self, state: DatasetState, *_, **__: Any) -> CouplingSet:
        return state.couplings
