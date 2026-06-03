"""patchframe.ops.builtin.bind_slice"""

from __future__ import annotations

from typing import Any

from patchframe.dataset.context import FieldHandle, resolve_field_name
from patchframe.dataset.couplings import BindSlice, FieldRef
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import (
    Cardinality,
    SchemaTransition,
    TableTransition,
    TransitionPlan,
)


class bind_slice(DatasetOperator):
    """Add a BindSlice coupling between a DimensionedSliceField and a DataField.

    Per-row access (``ds[item_id]``) returns a sliced ``DataAccessor`` for the
    data column. Bulk materialization runs through ``consume(ds, data_field)``.

    Parameters
    ----------
    slice_field:
        Name of the existing DimensionedSliceField (input).
    data_field:
        Name of the existing DataField (input + output, sliced in place).
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.preserve(),
        table=TableTransition.preserve(),
    )
    cardinality = Cardinality.PRESERVE

    def new_couplings(
        self,
        state: DatasetState,
        slice_field: str | FieldHandle,
        data_field: str | FieldHandle,
        **_: Any,
    ) -> tuple[BindSlice, ...]:
        slice_field = resolve_field_name(slice_field, state.schema, op_name=self.name)
        data_field = resolve_field_name(data_field, state.schema, op_name=self.name)
        return (
            BindSlice(slice_field=FieldRef(slice_field), data_field=FieldRef(data_field)),
        )
