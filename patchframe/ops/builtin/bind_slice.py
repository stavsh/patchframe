"""patchframe.ops.builtin.bind_slice"""

from __future__ import annotations

from typing import Any

from patchframe.dataset.couplings import BindSlice, FieldRef
from patchframe.dataset.fields import DataField, DimensionedSliceField
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import FieldInput, FieldReturn
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
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
    per_row_independent = PerRowIndependence.INDEPENDENT
    slice_field = FieldInput(field_type=DimensionedSliceField)
    data_field = FieldInput(field_type=DataField, output=True)
    returns = FieldReturn()

    def new_couplings(
        self,
        state: DatasetState,
        slice_field: str,
        data_field: str,
        **_: Any,
    ) -> tuple[BindSlice, ...]:
        return (
            BindSlice(slice_field=FieldRef(slice_field), data_field=FieldRef(data_field)),
        )
