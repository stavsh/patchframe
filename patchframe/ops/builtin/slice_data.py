"""patchframe.ops.builtin.slice_data"""

from __future__ import annotations

from typing import Any

from patchframe.dataset.couplings import BindSlice, FieldRef
from patchframe.dataset.dataset import Dataset
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


class slice_data(DatasetOperator):
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Operand-dispatch law (IO-criterion refinement, 2026-06-16): attaching
        # a slice to the data accessors is *metadata* (the accessors stay lazy —
        # the decode is ``materialize``'s job, still deferred), not IO, so a
        # Dataset operand does it now: record BindSlice and consume immediately
        # (producing sliced-but-undecoded accessors; consume discharges). A
        # FieldHandle operand records only and defers — the explicit
        # relationship-coupling path.
        result = DatasetOperator.__call__(self, *args, **kwargs)
        if not isinstance(result, Dataset):
            return result  # handle arm: a chaining FieldHandle (deferred path)
        context = self.resolve_dataset_context()
        if context is not None and context.dataset is result:
            # Ambient cursor (string dispatch in a `with ctx:` block): a
            # deferred-building idiom, not a Dataset operand — leave the
            # coupling pending for the terminal consume.
            return result
        # Explicit Dataset operand: attach the slice now (args[0] is the ds).
        data_field = kwargs.get("data_field")
        if data_field is None and len(args) > 2:
            data_field = args[2]
        from patchframe.ops.builtin.consume import consume

        return consume(result, data_field)

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
