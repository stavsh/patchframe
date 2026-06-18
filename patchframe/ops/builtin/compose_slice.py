"""patchframe.ops.builtin.compose_slice"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import BindDimensions
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import DimensionedSliceField, DimensionField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import FieldOutput, FieldReturn, SelectionInput
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class compose_slice(DatasetOperator):
    """Add a DimensionedSliceField column and a BindDimensions coupling in one call.

    Convenience wrapper around ``add_column`` + ``BindDimensions``. The new
    slice field is initialised to ``None`` for every row; ``consume(ds,
    slice_field)`` (or row access) materialises the slices from the referenced
    DimensionField columns.

    Calling ``compose_slice`` again on an already-existing ``DimensionedSliceField``
    with different bindings appends another ``BindDimensions`` to the chain — useful
    for building multi-dimensional slices one dimension at a time. An identical
    (slice_field, bindings) pair is a no-op.

    Raises if ``slice_field`` exists but is not a ``DimensionedSliceField``, or if
    any binding field does not exist or is not a ``DimensionField``.

    Parameters
    ----------
    slice_field:
        Name for the DimensionedSliceField column (created if absent).
    bindings:
        Mapping of dimension name to field-name tuple(s), or an ordered sequence
        of field-name tuples — same format accepted by ``BindDimensions``.

    Usage
    -----
    compose_slice(ds, slice_field="clip", bindings={"x": ("start", "end")})
    compose_slice(ds, slice_field="clip", bindings=(("x0", "x1"), ("y0", "y1")))
    # Chain for separate x and y passes:
    ds = compose_slice(ds, slice_field="clip", bindings={"x": ("x0", "x1")})
    ds = compose_slice(ds, slice_field="clip", bindings={"y": ("y0", "y1")})
    """

    transitions = TransitionPlan(schema=SchemaTransition.extend())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    # Coupling-able with a *fresh* output: ``slice_field`` is the caller-named
    # produced field (a FieldOutput), and the handles live nested inside
    # ``bindings`` (a multi-field operand). The same-level lazy arm records the
    # BindDimensions coupling and returns a handle to ``slice_field`` — which is
    # the coupling's output_field.
    slice_field = FieldOutput(field_type=DimensionedSliceField)
    bindings = SelectionInput()
    returns = FieldReturn()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Operand-dispatch law (IO-criterion refinement, 2026-06-16): composing
        # a slice is *metadata*, not IO, so a Dataset operand computes it now —
        # record the BindDimensions coupling and consume it immediately (the
        # slice column is the product; consume discharges per the literal-consume
        # law). A FieldHandle operand records only and returns the chaining
        # handle: the explicit relationship-coupling (deferred) path. Only
        # ``materialize`` stays declare-only on both arms (its work *is* IO).
        result = DatasetOperator.__call__(self, *args, **kwargs)
        if not isinstance(result, Dataset):
            return result  # handle arm: a chaining FieldHandle (deferred path)
        context = self.resolve_dataset_context()
        if context is not None and context.dataset is result:
            # Ambient cursor (string dispatch in a `with ctx:` block): a
            # deferred-building idiom, not a Dataset operand — leave the
            # coupling pending for the terminal consume.
            return result
        # Explicit Dataset operand: compute the slice now (args[0] is the ds).
        slice_field = kwargs.get("slice_field")
        if slice_field is None and len(args) > 1:
            slice_field = args[1]
        from patchframe.ops.builtin.consume import consume

        return consume(result, slice_field)

    def apply_schema(
        self,
        state: DatasetState,
        slice_field: str,
        bindings: Any,
        **_: Any,
    ) -> Schema:
        _, norm_bindings = BindDimensions._normalize_bindings(bindings)
        for binding_refs in norm_bindings:
            for ref in binding_refs:
                if not state.schema.has(ref.name):
                    raise ValueError(f"compose_slice: binding field {ref.name!r} not in schema.")
                if not isinstance(state.schema.get(ref.name), DimensionField):
                    raise TypeError(f"compose_slice: field {ref.name!r} is not a DimensionField.")

        if state.schema.has(slice_field):
            existing = state.schema.get(slice_field)
            if not isinstance(existing, DimensionedSliceField):
                raise TypeError(
                    f"compose_slice: field {slice_field!r} already exists as "
                    f"{type(existing).__name__}, not DimensionedSliceField."
                )
            return state.schema

        return state.schema.add(DimensionedSliceField(name=slice_field))

    def apply_table(
        self,
        state: DatasetState,
        slice_field: str,
        bindings: Any,
        **_: Any,
    ) -> pd.DataFrame:
        if slice_field in state.table.columns:
            return state.table
        df = state.table.copy()
        df[slice_field] = None
        return df

    def new_couplings(
        self,
        state: DatasetState,
        slice_field: str,
        bindings: Any,
        **_: Any,
    ) -> tuple[BindDimensions, ...]:
        return (BindDimensions(slice_field=slice_field, bindings=bindings),)
