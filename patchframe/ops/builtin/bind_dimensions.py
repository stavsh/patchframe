"""patchframe.ops.builtin.bind_dimensions"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import BindDimensions, CouplingSet
from patchframe.dataset.fields import DimensionedSliceField, DimensionField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import AspectTransition, TransitionPlan


class bind_dimensions(DatasetOperator):
    """Add a DimensionedSliceField column and a BindDimensions coupling in one call.

    Convenience wrapper around ``add_column`` + ``BindDimensions``. The new
    slice field is initialised to ``None`` for every row; ``consume(ds,
    slice_field)`` (or row access) materialises the slices from the referenced
    DimensionField columns.

    Calling ``bind_dimensions`` again on an already-existing ``DimensionedSliceField``
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
    bind_dimensions(ds, slice_field="clip", bindings={"x": ("start", "end")})
    bind_dimensions(ds, slice_field="clip", bindings=(("x0", "x1"), ("y0", "y1")))
    # Chain for separate x and y passes:
    ds = bind_dimensions(ds, slice_field="clip", bindings={"x": ("x0", "x1")})
    ds = bind_dimensions(ds, slice_field="clip", bindings={"y": ("y0", "y1")})
    """

    transitions = TransitionPlan(
        schema=AspectTransition("derive"),
        table=AspectTransition("derive"),
        couplings=AspectTransition("derive"),
    )

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
                    raise ValueError(f"bind_dimensions: binding field {ref.name!r} not in schema.")
                if not isinstance(state.schema.get(ref.name), DimensionField):
                    raise TypeError(f"bind_dimensions: field {ref.name!r} is not a DimensionField.")

        if state.schema.has(slice_field):
            existing = state.schema.get(slice_field)
            if not isinstance(existing, DimensionedSliceField):
                raise TypeError(
                    f"bind_dimensions: field {slice_field!r} already exists as "
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

    def apply_couplings(
        self,
        state: DatasetState,
        slice_field: str,
        bindings: Any,
        **_: Any,
    ) -> CouplingSet:
        new_coupling = BindDimensions(slice_field=slice_field, bindings=bindings)
        if new_coupling in state.couplings.couplings:
            return state.couplings
        return state.couplings.add(new_coupling)
