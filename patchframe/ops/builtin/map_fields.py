"""patchframe.ops.builtin.map_fields"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CallSpec, MapCoupling, warn_if_unpicklable
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import ValueField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.builtin.consume import consume
from patchframe.ops.signature import FieldOutput, FieldReturn, ParamInput, SelectionInput
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class map_fields(DatasetOperator):
    """Add a value column computed per row from other fields, plus a MapCoupling.

    ``map_fields`` is a *computation*, not a *binding*, so it follows the
    operand-dispatch law (lazy-and-bundle.md §1): a ``Dataset`` operand means
    the work happens **now** — ``map_fields(ds, ["a", "b"], fn, out="c")``
    records a ``MapCoupling`` and immediately consumes it, returning a Dataset
    whose ``c`` column is filled and whose coupling is discharged (consume is
    literal: pending work, completed and removed). Deferral is opt-in via the
    handle arm, which records only — the coupling stays pending until
    consumed/collected, and row access evaluates it ephemerally.

    This deliberately diverges from the *binding* operators (``materialize``/
    ``slice_data``/``compose_slice``), which stay declare-only on both arms:
    a binding declares a structural relation that row access realizes, so the
    declaration *is* its eager work. A computation's eager work is computing.

    Coupling-able (schema ``extend``, one output row per input row,
    per-row-independent), so the lazy arm is same-level:
    ``map_fields(selection, fn, out="c")`` records the coupling *without*
    running it and returns a chaining ``FieldHandle`` to ``c``;
    ``consume(ds, "c")``, coupling-aware row access (``ds[item_id]["c"]``),
    ``handle.items()``, or ``handle.collect()`` runs it later.

    Inputs may be field names, ``FieldHandle``s, or a ``FieldSelection``; their
    per-row values are passed to ``fn`` positionally in order. ``fn`` should be a
    module-level function, not a lambda: an unpicklable ``fn`` warns at record
    time (``UnpicklableCallWarning``) because the dataset then cannot be persisted
    or dispatched to a worker while the coupling is present.

    Parameters
    ----------
    inputs:
        The fields whose per-row values are passed to ``fn`` (in order).
    fn:
        A callable ``fn(*values) -> value`` producing the output cell for a row.
    out:
        Name of the produced ``ValueField`` column.
    """

    transitions = TransitionPlan(schema=SchemaTransition.extend())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    inputs = SelectionInput()
    fn = ParamInput()
    out = FieldOutput(field_type=ValueField)
    returns = FieldReturn()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Operand-dispatch law: the interpreter routes handle operands to the
        # lazy arm (record only, return a chaining handle). A Dataset result
        # means the eager arm ran — a computation must then compute now, so
        # consume the freshly recorded coupling (which discharges it; the
        # column is the product, not the recipe).
        result = DatasetOperator.__call__(self, *args, **kwargs)
        if not isinstance(result, Dataset):
            return result
        out = kwargs.get("out")
        if out is None and len(args) > 3:
            out = args[3]
        return consume(result, out)

    @staticmethod
    def _input_names(inputs: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(inputs, str):
            return (inputs,)
        return tuple(inputs)

    def apply_schema(
        self,
        state: DatasetState,
        inputs: Any,
        fn: Callable[..., Any],
        out: str,
        **_: Any,
    ) -> Schema:
        names = self._input_names(inputs)
        if not names:
            raise ValueError("map_fields: at least one input field is required.")
        for name in names:
            if not state.schema.has(name):
                raise ValueError(f"map_fields: input field {name!r} not in schema.")
        if state.schema.has(out): #TODO: We should either allow overwriting existing fields, or allow autogenerating a fresh name. Currently user has to choose a fresh name manually, which is error-prone.
            raise ValueError(
                f"map_fields: output field {out!r} already exists; choose a fresh name."
            )
        return state.schema.add(ValueField(name=out))

    def apply_table(
        self,
        state: DatasetState,
        inputs: Any,
        fn: Callable[..., Any],
        out: str,
        **_: Any,
    ) -> pd.DataFrame:
        #TODO: table copy decision should be exposed as generic parameter, not hardcoded here. We should provide copy utilities with operator param overrides.
        df = state.table.copy() 

        df[out] = pd.Series([None] * len(df), index=df.index, dtype=object)
        return df

    def new_couplings(
        self,
        state: DatasetState,
        inputs: Any,
        fn: Callable[..., Any],
        out: str,
        **_: Any,
    ) -> tuple[MapCoupling, ...]:
        names = self._input_names(inputs)
        call = CallSpec(operator=fn)
        warn_if_unpicklable(call)
        return (MapCoupling(inputs=names, output=out, call=call),)
