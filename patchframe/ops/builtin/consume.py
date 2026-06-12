"""patchframe.ops.builtin.consume

Bulk materialization of class-based couplings — literal consumption.

``consume(ds, target)`` runs couplings in the topo-sorted order computed by
``CouplingEngine`` and **discharges** them: a coupling is pending work, and
consuming completes it and removes it from the output state
(lazy-duality-plan.md, 2026-06-11). ``target`` is one of:

- A column name (``str``): runs the full chain producing that column plus
  all transitive upstream dependencies.
- A ``Coupling`` instance: runs that coupling and its transitive upstream
  dependencies only — no downstream chain mates. This enables partial
  consumption (e.g. apply a slice without then materializing); only the
  couplings actually run are discharged.

Each coupling's ``compute(state)`` produces a new column written back to
``coupling.output_field()``. Schema and sources are unchanged; the output
coupling set is the input set minus the consumed couplings.

Consumption vs evaluation: ``consume`` is the state-producing completion of
pending work. Row access (``ds[item_id]``, ``items()``, ``handle.loc``)
*evaluates* pending work for one row ephemerally — nothing is persisted or
discharged; read paths cannot consume.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling, CouplingSet
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import DatasetReturn, FieldInput
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class consume(DatasetOperator):
    """Run couplings to produce or update columns, discharging them.

    Consuming the same *output* twice is work-idempotent: the second consume
    finds nothing producing the column and returns the (already materialized)
    state unchanged. Discharge travels with the output state — consume is a
    pure function, so consuming the same *input snapshot* twice recomputes
    twice.

    Parameters
    ----------
    target:
        Column name (full chain + upstream) or a Coupling instance (upstream
        of that specific coupling, plus the coupling itself, no downstream).
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.preserve(),
        couplings=CouplingsTransition.construct(),
    )
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.UNKNOWN  # inherited from the couplings it runs
    target = FieldInput()
    returns = DatasetReturn()  # the materialization terminal: a handle still yields a Dataset

    def _couplings_to_run(
        self,
        state: DatasetState,
        target: str | Coupling,
    ) -> tuple[Coupling, ...]:
        from patchframe.dataset.coupling_engine import CouplingEngine

        engine = CouplingEngine(schema=state.schema, couplings=state.couplings)
        if isinstance(target, Coupling):
            return tuple(engine.couplings_up_to(target))
        to_run = engine.couplings_for_column(target)
        if not to_run and target not in state.table.columns:
            raise ValueError(f"No couplings produce column {target!r}.")
        return tuple(to_run)

    def apply_table(
        self,
        state: DatasetState,
        target: str | Coupling,
        **_: Any,
    ) -> pd.DataFrame:
        to_run = self._couplings_to_run(state, target)
        if not to_run:
            # Already materialized and nothing produces it — idempotent.
            return state.table.copy()

        table = state.table.copy()
        working_state = state
        for c in to_run:
            new_col = c.compute(working_state)
            table[c.output_field()] = new_col
            working_state = DatasetState(
                schema=state.schema,
                table=table,
                couplings=state.couplings,
                sources=state.sources,
                source_descriptors=state.source_descriptors,
                assets=state.assets,
                views=state.views,
                metadata=state.metadata,
            )
        return table

    def apply_couplings(
        self,
        state: DatasetState,
        target: str | Coupling,
        **_: Any,
    ) -> CouplingSet:
        # Discharge: consuming completes pending work and removes it (identity
        # comparison — the engine hands back the state's own coupling objects).
        to_run = self._couplings_to_run(state, target)
        if not to_run:
            return state.couplings
        return CouplingSet(
            tuple(
                coupling
                for coupling in state.couplings.couplings
                if not any(coupling is ran for ran in to_run)
            )
        )
