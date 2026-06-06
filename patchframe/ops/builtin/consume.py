"""patchframe.ops.builtin.consume

Bulk materialization of class-based couplings.

``consume(ds, target)`` runs couplings in the topo-sorted order computed by
``CouplingEngine``. ``target`` is one of:

- A column name (``str``): runs the full chain producing that column plus
  all transitive upstream dependencies.
- A ``Coupling`` instance: runs that coupling and its transitive upstream
  dependencies only — no downstream chain mates. This enables partial
  consumption (e.g. apply a slice without then materializing).

Each coupling's ``compute(state)`` produces a new column written back to
``coupling.output_field()``. Schema, couplings, and sources are unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import Coupling
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)


class consume(DatasetOperator):
    """Run couplings to produce or update columns in the dataset table.

    Parameters
    ----------
    target:
        Column name (full chain + upstream) or a Coupling instance (upstream
        of that specific coupling, plus the coupling itself, no downstream).
    """

    transitions = TransitionPlan(schema=SchemaTransition.preserve())
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.UNKNOWN  # inherited from the couplings it runs
    field_handle_inputs = ("target",)

    def apply_table(
        self,
        state: DatasetState,
        target: str | Coupling,
        **_: Any,
    ) -> pd.DataFrame:
        from patchframe.dataset.coupling_engine import CouplingEngine

        engine = CouplingEngine(schema=state.schema, couplings=state.couplings)
        if isinstance(target, Coupling):
            to_run = engine.couplings_up_to(target)
        else:
            to_run = engine.couplings_for_column(target)
            if not to_run:
                if target in state.table.columns:
                    # Already materialized and nothing produces it — idempotent.
                    return state.table.copy()
                raise ValueError(f"No couplings produce column {target!r}.")

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
