"""patchframe.ops.builtin.bind_materialize"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from patchframe.dataset.couplings import FieldRef, Materialize
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


class bind_materialize(DatasetOperator):
    """Attach Materialize couplings to one or more data fields.

    Coupling-able (schema/table preserve, one-to-one, per-row-independent), so
    the lazy arm is same-level: a ``FieldHandle`` records the ``Materialize``
    coupling on the dataset (no bundle) and returns a chaining handle to that
    same field — its output is the field itself (in-place). Plain names and a
    ``Dataset``-first call take the eager path; a sequence of names stays eager.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.preserve(),
        table=TableTransition.preserve(),
    )
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    field = FieldInput(output=True)
    returns = FieldReturn()

    def new_couplings(
        self,
        state: DatasetState,
        field: str | Iterable[str],
        **_: Any,
    ) -> tuple[Materialize, ...]:
        names = (field,) if isinstance(field, str) else tuple(field)
        return tuple(Materialize(field=FieldRef(name)) for name in names)
