"""patchframe.ops.builtin.bind_materialize"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from patchframe.dataset.context import FieldHandle, resolve_field_name
from patchframe.dataset.couplings import FieldRef, Materialize
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import (
    Cardinality,
    SchemaTransition,
    TableTransition,
    TransitionPlan,
)


class bind_materialize(DatasetOperator):
    """Attach Materialize couplings to one or more data fields."""

    transitions = TransitionPlan(
        schema=SchemaTransition.preserve(),
        table=TableTransition.preserve(),
    )
    cardinality = Cardinality.PRESERVE

    def new_couplings(
        self,
        state: DatasetState,
        field: str | FieldHandle | Iterable[str | FieldHandle],
        **_: Any,
    ) -> tuple[Materialize, ...]:
        fields = (field,) if isinstance(field, (str, FieldHandle)) else tuple(field)
        names = tuple(resolve_field_name(name, state.schema, op_name=self.name) for name in fields)
        return tuple(Materialize(field=FieldRef(name)) for name in names)
