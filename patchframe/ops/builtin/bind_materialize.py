"""patchframe.ops.builtin.bind_materialize"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from patchframe.dataset.couplings import CouplingSet, FieldRef, Materialize
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.transitions import AspectTransition, TransitionPlan


class bind_materialize(DatasetOperator):
    """Attach Materialize couplings to one or more data fields."""

    transitions = TransitionPlan(couplings=AspectTransition("derive"))

    def apply_couplings(
        self,
        state: DatasetState,
        field: str | Iterable[str],
        **_: Any,
    ) -> CouplingSet:
        fields = (field,) if isinstance(field, str) else tuple(field)
        couplings = state.couplings
        for name in fields:
            coupling = Materialize(field=FieldRef(name))
            if coupling not in couplings.couplings:
                couplings = couplings.add(coupling)
        return couplings
