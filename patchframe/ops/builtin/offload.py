"""patchframe.ops.builtin.offload

``offload`` — persist a dataset into a ``DatasetStore`` and return a streamable
**chunk bundle** of lazy ``DatasetAccessor``s (docs/design/dataset-accessor.md §5).

``offload(ds, store=…, chunk_size=k)`` registers ``ds``'s content in ``store``
(the memory backend now; a disk backend behind the same ``put`` later), then
constructs a tall bundle — one row per positional chunk, a ``BundleField`` column
of ``DatasetAccessor``s carrying ``IndexDimension`` row-ranges. The original
``ds`` facade can be released; the store holds the one state (the §5 *replace*,
not duplicate). ``flatten(offload(ds, store))`` round-trips ``ds``;
``offload(ds, store).rows()`` streams chunks, reading only the touched ones via
``read_partial(index)``.

Eager-only (``DatasetReturn``): offloading is a persist act, so a ``FieldHandle``
operand is rejected rather than bundle-lifted. Chunking is folded in via
``chunk_size`` (``None`` = the whole dataset as one fiber); the ``chunk`` /
``FiberSpec`` split is the later generalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from patchframe.data.dataset_accessor import DatasetAccessor
from patchframe.data.dataset_source import ROW_DIMENSION
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import BundleField, IndexField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, Operator, OperatorCall
from patchframe.ops.signature import DatasetInput, DatasetReturn, ParamInput
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)

if TYPE_CHECKING:
    from patchframe.storage.dataset_store import DatasetStore

#: Default name of the produced ``BundleField`` column of chunk fibers.
DEFAULT_FIBER_FIELD = "fiber"

#: Name of the bundle's base index (the chunk ordinal).
CHUNK_INDEX = "chunk"


def _chunk_ranges(n: int, chunk_size: int | None) -> list[tuple[int, int]]:
    """Positional ``[start, stop)`` row-ranges. ``None`` / ``>= n`` → one whole chunk."""
    if n == 0:
        return [(0, 0)]
    size = n if (chunk_size is None or chunk_size >= n) else chunk_size
    return [(start, min(start + size, n)) for start in range(0, n, size)]


class offload(Operator):
    """Persist a dataset into a store and return a lazy chunk bundle (eager op)."""

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.clear(),
        index_identity=IndexIdentityTransition.mint(),
    )
    cardinality = Cardinality.UNKNOWN
    per_row_independent = PerRowIndependence.DEPENDENT
    advances_dataset_context = False
    ds = DatasetInput()
    store = ParamInput()
    chunk_size = ParamInput(default=None)
    into = ParamInput(default=DEFAULT_FIBER_FIELD)
    returns = DatasetReturn()

    def __call__(
        self,
        ds: Dataset | Any = MISSING,
        *,
        store: "DatasetStore | None" = None,
        chunk_size: int | None = None,
        into: str = DEFAULT_FIBER_FIELD,
    ) -> Dataset:
        return Operator.__call__(self, ds, store=store, chunk_size=chunk_size, into=into)

    def normalize_call(
        self,
        ds: Dataset | Any = MISSING,
        *,
        store: "DatasetStore | None" = None,
        chunk_size: int | None = None,
        into: str = DEFAULT_FIBER_FIELD,
    ) -> OperatorCall:
        self._assert_field_handles_allowed(store, chunk_size, into)
        if not isinstance(ds, Dataset):
            raise TypeError(f"{self.name} requires a Dataset operand.")
        if store is None or not (hasattr(store, "put") and hasattr(store, "manager")):
            raise TypeError(
                f"{self.name} requires a `store` with put()/manager (a DatasetStore)."
            )
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError(f"{self.name}: chunk_size must be a positive int or None.")
        if not isinstance(into, str) or not into:
            raise ValueError(f"{self.name}: into must be a non-empty field name.")
        if into == CHUNK_INDEX:
            raise ValueError(f"{self.name}: into {into!r} collides with the chunk index name.")
        return OperatorCall(
            operator=self,
            datasets=(ds,),
            kwargs={"store": store, "chunk_size": chunk_size, "into": into},
        )

    def resolve_call_transitions(self, call: OperatorCall) -> TransitionPlan:
        # The plan is fully static (construct/clear/mint); run() builds the bundle
        # directly, so there is nothing per-call to resolve.
        return self.transitions

    def run(self, call: OperatorCall, _: TransitionPlan) -> Dataset:
        ds: Dataset = call.datasets[0]
        store = call.kwargs["store"]
        chunk_size: int | None = call.kwargs["chunk_size"]
        into: str = call.kwargs["into"]

        desc_id = store.put(ds)
        manager = store.manager
        ranges = _chunk_ranges(len(ds.table), chunk_size)
        accessors = [
            DatasetAccessor(
                source_desc_id=desc_id,
                dimensioned_slice=DimensionedSlice(dims={ROW_DIMENSION: slice(start, stop)}),
                manager_hint=manager,
            )
            for start, stop in ranges
        ]
        base_index = pd.Index(range(len(ranges)), name=CHUNK_INDEX)
        base_table = pd.DataFrame(
            {into: pd.Series(accessors, index=base_index, dtype=object)},
            index=base_index,
        )
        base_schema = Schema(fields=(IndexField(name=CHUNK_INDEX), BundleField(name=into)))
        return Dataset(
            state=DatasetState(schema=base_schema, table=base_table),
            source_manager=manager,
        )

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)
