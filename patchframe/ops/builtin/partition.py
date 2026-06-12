"""patchframe.ops.builtin.partition"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    BundleField,
    DataField,
    DimensionedSliceField,
    ForeignIndexField,
    IndexField,
)
from patchframe.dataset.identity import primary_index_field, primary_index_identity
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, Operator, OperatorCall
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn, ParamInput
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

#: Default name of the produced ``BundleField`` column of group sub-datasets.
DEFAULT_FIBER_FIELD = "fiber"


class partition(Operator):
    """Split one dataset into key-fibers: the tall-bundle group-by.

    ``partition(ds, by=key)`` returns an ordinary ``Dataset`` playing the tall
    bundle role (lazy-and-bundle.md §3): one row per distinct key, indexed by
    the key labels, with a single ``BundleField`` column whose cells are the
    member sub-datasets (fibers). Fibers are unmodified row-subsets of ``ds`` —
    same schema (including the key column), couplings, and sources — so lazy
    ``DataField``s inside a fiber materialize through their accessors as usual,
    and ``flatten(partition(ds, by=key))`` round-trips ``ds``'s rows and row
    identity (modulo order).

    ``partition`` takes no aggregation function — per-group computation is
    ``map_fields`` over the fiber column, where the operand-dispatch law
    governs eager-vs-deferred timing (partition-aggregate.md §2).

    ``by=`` dispatches on field type, the same split as ``join``'s ``on=``
    (partition-aggregate.md §4):

    - **``ForeignIndexField`` key — identity scope.** The base index inherits
      the referenced ``IndexIdentity``, so attaching base columns to the
      target dataset is identity-aligned composition
      (``concat_columns(target, keep(groups, [...]))``). An optional
      ``domain=`` (the target dataset) makes the base total over the target
      index — groups appear in domain order and a key with no members gets an
      *empty fiber* (zero rows, member schema preserved) — and carries the
      domain's index field (name, dtype, identity) onto the base. Foreign
      labels are validated against the domain index. Without ``domain=`` the
      base covers observed keys in first-appearance order.
    - **Plain value key — categorical.** Fresh-minted base identity, observed
      keys in first-appearance order. ``domain=`` is rejected (no identity to
      validate it against); ``link`` the key column first to opt into the
      identity arm.

    Null keys error: a null foreign label is a dangling reference, a null
    categorical key has no principled group.

    The gather lowers to the table engine's group-by internally
    (``groupby(sort=False)``), but the engine never decides semantics:
    ordering, null policy, and ``domain=`` totality are this operator's
    contract (partition-aggregate.md §5).
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.clear(),
        # Dispatches on the by-field's type: a ForeignIndexField key inherits
        # the referenced namespace; a plain value key mints. Schema-determined,
        # so the contract stays mechanically checkable (partition-aggregate.md §4).
        index_identity=IndexIdentityTransition.custom(),
    )
    cardinality = Cardinality.UNKNOWN
    # The shuffle: a group's fiber needs every input row with that key, so
    # partition is a blocking node (lazy-and-bundle.md §4) — eager-only until
    # the executor, like join/merge/set_index.
    per_row_independent = PerRowIndependence.DEPENDENT
    advances_dataset_context = False
    ds = DatasetInput()
    by = ParamInput()
    domain = DatasetInput()
    into = FieldOutput(field_type=BundleField)
    returns = FieldReturn()

    def __call__(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Any = None,
        domain: Dataset | None = None,
        *,
        into: str = DEFAULT_FIBER_FIELD,
    ) -> Dataset:
        return Operator.__call__(self, ds, by, domain=domain, into=into)

    def normalize_call(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Any = None,
        domain: Dataset | None = None,
        *,
        into: str = DEFAULT_FIBER_FIELD,
    ) -> OperatorCall:
        self._assert_field_handles_allowed(by, domain, {"into": into})
        if not isinstance(ds, Dataset):
            raise TypeError(f"{self.name} requires a Dataset operand.")
        if not isinstance(by, str) or not by:
            raise TypeError(f"{self.name} requires `by`: the name of the key field.")
        if domain is not None and not isinstance(domain, Dataset):
            raise TypeError(f"{self.name}: `domain` must be a Dataset.")
        if not isinstance(into, str) or not into:
            raise ValueError(f"{self.name}: `into` must be a non-empty field name.")
        datasets = (ds,) if domain is None else (ds, domain)
        return OperatorCall(
            operator=self,
            datasets=datasets,
            kwargs={"by": by, "into": into},
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        ds = call.datasets[0]
        domain = call.datasets[1] if len(call.datasets) > 1 else None
        by: str = call.kwargs["by"]
        into: str = call.kwargs["into"]

        key_field = self._validate_key_field(ds, by)
        keys = ds.table[by]
        null_count = int(keys.isna().sum())
        if null_count:
            raise ValueError(
                f"{self.name}: key field {by!r} has {null_count} null value(s); "
                "null keys have no group (partition-aggregate.md §4)."
            )

        index_field, base_index = self._resolve_base(key_field, keys, by, domain)
        if into == index_field.name:
            raise ValueError(
                f"{self.name}: `into` {into!r} collides with the base index name."
            )

        positions_by_key = ds.table.groupby(by, sort=False, observed=True).indices
        empty = ds.table.iloc[0:0]
        fibers = [
            self._fiber_dataset(
                ds,
                ds.table.iloc[positions]
                if (positions := positions_by_key.get(label)) is not None
                else empty,
            )
            for label in base_index
        ]

        base_table = pd.DataFrame(
            {into: pd.Series(fibers, index=base_index, dtype=object)},
            index=base_index,
        )
        base_schema = Schema(fields=(index_field, BundleField(name=into)))
        return Dataset(state=DatasetState(schema=base_schema, table=base_table))

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)

    def _validate_key_field(self, ds: Dataset, by: str):
        if not ds.schema.has(by):
            raise ValueError(f"{self.name}: key field {by!r} is not in the schema.")
        key_field = ds.schema.get(by)
        if isinstance(key_field, IndexField):
            raise TypeError(
                f"{self.name}: {by!r} is the primary index — every row is its own "
                "group; partition needs a key column."
            )
        if isinstance(key_field, (DataField, DimensionedSliceField, BundleField)):
            raise TypeError(
                f"{self.name}: {by!r} is a {type(key_field).__name__}; keys must "
                "be hashable scalar values."
            )
        return key_field

    def _resolve_base(
        self,
        key_field: Any,
        keys: pd.Series,
        by: str,
        domain: Dataset | None,
    ) -> tuple[IndexField, pd.Index]:
        if isinstance(key_field, ForeignIndexField):
            identity = key_field.target_identity
            if domain is None:
                index_field = IndexField(
                    name=by, dtype=key_field.dtype, identity=identity
                )
                return index_field, pd.Index(keys.drop_duplicates(), name=by)
            domain_identity = primary_index_identity(domain.state)
            if domain_identity != identity:
                raise ValueError(
                    f"{self.name}: `domain` index identity does not match the "
                    f"namespace referenced by {by!r}."
                )
            dangling = set(keys.unique()) - set(domain.table.index)
            if dangling:
                sample = ", ".join(sorted(repr(label) for label in dangling)[:5])
                raise ValueError(
                    f"{self.name}: {by!r} labels not present in the domain index: "
                    f"{sample}."
                )
            return primary_index_field(domain.schema), pd.Index(domain.table.index)

        if domain is not None:
            raise TypeError(
                f"{self.name}: `domain` requires a ForeignIndexField key, but "
                f"{by!r} is a {type(key_field).__name__}; link the key column to "
                "the target dataset first."
            )
        index_field = IndexField(name=by, dtype=key_field.dtype)
        return index_field, pd.Index(keys.drop_duplicates(), name=by)

    @staticmethod
    def _fiber_dataset(ds: Dataset, fiber_table: pd.DataFrame) -> Dataset:
        return Dataset(
            state=DatasetState(
                schema=ds.schema,
                table=fiber_table,
                couplings=ds.couplings,
                sources=ds.sources,
                source_descriptors=ds.state.source_descriptors,
                assets=ds.state.assets,
                views=ds.state.views,
                metadata=dict(ds.state.metadata),
            ),
            source_manager=ds.source_manager,
        )
