"""patchframe.ops.builtin.partition"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    BundleField,
    CompositeIndexField,
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

#: Default name of the produced ``CompositeIndexField`` for a composite key. The
#: levels carry the real names (the key columns); this names the index as a whole.
DEFAULT_COMPOSITE_KEY = "key"

#: Allowed null-key policies (partition-aggregate.md §4).
_NULL_KEYS = ("error", "drop", "group")


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

    ``by`` may be a single field name (a single-level index) or a **list** of
    names — a *composite* key producing a native pandas ``MultiIndex`` base
    described by one ``CompositeIndexField`` (docs/design/composite-field.md §2).
    The level sub-fields are the key columns' own fields, so a level that was a
    ``ForeignIndexField`` keeps its reference — and after a ``reset_index`` the
    rollup ``partition(by=that_level)`` re-aligns by identity. ``domain=`` is
    single-key only for now.

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

    ``null_keys=`` is the policy for null keys (or any null component of a
    composite key): ``"error"`` (default — a null key has no group), ``"drop"``
    (drop the null-key rows before grouping). ``"group"`` (keep the unattributed
    bucket as a null-labelled group) is not yet implemented.

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
    null_keys = ParamInput(default="error")
    into = FieldOutput(field_type=BundleField)
    returns = FieldReturn()

    def __call__(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Iterable[str] | Any = None,
        domain: Dataset | None = None,
        *,
        into: str = DEFAULT_FIBER_FIELD,
        null_keys: str = "error",
    ) -> Dataset:
        return Operator.__call__(self, ds, by, domain=domain, into=into, null_keys=null_keys)

    def normalize_call(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Iterable[str] | Any = None,
        domain: Dataset | None = None,
        *,
        into: str = DEFAULT_FIBER_FIELD,
        null_keys: str = "error",
    ) -> OperatorCall:
        self._assert_field_handles_allowed(by, domain, {"into": into})
        if not isinstance(ds, Dataset):
            raise TypeError(f"{self.name} requires a Dataset operand.")
        by = self._normalize_by(by)
        if domain is not None and not isinstance(domain, Dataset):
            raise TypeError(f"{self.name}: `domain` must be a Dataset.")
        if not isinstance(into, str) or not into:
            raise ValueError(f"{self.name}: `into` must be a non-empty field name.")
        if null_keys not in _NULL_KEYS:
            raise ValueError(f"{self.name}: null_keys must be one of {_NULL_KEYS}, got {null_keys!r}.")
        datasets = (ds,) if domain is None else (ds, domain)
        return OperatorCall(
            operator=self,
            datasets=datasets,
            kwargs={"by": by, "into": into, "null_keys": null_keys},
        )

    def _normalize_by(self, by: Any) -> str | tuple[str, ...]:
        if isinstance(by, str):
            if not by:
                raise TypeError(f"{self.name} requires `by`: the name of the key field.")
            return by
        if isinstance(by, (list, tuple)) and by and all(isinstance(k, str) and k for k in by):
            return tuple(by)
        raise TypeError(
            f"{self.name} requires `by`: a field name or a non-empty list of field names."
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        ds = call.datasets[0]
        domain = call.datasets[1] if len(call.datasets) > 1 else None
        by = call.kwargs["by"]
        into: str = call.kwargs["into"]
        null_keys: str = call.kwargs["null_keys"]

        by_list = [by] if isinstance(by, str) else list(by)
        composite = len(by_list) > 1
        key_fields = [self._validate_key_field(ds, key) for key in by_list]

        table = self._apply_null_policy(ds.table, by_list, by, null_keys)
        index_field, base_index = self._resolve_base(
            by_list, key_fields, table, domain, composite
        )
        self._check_into_collision(into, index_field, by_list)

        group_key = by_list if composite else by_list[0]
        positions_by_key = table.groupby(group_key, sort=False, observed=True).indices
        empty = table.iloc[0:0]
        fibers = [
            self._fiber_dataset(
                ds,
                table.iloc[positions]
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

    def _apply_null_policy(
        self, table: pd.DataFrame, by_list: list[str], by: Any, null_keys: str
    ) -> pd.DataFrame:
        null_mask = table[by_list].isna().any(axis=1)
        n_null = int(null_mask.sum())
        if not n_null:
            return table
        if null_keys == "error":
            raise ValueError(
                f"{self.name}: key {by!r} has {n_null} null value(s); a null key "
                "has no group (partition-aggregate.md §4). Use null_keys='drop'."
            )
        if null_keys == "drop":
            return table.loc[~null_mask]
        # "group" — keep the unattributed bucket as a null-labelled group.
        raise NotImplementedError(
            f"{self.name}: null_keys='group' (keep the unattributed bucket) is not "
            "yet implemented; use 'drop' or 'error'."
        )

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
        if len(key_field.table_columns()) > 1:
            raise TypeError(
                f"{self.name}: {by!r} spans {len(key_field.table_columns())} table "
                "columns (a CompositeField is atomic); a key must be a single column."
            )
        return key_field

    def _resolve_base(
        self,
        by_list: list[str],
        key_fields: list[Any],
        table: pd.DataFrame,
        domain: Dataset | None,
        composite: bool,
    ) -> tuple[IndexField, pd.Index]:
        if composite:
            if domain is not None:
                raise TypeError(
                    f"{self.name}: domain= with a composite key is not yet supported "
                    "(single-key only)."
                )
            # Reuse the key columns' own fields as the index levels — a level that
            # was a ForeignIndexField keeps its reference. Identity is minted v1.
            index_field = CompositeIndexField(
                name=DEFAULT_COMPOSITE_KEY, sub_schema=Schema(fields=tuple(key_fields))
            )
            base_index = pd.MultiIndex.from_frame(table[by_list].drop_duplicates())
            return index_field, base_index

        key_field = key_fields[0]
        by = by_list[0]
        keys = table[by]
        if isinstance(key_field, ForeignIndexField):
            identity = key_field.target_identity
            if domain is None:
                index_field = IndexField(name=by, dtype=key_field.dtype, identity=identity)
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

    def _check_into_collision(
        self, into: str, index_field: IndexField, by_list: list[str]
    ) -> None:
        reserved = {index_field.name, *by_list}
        if into in reserved:
            raise ValueError(
                f"{self.name}: `into` {into!r} collides with the base index or its levels."
            )

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
