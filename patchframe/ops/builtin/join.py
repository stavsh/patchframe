"""patchframe.ops.builtin.join"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pandas as pd

from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.field_composition import CompositionContext, compose_key
from patchframe.dataset.fields import DimensionedSliceField, ForeignIndexField, IndexField
from patchframe.dataset.identity import primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator, OperatorCall
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn
from patchframe.ops.builtin._composition import normalize_field_names
from patchframe.ops.transitions import (
    CouplingsTransition,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)

JoinHow = Literal["inner", "left", "right", "outer"]
_JOIN_HOWS: set[str] = {"inner", "left", "right", "outer"}


@dataclass(frozen=True, slots=True)
class JoinStrategy:
    """Base class for row-matching strategies used by ``join``."""

    how: JoinHow = "inner"

    def __post_init__(self) -> None:
        if self.how not in _JOIN_HOWS:
            raise ValueError(f"join: how must be one of {sorted(_JOIN_HOWS)}.")


@dataclass(frozen=True, slots=True)
class IndexJoin(JoinStrategy):
    """Join rows by dataset index label."""


@dataclass(frozen=True, slots=True)
class FieldEqualityJoin(JoinStrategy):
    """Join rows by equality over one or more table-backed fields."""

    on: str | Iterable[str] = ()

    def __post_init__(self) -> None:
        JoinStrategy.__post_init__(self)
        object.__setattr__(self, "on", normalize_field_names(self.on))


@dataclass(frozen=True, slots=True)
class DimensionJoin(JoinStrategy):
    """Join bounded dimensional slices by interval overlap.

    ``left_field`` and ``right_field`` name ``DimensionedSliceField`` columns.
    ``dimensions`` selects the slice dimensions that must all overlap. ``on``
    optionally scopes candidate pairs by same-name equality fields before
    interval matching, which is useful for tile-local pixel coordinates.
    """

    left_field: str = ""
    right_field: str = ""
    dimensions: str | Iterable[str] = ()
    on: str | Iterable[str] = ()
    predicates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        JoinStrategy.__post_init__(self)
        object.__setattr__(self, "dimensions", normalize_field_names(self.dimensions))
        object.__setattr__(self, "on", normalize_field_names(self.on))


class join(CompositionOperator):
    """Build a join-plan dataset mapping rows from two input datasets.

    Constructs a new plan schema, so it is not coupling-able: its lazy arm (two
    bundle handle operands) lifts onto a ``BundleField`` carrier and defers the
    join planning to ``collect`` time.
    """

    advances_dataset_context = False
    per_row_independent = PerRowIndependence.DEPENDENT
    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.compose(),
        index_identity=IndexIdentityTransition.mint(),
    )
    operands = DatasetInput(variadic=True)
    out = FieldOutput()
    returns = FieldReturn()

    def __call__(
        self,
        *datasets,
        strategy: JoinStrategy | None = None,
        on: str | Iterable[str] | None = None,
        how: JoinHow | None = None,
        out: str | None = None,
    ):
        # ``out`` flows through to the interpreter (Operator.__call__): with
        # bundle handle operands it names the deferred result cell; the eager
        # path ignores it.
        return super().__call__(*datasets, strategy=strategy, on=on, how=how, out=out)

    def normalize_call(
        self,
        *datasets,
        strategy: JoinStrategy | None = None,
        on: str | Iterable[str] | None = None,
        how: JoinHow | None = None,
        out: str | None = None,
    ) -> OperatorCall:
        resolved = _resolve_strategy(strategy=strategy, on=on, how=how)
        return super().normalize_call(*datasets, strategy=resolved)

    def apply_schema(
        self,
        *states: DatasetState,
        strategy: JoinStrategy,
        **_: Any,
    ) -> Schema:
        left, right = _require_binary(states, self.name)
        _validate_strategy(left, right, strategy, self.name)
        return _join_schema(left, right)

    def apply_table(
        self,
        *states: DatasetState,
        strategy: JoinStrategy,
        **_: Any,
    ) -> pd.DataFrame:
        left, right = _require_binary(states, self.name)
        _validate_strategy(left, right, strategy, self.name)
        if isinstance(strategy, IndexJoin):
            return _index_join_table(left, right, strategy)
        if isinstance(strategy, FieldEqualityJoin):
            return _field_equality_join_table(left, right, strategy)
        if isinstance(strategy, DimensionJoin):
            return _dimension_join_table(left, right, strategy)
        raise TypeError(f"Unsupported join strategy: {type(strategy).__name__}.")

    def apply_couplings(self, *states: DatasetState, **_: Any) -> CouplingSet:
        return CouplingSet()


def _resolve_strategy(
    *,
    strategy: JoinStrategy | None,
    on: str | Iterable[str] | None,
    how: JoinHow | None,
) -> JoinStrategy:
    if strategy is not None and on is not None:
        raise ValueError("join: pass either 'strategy' or 'on', not both.")
    if strategy is None:
        strategy = IndexJoin(how=how or "inner") if on is None else FieldEqualityJoin(
            how=how or "inner",
            on=on,
        )
    elif how is not None:
        strategy = replace(strategy, how=how)
    return strategy


def _join_schema(left: DatasetState, right: DatasetState) -> Schema:
    left_index = primary_index_field(left.schema)
    right_index = primary_index_field(right.schema)
    return Schema(
        fields=(
            IndexField(name="join_id"),
            ForeignIndexField(
                name="left_index",
                index_identity=left_index.identity,
            ),
            ForeignIndexField(
                name="right_index",
                index_identity=right_index.identity,
            ),
        )
    )


def _index_join_table(
    left: DatasetState,
    right: DatasetState,
    strategy: IndexJoin,
) -> pd.DataFrame:
    left_labels = list(left.table.index)
    right_labels = list(right.table.index)
    left_set = set(left_labels)
    right_set = set(right_labels)

    pairs: list[tuple[Any, Any]] = []
    if strategy.how in {"inner", "left", "outer"}:
        for label in left_labels:
            if label in right_set:
                pairs.append((label, label))
            elif strategy.how in {"left", "outer"}:
                pairs.append((label, pd.NA))

    if strategy.how == "right":
        for label in right_labels:
            pairs.append((label if label in left_set else pd.NA, label))
    elif strategy.how == "outer":
        for label in right_labels:
            if label not in left_set:
                pairs.append((pd.NA, label))

    return _join_table_from_pairs(pairs)


def _field_equality_join_table(
    left: DatasetState,
    right: DatasetState,
    strategy: FieldEqualityJoin,
) -> pd.DataFrame:
    left_index_col, right_index_col = _reserved_index_columns(left, right)

    left_keys = left.table.loc[:, list(strategy.on)].copy()
    left_keys[left_index_col] = pd.Series(
        list(left.table.index),
        index=left.table.index,
        dtype=object,
    )

    right_keys = right.table.loc[:, list(strategy.on)].copy()
    right_keys[right_index_col] = pd.Series(
        list(right.table.index),
        index=right.table.index,
        dtype=object,
    )

    matched = left_keys.merge(
        right_keys,
        on=list(strategy.on),
        how=strategy.how,
    )
    return _join_table_from_pairs(
        zip(
            _null_to_na(matched[left_index_col]),
            _null_to_na(matched[right_index_col]),
            strict=True,
        )
    )


def _dimension_join_table(
    left: DatasetState,
    right: DatasetState,
    strategy: DimensionJoin,
) -> pd.DataFrame:
    right_by_scope: dict[tuple[Any, ...], list[tuple[Any, DimensionedSlice]]] = {}
    for label, row in right.table.iterrows():
        scope = _scope_key(row, strategy.on)
        right_by_scope.setdefault(scope, []).append((label, row[strategy.right_field]))

    matches_by_left: dict[Any, list[Any]] = {}
    matches_by_right: dict[Any, list[Any]] = {}
    for left_label, row in left.table.iterrows():
        left_slice = row[strategy.left_field]
        for right_label, right_slice in right_by_scope.get(_scope_key(row, strategy.on), []):
            if not _slices_overlap(left_slice, right_slice, strategy.dimensions):
                continue
            matches_by_left.setdefault(left_label, []).append(right_label)
            matches_by_right.setdefault(right_label, []).append(left_label)

    pairs: list[tuple[Any, Any]] = []
    if strategy.how in {"inner", "left", "outer"}:
        for left_label in left.table.index:
            matches = matches_by_left.get(left_label, ())
            if matches:
                pairs.extend((left_label, right_label) for right_label in matches)
            elif strategy.how in {"left", "outer"}:
                pairs.append((left_label, pd.NA))

    if strategy.how == "right":
        for right_label in right.table.index:
            matches = matches_by_right.get(right_label, ())
            if matches:
                pairs.extend((left_label, right_label) for left_label in matches)
            else:
                pairs.append((pd.NA, right_label))
    elif strategy.how == "outer":
        for right_label in right.table.index:
            if right_label not in matches_by_right:
                pairs.append((pd.NA, right_label))

    return _join_table_from_pairs(pairs)


def _join_table_from_pairs(pairs: Iterable[tuple[Any, Any]]) -> pd.DataFrame:
    pair_list = list(pairs)
    return pd.DataFrame(
        {
            "left_index": pd.Series(
                [left for left, _ in pair_list],
                dtype=object,
            ),
            "right_index": pd.Series(
                [right for _, right in pair_list],
                dtype=object,
            ),
        },
        index=pd.RangeIndex(len(pair_list), name="join_id"),
    )


def _null_to_na(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), pd.NA)


def _reserved_index_columns(left: DatasetState, right: DatasetState) -> tuple[str, str]:
    existing = set(left.table.columns).union(right.table.columns)
    left_name = _available_name("__patchframe_left_index", existing)
    existing.add(left_name)
    right_name = _available_name("__patchframe_right_index", existing)
    return left_name, right_name


def _available_name(base: str, existing: set[str]) -> str:
    name = base
    suffix = 1
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _validate_strategy(
    left: DatasetState,
    right: DatasetState,
    strategy: JoinStrategy,
    op_name: str,
) -> None:
    if isinstance(strategy, IndexJoin):
        return
    if isinstance(strategy, FieldEqualityJoin):
        _validate_field_equality_strategy(left, right, strategy, op_name)
        return
    if isinstance(strategy, DimensionJoin):
        _validate_dimension_strategy(left, right, strategy, op_name)
        return
    raise TypeError(f"Unsupported join strategy: {type(strategy).__name__}.")


def _validate_field_equality_strategy(
    left: DatasetState,
    right: DatasetState,
    strategy: FieldEqualityJoin,
    op_name: str,
) -> None:
    if not strategy.on:
        raise ValueError(f"{op_name}: FieldEqualityJoin requires at least one field.")

    missing = [
        name
        for name in strategy.on
        if not left.schema.has(name) or not right.schema.has(name)
    ]
    if missing:
        raise ValueError(f"{op_name}: join fields are not present in both schemas: {missing}")

    non_columns = [
        name
        for name in strategy.on
        if name not in left.table.columns or name not in right.table.columns
    ]
    if non_columns:
        raise ValueError(f"{op_name}: join fields must be table columns: {non_columns}")

    for name in strategy.on:
        compose_key(
            (left.schema.get(name), right.schema.get(name)),
            CompositionContext(role="key_coalesce", op=op_name),
        )


def _validate_dimension_strategy(
    left: DatasetState,
    right: DatasetState,
    strategy: DimensionJoin,
    op_name: str,
) -> None:
    if not strategy.left_field or not strategy.right_field:
        raise ValueError(f"{op_name}: DimensionJoin requires left_field and right_field.")
    if not strategy.dimensions:
        raise ValueError(f"{op_name}: DimensionJoin requires at least one dimension.")

    for state, field_name, side in (
        (left, strategy.left_field, "left"),
        (right, strategy.right_field, "right"),
    ):
        if not state.schema.has(field_name) or field_name not in state.table.columns:
            raise ValueError(
                f"{op_name}: DimensionJoin {side} field {field_name!r} is not present."
            )
        if not isinstance(state.schema.get(field_name), DimensionedSliceField):
            raise TypeError(
                f"{op_name}: DimensionJoin {side} field {field_name!r} "
                "must be a DimensionedSliceField."
            )
        for value in state.table[field_name]:
            _validate_dimension_slice(value, strategy.dimensions, op_name=op_name)

    if strategy.on:
        _validate_shared_key_fields(left, right, strategy.on, op_name)
        for state, side in ((left, "left"), (right, "right")):
            null_fields = [name for name in strategy.on if state.table[name].isna().any()]
            if null_fields:
                raise ValueError(
                    f"{op_name}: DimensionJoin {side} scope fields contain nulls: "
                    f"{null_fields}"
                )

    unsupported_dimensions = set(strategy.predicates) - set(strategy.dimensions)
    if unsupported_dimensions:
        raise ValueError(
            f"{op_name}: DimensionJoin predicates reference dimensions not selected "
            f"for matching: {sorted(unsupported_dimensions)}"
        )
    unsupported_predicates = {
        name: predicate
        for name, predicate in strategy.predicates.items()
        if predicate != "overlap"
    }
    if unsupported_predicates:
        raise ValueError(
            f"{op_name}: DimensionJoin only supports the 'overlap' predicate; "
            f"got {unsupported_predicates}."
        )


def _validate_shared_key_fields(
    left: DatasetState,
    right: DatasetState,
    names: tuple[str, ...],
    op_name: str,
) -> None:
    missing = [
        name
        for name in names
        if not left.schema.has(name) or not right.schema.has(name)
    ]
    if missing:
        raise ValueError(f"{op_name}: join fields are not present in both schemas: {missing}")

    non_columns = [
        name
        for name in names
        if name not in left.table.columns or name not in right.table.columns
    ]
    if non_columns:
        raise ValueError(f"{op_name}: join fields must be table columns: {non_columns}")

    for name in names:
        compose_key(
            (left.schema.get(name), right.schema.get(name)),
            CompositionContext(role="key_coalesce", op=op_name),
        )


def _scope_key(row: pd.Series, names: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[name] for name in names)


def _validate_dimension_slice(
    value: Any,
    dimensions: tuple[str, ...],
    *,
    op_name: str,
) -> None:
    if not isinstance(value, DimensionedSlice):
        raise TypeError(f"{op_name}: DimensionJoin requires DimensionedSlice values.")
    for name in dimensions:
        if name not in value.dims:
            raise ValueError(f"{op_name}: DimensionJoin slice is missing dimension {name!r}.")
        _bounded_interval(value.dims[name], name=name, op_name=op_name)


def _slices_overlap(
    left: DimensionedSlice,
    right: DimensionedSlice,
    dimensions: tuple[str, ...],
) -> bool:
    return all(
        _intervals_overlap(
            _bounded_interval(left.dims[name], name=name, op_name="join"),
            _bounded_interval(right.dims[name], name=name, op_name="join"),
        )
        for name in dimensions
    )


def _bounded_interval(value: Any, *, name: str, op_name: str) -> tuple[Any, Any]:
    if not isinstance(value, slice) or value.start is None or value.stop is None:
        raise ValueError(
            f"{op_name}: DimensionJoin dimension {name!r} must be a bounded slice."
        )
    if value.start >= value.stop:
        raise ValueError(
            f"{op_name}: DimensionJoin dimension {name!r} must have start < stop."
        )
    return value.start, value.stop


def _intervals_overlap(left: tuple[Any, Any], right: tuple[Any, Any]) -> bool:
    left_start, left_stop = left
    right_start, right_stop = right
    return left_start < right_stop and right_start < left_stop


def _require_binary(
    states: tuple[DatasetState, ...],
    op_name: str,
) -> tuple[DatasetState, DatasetState]:
    if len(states) != 2:
        raise ValueError(f"{op_name} requires exactly two datasets.")
    return states[0], states[1]
