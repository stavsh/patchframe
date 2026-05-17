"""patchframe.ops.builtin.join"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.field_composition import CompositionContext, compose_key
from patchframe.dataset.fields import IndexColumnField, IndexField
from patchframe.dataset.identity import primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator
from patchframe.ops.builtin._composition import normalize_field_names
from patchframe.ops.transitions import AspectTransition, TransitionPlan

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
    """Reserved strategy for dimension-aware joins."""

    dimensions: str | Iterable[str] | None = None
    predicates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        JoinStrategy.__post_init__(self)
        if self.dimensions is not None:
            object.__setattr__(self, "dimensions", normalize_field_names(self.dimensions))


class join(CompositionOperator):
    """Build a join-plan dataset mapping rows from two input datasets."""

    transitions = TransitionPlan(
        schema=AspectTransition("derive"),
        table=AspectTransition("derive"),
        couplings=AspectTransition("derive"),
        sources=AspectTransition("union"),
        index_identity=AspectTransition("mint"),
    )

    def __call__(
        self,
        *datasets,
        strategy: JoinStrategy | None = None,
        on: str | Iterable[str] | None = None,
        how: JoinHow | None = None,
    ):
        resolved = _resolve_strategy(strategy=strategy, on=on, how=how)
        return self._compose(*datasets, strategy=resolved)

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
            raise NotImplementedError("DimensionJoin requires dimension-scope join support.")
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
            IndexColumnField(
                name="left_index",
                index_identity=left_index.identity,
            ),
            IndexColumnField(
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
        raise NotImplementedError("DimensionJoin requires dimension-scope join support.")
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


def _require_binary(
    states: tuple[DatasetState, ...],
    op_name: str,
) -> tuple[DatasetState, DatasetState]:
    if len(states) != 2:
        raise ValueError(f"{op_name} requires exactly two datasets.")
    return states[0], states[1]
