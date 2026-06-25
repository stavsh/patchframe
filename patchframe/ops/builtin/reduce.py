"""patchframe.ops.builtin.reduce

Group-and-aggregate as a *computation graph*. A reduction is a first-class
operator (``ReducingOperator``); it accepts only a ``BundleField`` field handle
(the fibers) plus its column parameter and returns a field handle — so the lift
(``partition``) always happens before it, and it records a typed reduce-coupling
on the carrier rather than computing eagerly. ``reduce`` is the convenience
wrapper: ``partition`` (the lift) -> apply each reducing operator over the fiber
field -> one ``consume`` to run the graph.

Building the graph (rather than an eager loop) is the point: each reduce-coupling
carries its reducing operator, which exposes a reserved ``bulk_kernel``, so a
future engine / coupling-engine fuser can recognize "N reductions over one
partition" and lower them to a single ``groupby().agg()`` — the engine-owned
optimization. Baking that fused kernel into the operator would couple ``reduce``
to pandas and foreclose a swappable engine (e.g. a differentiable torch backend
for the v2 returning-arrow). v1 runs the graph as a per-fiber loop, which is
engine-agnostic at the operator layer and adequate at current group counts.

Ratios are not reductions: compute them as expressions over the aggregated
columns afterward (a ratio is non-additive — recompute from its additive
components at the target grain). Correct re-aggregation is domain/question-
dependent (micro vs macro), so reductions enforce nothing; they just make the
additive path the natural one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from patchframe.dataset.couplings import Coupling, CouplingSet, FieldRef, _coerce_field_ref
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import BundleField, ValueField
from patchframe.dataset.identity import primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, DatasetOperator, Operator, OperatorCall, Parameter
from patchframe.ops.builtin.consume import consume
from patchframe.ops.builtin.drop import drop
from patchframe.ops.builtin.partition import partition
from patchframe.ops.signature import (
    DatasetInput,
    DatasetReturn,
    FieldInput,
    FieldOutput,
    FieldReturn,
    ParamInput,
)
from patchframe.ops.transitions import (
    Cardinality,
    PerRowIndependence,
    SchemaTransition,
    TransitionPlan,
)

_FIBER_FIELD = "__reduce_fiber__"


def _typed_series(values: list[Any], dtype: Any, index: pd.Index) -> pd.Series:
    if dtype is float:
        return pd.Series(pd.array(values, dtype="Float64"), index=index)
    if dtype is int:
        return pd.Series(pd.array(values, dtype="Int64"), index=index)
    return pd.Series(values, index=index)


@dataclass(frozen=True, slots=True)
class ReduceCoupling(Coupling):
    """A per-fiber reduction recorded on the carrier: read a ``BundleField`` cell
    (a fiber ``Dataset``), reduce one of its columns to a scalar, write the scalar.

    Carries the reducing operator itself (typed), so the engine can introspect it
    (``it's a Sum, kernel "sum"``) to fuse the graph.
    """

    fiber: FieldRef
    output: FieldRef
    reducer: Any
    column: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fiber", _coerce_field_ref(self.fiber))
        object.__setattr__(self, "output", _coerce_field_ref(self.output))

    def input_fields(self) -> tuple[str, ...]:
        return (self.fiber.name,)

    def output_field(self) -> str:
        return self.output.name

    def _reduce(self, fiber: Dataset) -> Any:
        series = fiber.table[self.column] if self.column is not None else fiber.table.index
        return self.reducer.apply(series)

    def compute(self, state: DatasetState) -> pd.Series:
        values = [self._reduce(state.table.at[label, self.fiber.name]) for label in state.table.index]
        return _typed_series(values, self.reducer.dtype, state.table.index)

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        row = dict(row)
        row[self.output.name] = self._reduce(row.get(self.fiber.name))
        return row


class ReducingOperator(DatasetOperator):
    """Baseline reducing operator: reduce a fiber's column to one scalar.

    Concretes (``Sum``/``Count``/``Mean``/``Min``/``Max``/``Distinct``) inherit and
    implement only ``apply``. The operator takes a ``BundleField`` field handle
    (the fibers) + its ``column`` parameter and returns a ``FieldReturn`` — so the
    lift (``partition``) always precedes it, and it is recorded as a same-level
    reduce-coupling on the carrier (``coupling_able`` is ``True``: one scalar
    column per base row). Its declared cardinality is ``REDUCE`` (the fiber-
    internal collapse); ``bulk_kernel`` reserves the future fused kernel.

    Bind the column for use in ``reduce``'s ``aggs`` with ``Sum.on("revenue")``
    (``= instance(column=...)``), which configures without executing.
    """

    transitions = TransitionPlan(schema=SchemaTransition.extend())
    cardinality = Cardinality.REDUCE
    per_row_independent = PerRowIndependence.INDEPENDENT

    bulk_kernel: ClassVar[str | None] = None
    dtype: ClassVar[Any] = None  # honest output dtype (None = engine-inferred)
    needs_column: ClassVar[bool] = True

    column = Parameter(default=None)
    fiber = FieldInput(field_type=BundleField)
    out = FieldOutput(field_type=ValueField)
    returns = FieldReturn()

    def coupling_able(self) -> bool:
        # Always applied over a bundle field (the lift happened first), so the
        # base-level effect is one scalar column per base row — a same-level
        # coupling, despite the REDUCE (fiber-internal) cardinality.
        return True

    @classmethod
    def on(cls, column: str | None = None) -> ReducingOperator:
        """Bind the reduced column (for ``reduce``'s ``aggs``): ``Sum.on("x")``,
        ``Count.on()``. Configures without executing (``= instance(column=...)``)."""
        return cls.instance(column=column)

    def apply(self, series: pd.Series) -> Any:
        raise NotImplementedError

    def _column(self) -> str | None:
        column = self.bound_params().get("column")
        if self.needs_column and not column:
            raise ValueError(
                f"{self.name}: requires a column — bind it with {type(self).__name__}.on(col)."
            )
        return column

    def apply_schema(self, state: DatasetState, fiber: Any, out: str, **_: Any) -> Schema:
        if not isinstance(state.schema.get(fiber), BundleField):
            raise TypeError(f"{self.name}: {fiber!r} must be a BundleField (a partitioned fiber).")
        if state.schema.has(out):
            raise ValueError(f"{self.name}: output field {out!r} already exists.")
        return state.schema.add(ValueField(name=out, dtype=self.dtype))

    def apply_table(self, state: DatasetState, fiber: Any, out: str, **_: Any) -> pd.DataFrame:
        df = state.table.copy()
        df[out] = _typed_series([pd.NA] * len(df), self.dtype, df.index)
        return df

    def new_couplings(self, state: DatasetState, fiber: Any, out: str, **_: Any) -> tuple[ReduceCoupling, ...]:
        return (ReduceCoupling(fiber=fiber, output=out, reducer=self, column=self._column()),)


class Sum(ReducingOperator):
    """Additive sum over a column."""

    bulk_kernel: ClassVar[str | None] = "sum"
    dtype: ClassVar[Any] = float

    def apply(self, series: pd.Series) -> float:
        return float(series.sum()) if len(series) else 0.0


class Count(ReducingOperator):
    """Row count of the group (no column)."""

    needs_column: ClassVar[bool] = False
    bulk_kernel: ClassVar[str | None] = "size"
    dtype: ClassVar[Any] = int

    def apply(self, series: pd.Series) -> int:
        return int(len(series))


class Mean(ReducingOperator):
    """Arithmetic mean over a column (a within-group statistic, not additive)."""

    bulk_kernel: ClassVar[str | None] = "mean"
    dtype: ClassVar[Any] = float

    def apply(self, series: pd.Series) -> float | None:
        return float(series.mean()) if len(series) else None


class Min(ReducingOperator):
    bulk_kernel: ClassVar[str | None] = "min"

    def apply(self, series: pd.Series) -> Any:
        return series.min() if len(series) else None


class Max(ReducingOperator):
    bulk_kernel: ClassVar[str | None] = "max"

    def apply(self, series: pd.Series) -> Any:
        return series.max() if len(series) else None


class Distinct(ReducingOperator):
    """Distinct-count over a column. NON-additive: do not sum distinct counts
    across groups (overlapping members double-count); recompute from base rows."""

    bulk_kernel: ClassVar[str | None] = "nunique"
    dtype: ClassVar[Any] = int

    def apply(self, series: pd.Series) -> int:
        return int(series.nunique())


class reduce(Operator):
    """Group-and-aggregate: ``partition`` then a graph of per-fiber reductions.

    ``reduce(ds, by="creative_id", aggs={"revenue": Sum.on("revenue"), "reach":
    Distinct.on("device_id")})`` returns one row per group with the declared,
    honestly-typed aggregate columns. It composes ``partition`` (inheriting its
    full ``by=``/``domain=`` dispatch and index-identity inheritance), records one
    reduce-coupling per aggregation (carrying the reducing operator), and runs the
    whole graph with a single ``consume`` — the engine's fusion surface.
    """

    transitions = TransitionPlan()
    cardinality = Cardinality.UNKNOWN  # N input rows -> K group rows (a reshape)
    per_row_independent = PerRowIndependence.DEPENDENT  # the shuffle: blocking node
    advances_dataset_context = False
    ds = DatasetInput()
    by = ParamInput()
    domain = DatasetInput()
    aggs = ParamInput()
    returns = DatasetReturn()

    def __call__(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Iterable[str] | Any = None,
        aggs: Mapping[str, ReducingOperator] | None = None,
        domain: Dataset | None = None,
        *,
        null_keys: str = "error",
    ) -> Dataset:
        return Operator.__call__(self, ds, by, aggs, domain=domain, null_keys=null_keys)

    def normalize_call(
        self,
        ds: Dataset | Any = MISSING,
        by: str | Iterable[str] | Any = None,
        aggs: Mapping[str, ReducingOperator] | None = None,
        domain: Dataset | None = None,
        *,
        null_keys: str = "error",
    ) -> OperatorCall:
        if not isinstance(ds, Dataset):
            raise TypeError(f"{self.name} requires a Dataset operand.")
        if isinstance(by, str):
            if not by:
                raise TypeError(f"{self.name} requires `by`: the name of the key field.")
        elif isinstance(by, (list, tuple)) and by and all(isinstance(k, str) and k for k in by):
            by = tuple(by)
        else:
            raise TypeError(
                f"{self.name} requires `by`: a field name or a non-empty list of field names."
            )
        if not aggs or not isinstance(aggs, Mapping):
            raise TypeError(f"{self.name} requires `aggs`: a {{name: reduction}} mapping.")
        for name, reduction in aggs.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{self.name}: aggregation names must be non-empty strings.")
            if not isinstance(reduction, ReducingOperator):
                raise TypeError(
                    f"{self.name}: aggregation {name!r} must be a reduction "
                    f"(Sum/Count/Mean/Min/Max/Distinct), got {type(reduction).__name__}."
                )
        if domain is not None and not isinstance(domain, Dataset):
            raise TypeError(f"{self.name}: `domain` must be a Dataset.")
        if null_keys not in ("error", "drop", "group"):
            raise ValueError(
                f"{self.name}: null_keys must be one of ('error', 'drop', 'group'), got {null_keys!r}."
            )
        datasets = (ds,) if domain is None else (ds, domain)
        return OperatorCall(
            operator=self,
            datasets=datasets,
            kwargs={"by": by, "aggs": dict(aggs), "null_keys": null_keys},
        )

    def run(self, call: OperatorCall, _: Any) -> Dataset:
        ds = call.datasets[0]
        domain = call.datasets[1] if len(call.datasets) > 1 else None
        by = call.kwargs["by"]
        aggs: Mapping[str, ReducingOperator] = call.kwargs["aggs"]
        null_keys: str = call.kwargs["null_keys"]

        # partition owns the key dispatch, domain totality, and index identity.
        groups = partition(ds, by, domain=domain, into=_FIBER_FIELD, null_keys=null_keys)
        index_field = primary_index_field(groups.schema)
        reserved = {index_field.name, *index_field.level_names()}
        for name in aggs:
            if name in reserved:
                raise ValueError(
                    f"{self.name}: aggregation name {name!r} collides with the group key."
                )

        # Build the reduce-coupling graph over the fibers, then run it once.
        table = groups.table.copy()
        schema_fields = list(groups.schema.fields)
        couplings = []
        for name, reduction in aggs.items():
            table[name] = _typed_series([pd.NA] * len(table), reduction.dtype, table.index)
            schema_fields.append(ValueField(name=name, dtype=reduction.dtype))
            couplings.append(
                ReduceCoupling(
                    fiber=_FIBER_FIELD, output=name, reducer=reduction, column=reduction._column()
                )
            )

        realized = Dataset(
            state=DatasetState(
                schema=Schema(fields=tuple(schema_fields)),
                table=table,
                couplings=CouplingSet(tuple(couplings)),
            )
        )
        # Run the graph (one consume per aggregate column discharges its coupling).
        for name in aggs:
            realized = consume(realized, name)
        return drop(realized, [_FIBER_FIELD])

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(f"{self.name}: expected run() to return a Dataset.")
        result.schema.validate_table(result.table)
