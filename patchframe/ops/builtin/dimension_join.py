"""patchframe.ops.builtin.dimension_join"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from patchframe.data.predicates import MatchPredicate
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    DimensionedSliceField,
    ForeignIndexField,
    IndexField,
)
from patchframe.dataset.identity import primary_index_identity
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

#: Index name minted for a correspondence (pair) plan.
PAIR_INDEX_NAME = "pair_id"


class dimension_join(Operator):
    """Apply one single-dimension predicate to two datasets → a correspondence plan.

    The atom of the dimensional join (``dimension-join-execution.md``): given a
    ``MatchPredicate`` and the field(s) it reads on each side, produce or narrow
    a correspondence — a plan dataset whose ``left_index``/``right_index``
    ``ForeignIndexField`` columns reference the two inputs' identities (the same
    shape ``merge``/``explode``/``partition`` consume downstream).

    Chaining is explicit (the compositional path): pass a prior correspondence
    as ``candidates`` and the predicate narrows it. The high-level
    ``join(..., predicates=...)`` wrapper sequences these in stage order (cheap
    partitioners first), so the common surface stays declarative.

    Like ``merge``/``join`` it builds a new plan schema (not coupling-able), so
    its lazy arm lifts onto a ``BundleField`` carrier: bundle the inputs, pass
    the cell handles, and the deferred apply runs at ``collect`` time. The
    operands are datasets-first (``left, right, candidates``) so the deferred
    replay binds cells positionally; ``predicate`` is keyword-only.

    ``left_on``/``right_on`` name the operand on each side: a scalar field (for
    ``equals``), a ``DimensionedSliceField`` plus ``dimension`` (the interval is
    ``slice.dims[dimension]``), or a ``(start_field, end_field)`` pair (an
    interval without a composed slice — a convenience; the wrapper prefers
    composed slices). v1 carries the correspondence as expanded pairs; the
    dimension-keyed term resolution is the wrapper's job.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.clear(),
        index_identity=IndexIdentityTransition.mint(),
    )
    cardinality = Cardinality.UNKNOWN
    per_row_independent = PerRowIndependence.DEPENDENT  # a correlation across two sides
    advances_dataset_context = False
    left = DatasetInput()
    right = DatasetInput()
    candidates = DatasetInput()
    predicate = ParamInput()
    left_on = ParamInput()
    right_on = ParamInput()
    dimension = ParamInput(default=None)
    out = FieldOutput()
    returns = FieldReturn()

    def __call__(
        self,
        left: Dataset | Any = MISSING,
        right: Dataset | Any = MISSING,
        candidates: Dataset | None = None,
        *,
        predicate: MatchPredicate,
        left_on: str | tuple[str, str],
        right_on: str | tuple[str, str],
        dimension: str | None = None,
        out: str | None = None,
    ) -> Dataset:
        # ``out`` flows through to the interpreter: with bundle handle operands
        # it names the deferred result cell; the eager path ignores it. Pass
        # ``candidates`` positionally only when present — a positional ``None``
        # would be bound as a (null) third operand by the bundle binder.
        operands = (left, right) if candidates is None else (left, right, candidates)
        return Operator.__call__(
            self,
            *operands,
            predicate=predicate,
            left_on=left_on,
            right_on=right_on,
            dimension=dimension,
            out=out,
        )

    def normalize_call(
        self,
        left: Dataset | Any = MISSING,
        right: Dataset | Any = MISSING,
        candidates: Dataset | None = None,
        *,
        predicate: MatchPredicate = MISSING,
        left_on: str | tuple[str, str] = MISSING,
        right_on: str | tuple[str, str] = MISSING,
        dimension: str | None = None,
        out: str | None = None,
    ) -> OperatorCall:
        if not isinstance(left, Dataset) or not isinstance(right, Dataset):
            raise TypeError(f"{self.name} requires two Dataset operands (left, right).")
        if not isinstance(predicate, MatchPredicate):
            raise TypeError(f"{self.name} requires a MatchPredicate (predicate=).")
        if candidates is not None and not isinstance(candidates, Dataset):
            raise TypeError(f"{self.name}: `candidates` must be a correspondence Dataset.")
        datasets = (left, right) if candidates is None else (left, right, candidates)
        return OperatorCall(
            operator=self,
            datasets=datasets,
            states=tuple(d.state for d in datasets),
            kwargs={
                "predicate": predicate,
                "left_on": left_on,
                "right_on": right_on,
                "dimension": dimension,
                "has_candidates": candidates is not None,
            },
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        left, right = call.datasets[0], call.datasets[1]
        kwargs = dict(call.kwargs)
        predicate: MatchPredicate = kwargs["predicate"]
        dimension: str | None = kwargs["dimension"]
        candidates = call.datasets[2] if kwargs["has_candidates"] else None

        left_vals = _resolve_operand(left, kwargs["left_on"], dimension, self.name)
        right_vals = _resolve_operand(right, kwargs["right_on"], dimension, self.name)
        candidate_positions = (
            _plan_to_positions(candidates, left, right, self.name)
            if candidates is not None
            else None
        )

        li, ri = predicate.correspond(left_vals, right_vals, candidate_positions)
        return _correspondence_plan(left, right, np.asarray(li), np.asarray(ri))

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)


def _resolve_operand(
    dataset: Dataset,
    on: str | tuple[str, str],
    dimension: str | None,
    op_name: str,
) -> list[Any]:
    """Extract per-row comparison operands from the named field(s).

    - ``(start_field, end_field)`` → ``(start, stop)`` intervals.
    - a ``DimensionedSliceField`` (+ ``dimension``) → ``slice.dims[dimension]``
      as ``(start, stop)`` intervals.
    - any other field → its scalar values (labels for ``equals``).
    """

    if isinstance(on, tuple):
        start_field, end_field = on
        for name in on:
            if name not in dataset.table.columns:
                raise ValueError(f"{op_name}: field {name!r} is not a table column.")
        return list(
            zip(
                dataset.table[start_field].tolist(),
                dataset.table[end_field].tolist(),
                strict=True,
            )
        )

    if not dataset.schema.has(on) or on not in dataset.table.columns:
        raise ValueError(f"{op_name}: field {on!r} is not present on this side.")

    if isinstance(dataset.schema.get(on), DimensionedSliceField):
        if dimension is None:
            raise ValueError(
                f"{op_name}: field {on!r} is a slice; pass `dimension` to select "
                "the interval to compare."
            )
        intervals: list[Any] = []
        for value in dataset.table[on]:
            interval = value.dims[dimension]
            intervals.append((interval.start, interval.stop))
        return intervals

    return dataset.table[on].tolist()


def _plan_to_positions(
    plan: Dataset,
    left: Dataset,
    right: Dataset,
    op_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a prior correspondence plan (labels) to row positions.

    The open-seam safety check: any operator emitting a left_index/right_index
    plan may feed ``candidates``, so validate the plan actually references
    *these* sides — when the mapping columns are typed ``ForeignIndexField``s
    their identities must match left/right (a plan with matching labels but the
    wrong identity is a footgun). Plain (untyped) columns are accepted on a
    labels-only basis.
    """

    for name in ("left_index", "right_index"):
        if name not in plan.table.columns:
            raise ValueError(
                f"{op_name}: `candidates` is not a correspondence plan "
                f"(missing {name!r})."
            )
    _check_correspondence_identity(plan, "left_index", left, op_name)
    _check_correspondence_identity(plan, "right_index", right, op_name)

    li = left.table.index.get_indexer(plan.table["left_index"])
    ri = right.table.index.get_indexer(plan.table["right_index"])
    if (li < 0).any() or (ri < 0).any():
        raise ValueError(
            f"{op_name}: `candidates` references rows absent from left/right."
        )
    return li, ri


def _check_correspondence_identity(
    plan: Dataset,
    field_name: str,
    target: Dataset,
    op_name: str,
) -> None:
    field = plan.schema.get(field_name)
    if not isinstance(field, ForeignIndexField):
        return  # untyped column: labels-only, no identity to validate
    if field.target_identity != primary_index_identity(target.state):
        raise ValueError(
            f"{op_name}: `candidates` field {field_name!r} references a different "
            "dataset's identity than the operand it is matched against."
        )


def _correspondence_plan(
    left: Dataset,
    right: Dataset,
    li: np.ndarray,
    ri: np.ndarray,
) -> Dataset:
    left_labels = left.table.index.to_numpy(dtype=object)[li]
    right_labels = right.table.index.to_numpy(dtype=object)[ri]
    index = pd.RangeIndex(len(li), name=PAIR_INDEX_NAME)
    table = pd.DataFrame(
        {
            "left_index": pd.Series(left_labels, index=index, dtype=object),
            "right_index": pd.Series(right_labels, index=index, dtype=object),
        },
        index=index,
    )
    schema = Schema(
        fields=(
            IndexField(name=PAIR_INDEX_NAME),
            ForeignIndexField(
                name="left_index", index_identity=primary_index_identity(left.state)
            ),
            ForeignIndexField(
                name="right_index", index_identity=primary_index_identity(right.state)
            ),
        )
    )
    # DatasetState mints the pair index identity (ensure_primary_index_identity).
    return Dataset(state=DatasetState(schema=schema, table=table))
