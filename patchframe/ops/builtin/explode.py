"""patchframe.ops.builtin.explode"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import (
    CompositionContext,
    compose_rows,
)
from patchframe.dataset.fields import ForeignIndexField, IndexField
from patchframe.dataset.identity import primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, ContextEffect, Operator, OperatorCall, PlanConsumerMixin
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn
from patchframe.ops.builtin._composition import normalize_field_names, normalize_table_to_schema
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


class explode(PlanConsumerMixin, Operator):
    """Apply a source-indexed row expansion plan to one dataset."""

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.construct(),
        sources=SourcesTransition.inherit(),
        index_identity=IndexIdentityTransition.inherit(input="plan"),
    )
    cardinality = Cardinality.EXPAND
    per_row_independent = PerRowIndependence.INDEPENDENT
    source = DatasetInput()
    plan = DatasetInput()
    out = FieldOutput()
    returns = FieldReturn()

    def __call__(
        self,
        source: Dataset | Any = MISSING,
        plan: Dataset | Any = MISSING,
        *,
        foreign_index_field: str | None = None,
        overlay_fields: str | Iterable[str] | None = None,
        out: str | None = None,
    ) -> Dataset:
        # ``out`` flows through to the interpreter (Operator.__call__): with
        # bundle handle operands it names the deferred result cell; the eager
        # path ignores it.
        return Operator.__call__(
            self,
            source,
            plan,
            foreign_index_field=foreign_index_field,
            overlay_fields=overlay_fields,
            out=out,
        )

    def normalize_call(
        self,
        source: Dataset | Any = MISSING,
        plan: Dataset | Any = MISSING,
        *,
        foreign_index_field: str | None = None,
        overlay_fields: str | Iterable[str] | None = None,
        out: str | None = None,
    ) -> OperatorCall:
        self._assert_field_handles_allowed(
            source,
            plan,
            {
                "foreign_index_field": foreign_index_field,
                "overlay_fields": overlay_fields,
            },
        )
        dataset_context = self.resolve_dataset_context()
        if plan is MISSING:
            if dataset_context is None or not isinstance(source, Dataset):
                raise TypeError("explode requires a source Dataset and a plan Dataset.")
            plan = source
            source = dataset_context.dataset
        elif source is MISSING:
            if dataset_context is None:
                raise TypeError("explode requires a source Dataset.")
            source = dataset_context.dataset
        if not isinstance(source, Dataset) or not isinstance(plan, Dataset):
            raise TypeError("explode requires a source Dataset and a plan Dataset.")

        context_effects: tuple[ContextEffect, ...] = ()
        if dataset_context is not None and source is dataset_context.dataset:
            context_effects = (
                ContextEffect(
                    context=dataset_context,
                    anchor=source,
                    effect="advance",
                ),
            )
        return OperatorCall(
            operator=self,
            datasets=(source, plan),
            states=(source.state, plan.state),
            kwargs={
                "foreign_index_field": foreign_index_field,
                "overlay_fields": overlay_fields,
            },
            context_effects=context_effects,
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        source, plan = call.datasets
        kwargs = dict(call.kwargs)
        foreign_index_field = kwargs["foreign_index_field"]
        overlay_fields = kwargs["overlay_fields"]

        self.validate_plan_dataset(plan)
        source_index_field = self.resolve_plan_foreign_index(
            plan,
            source,
            field_name=foreign_index_field,
        )
        source_labels = plan.table[source_index_field.name]
        self.validate_foreign_index_labels(
            source,
            source_labels,
            field_name=source_index_field.name,
            allow_null=False,
        )

        overlays = _resolve_overlay_fields(
            source.state,
            plan.state,
            source_index_field=source_index_field,
            overlay_fields=overlay_fields,
            op_name=self.name,
        )
        schema = _exploded_schema(source.state, plan.state)
        table = _exploded_table(
            source.state,
            plan.state,
            source_labels=source_labels,
            overlay_fields=overlays,
            schema=schema,
            op_name=self.name,
        )
        result = Dataset(
            state=DatasetState(
                schema=schema,
                table=table,
                couplings=source.couplings,
                sources=source.sources,
                source_descriptors=source.state.source_descriptors,
                assets=source.state.assets,
                views=source.state.views,
                metadata=dict(source.state.metadata),
            ),
            source_manager=source.source_manager,
        )
        return result

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)


def _exploded_schema(source: DatasetState, plan: DatasetState) -> Schema:
    plan_index = primary_index_field(plan.schema)
    source_fields = tuple(field for field in source.schema if not isinstance(field, IndexField))
    return Schema(fields=(plan_index, *source_fields))


def _exploded_table(
    source: DatasetState,
    plan: DatasetState,
    *,
    source_labels: pd.Series,
    overlay_fields: tuple[str, ...],
    schema: Schema,
    op_name: str,
) -> pd.DataFrame:
    columns = tuple(field.name for field in source.schema if not isinstance(field, IndexField))
    missing_columns = [name for name in columns if name not in source.table.columns]
    if missing_columns:
        raise ValueError(f"{op_name}: source table is missing schema columns: {missing_columns}")

    result = source.table.reindex(list(source_labels)).loc[:, list(columns)]
    result.index = plan.table.index
    for name in overlay_fields:
        result[name] = plan.table[name]
    result = result.loc[:, [field.name for field in schema if not isinstance(field, IndexField)]]
    result = normalize_table_to_schema(
        result,
        schema,
        CompositionContext(role="column_add", op=op_name),
    )
    schema.validate_table(result)
    return result


def _resolve_overlay_fields(
    source: DatasetState,
    plan: DatasetState,
    *,
    source_index_field: ForeignIndexField,
    overlay_fields: str | Iterable[str] | None,
    op_name: str,
) -> tuple[str, ...]:
    names = (
        _infer_overlay_fields(source, plan, source_index_field=source_index_field)
        if overlay_fields is None
        else normalize_field_names(overlay_fields)
    )
    if not names:
        raise ValueError(f"{op_name}: no overlay fields were provided or inferred.")

    for name in names:
        _validate_overlay_field(source, plan, name, op_name=op_name)
    return names


def _infer_overlay_fields(
    source: DatasetState,
    plan: DatasetState,
    *,
    source_index_field: ForeignIndexField,
) -> tuple[str, ...]:
    excluded = {
        field.name
        for field in plan.schema
        if isinstance(field, (IndexField, ForeignIndexField))
    }
    excluded.add(source_index_field.name)
    return tuple(
        field.name
        for field in plan.schema
        if field.name not in excluded and source.schema.has(field.name)
    )


def _validate_overlay_field(
    source: DatasetState,
    plan: DatasetState,
    name: str,
    *,
    op_name: str,
) -> None:
    if not plan.schema.has(name) or name not in plan.table.columns:
        raise ValueError(f"{op_name}: overlay field {name!r} is not present in the plan.")
    if not source.schema.has(name):
        raise ValueError(f"{op_name}: overlay field {name!r} is not present in the source.")

    plan_field = plan.schema.get(name)
    if isinstance(plan_field, (IndexField, ForeignIndexField)):
        raise TypeError(f"{op_name}: overlay field {name!r} must be a non-index field.")

    source_field = source.schema.get(name)
    try:
        compose_rows(
            (source_field, plan_field),
            CompositionContext(role="row_stack", op=op_name),
        )
    except TypeError as err:
        raise TypeError(f"{op_name}: overlay field {name!r} is incompatible: {err}") from err
