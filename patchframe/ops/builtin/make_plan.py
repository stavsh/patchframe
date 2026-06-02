"""Generic source-indexed plan construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import ForeignIndexField, IndexField, ValueField
from patchframe.dataset.identity import primary_index_identity
from patchframe.dataset.schema import Schema
from patchframe.ops.base import PlanOperator

PLAN_INDEX_NAME = "plan_id"
SOURCE_INDEX_FIELD = "source_index"
PLAN_METADATA_KEY = "patchframe.plan"


class make_plan(PlanOperator):
    """Build a plan dataset with one foreign index into a target dataset."""

    plan_index_name = PLAN_INDEX_NAME
    required_plan_fields = (SOURCE_INDEX_FIELD,)

    def __call__(
        self,
        target: Dataset,
        source_index: Any,
        *,
        source_index_field: str = SOURCE_INDEX_FIELD,
        plan_index_name: str = PLAN_INDEX_NAME,
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        table = _plan_table(
            source_index=source_index,
            source_index_field=source_index_field,
            plan_index_name=plan_index_name,
        )
        return self._build_from_dataframe(
            target,
            table,
            source_index_field=source_index_field,
            plan_index_name=plan_index_name,
            metadata=metadata,
            copy=False,
        )

    @classmethod
    def from_dataframe(
        cls,
        target: Dataset,
        table: pd.DataFrame,
        *,
        source_index_field: str = SOURCE_INDEX_FIELD,
        plan_index_name: str = PLAN_INDEX_NAME,
        metadata: dict[str, Any] | None = None,
        copy: bool = True,
    ) -> Dataset:
        """Build a plan from a dataframe, inferring extra columns as ValueFields."""

        return cls.instance()._build_from_dataframe(
            target,
            table,
            source_index_field=source_index_field,
            plan_index_name=plan_index_name,
            metadata=metadata,
            copy=copy,
        )

    @classmethod
    def from_series(
        cls,
        target: Dataset,
        source_index: pd.Series,
        *,
        source_index_field: str | None = None,
        plan_index_name: str = PLAN_INDEX_NAME,
        metadata: dict[str, Any] | None = None,
        copy: bool = True,
    ) -> Dataset:
        """Build a one-column plan from a series of target index labels."""

        field_name = source_index_field or source_index.name or SOURCE_INDEX_FIELD
        table = source_index.to_frame(name=field_name)
        return cls.from_dataframe(
            target,
            table,
            source_index_field=field_name,
            plan_index_name=plan_index_name,
            metadata=metadata,
            copy=copy,
        )

    def _build_from_dataframe(
        self,
        target: Dataset,
        table: pd.DataFrame,
        *,
        source_index_field: str,
        plan_index_name: str,
        metadata: dict[str, Any] | None,
        copy: bool,
    ) -> Dataset:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("make_plan.from_dataframe expects a pandas DataFrame.")
        if source_index_field not in table.columns:
            raise ValueError(
                f"make_plan: source index field {source_index_field!r} "
                "is missing from the table."
            )

        working = table.copy() if copy else table
        working.index = working.index.rename(plan_index_name)
        _validate_source_index(target, working[source_index_field], field_name=source_index_field)
        schema = Schema(
            fields=(
                IndexField(name=plan_index_name),
                ForeignIndexField(
                    name=source_index_field,
                    nullable=False,
                    index_identity=primary_index_identity(target),
                ),
                *(
                    ValueField(name=name)
                    for name in working.columns
                    if name != source_index_field
                ),
            )
        )
        return self.build_plan_dataset(
            schema=schema,
            table=working,
            sources=tuple(target.sources),
            source_manager=target.source_manager,
            metadata=(
                metadata
                if metadata is not None
                else _plan_metadata(source_index_field=source_index_field)
            ),
            plan_index_name=plan_index_name,
            required_plan_fields=(source_index_field,),
        )


def _plan_table(
    *,
    source_index: Any,
    source_index_field: str,
    plan_index_name: str,
) -> pd.DataFrame:
    values = list(source_index)
    return pd.DataFrame(
        {source_index_field: pd.Series(values, dtype=object)},
        index=pd.RangeIndex(len(values), name=plan_index_name),
    )


def _validate_source_index(
    target: Dataset,
    labels: pd.Series,
    *,
    field_name: str,
) -> None:
    null_mask = pd.isna(labels)
    if null_mask.any():
        raise ValueError(f"make_plan: source index field {field_name!r} contains null labels.")
    missing = labels[~labels.isin(target.table.index)].tolist()
    if missing:
        raise ValueError(
            f"make_plan: source index field {field_name!r} references labels "
            f"missing from target dataset: {missing}"
        )


def _plan_metadata(*, source_index_field: str) -> dict[str, Any]:
    return {
        PLAN_METADATA_KEY: {
            "type": "source_indexed",
            "operator": "make_plan",
            "source_index_field": source_index_field,
        }
    }
