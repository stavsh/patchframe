"""patchframe.ops.builtin.make_from_dataframe"""

from __future__ import annotations

import pandas as pd

from patchframe.data.accessor import DataAccessor
from patchframe.data.manager import SourceManager, get_default_manager
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.fields import DataField
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, CreationOperator, Parameter


def _register_datafield_sources(schema: Schema, table: pd.DataFrame, target_manager: SourceManager) -> None:
    """Re-register any DataSource referenced by DataField accessors into target_manager.

    Iterates DataField columns, deduplicates by source_desc_id, and calls
    register_source() on each unique source. Idempotent: sources already
    present in target_manager are no-ops.
    """
    seen: set[int] = set()
    for field in schema.fields:
        if not isinstance(field, DataField) or field.name not in table.columns:
            continue
        for accessor in table[field.name]:
            if not isinstance(accessor, DataAccessor) or accessor.source_desc_id in seen:
                continue
            seen.add(accessor.source_desc_id)
            original_mgr = accessor.manager_hint or get_default_manager()
            source = original_mgr.get_source_by_descriptor_id(accessor.source_desc_id)
            target_manager.register_source(source)


class make_from_dataframe(CreationOperator):
    """Build a dataset from an existing dataframe and schema.

    If the schema contains DataField columns, the DataSource referenced by
    each unique accessor is re-registered into the dataset's SourceManager so
    the new dataset is self-contained.
    """

    copy = Parameter(default=True)

    def generate_source_info(
        self,
        table: pd.DataFrame,
        schema: Schema,
        *,
        couplings: CouplingSet | None = None,
        source_info: DatasetSourceInfo | None = None,
        source_desc_id: int | None = None,
        source_manager: SourceManager | None = None,
        copy: bool | object = MISSING,
        **_,
    ) -> DatasetSourceInfo:
        if source_info is not None:
            return source_info
        return DatasetSourceInfo(source_uri="memory://dataframe", source_type="dataframe")

    def build(
        self,
        table: pd.DataFrame,
        schema: Schema,
        *,
        couplings: CouplingSet | None = None,
        source_info: DatasetSourceInfo | None = None,
        source_desc_id: int | None = None,
        source_manager: SourceManager | None = None,
        copy: bool | object = MISSING,
        **_,
    ) -> DatasetState:
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{self.name} expects 'table' to be a pandas DataFrame")
        if not isinstance(schema, Schema):
            raise TypeError(f"{self.name} expects 'schema' to be a Schema")

        should_copy = self.resolve_param("copy", copy)
        working_table = table.copy() if should_copy else table
        schema.validate_table(working_table)

        if source_manager is not None:
            _register_datafield_sources(schema, working_table, source_manager)

        return DatasetState(
            schema=schema,
            table=working_table,
            couplings=couplings or CouplingSet(),
        )
