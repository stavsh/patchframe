"""Public dataset-layer exports."""

from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.couplings import (
    BindDimensions,
    BindSlice,
    Coupling,
    CouplingSet,
    FieldRef,
    Materialize,
)
from patchframe.dataset.context import (
    DatasetContext,
    FieldHandle,
    get_active_dataset_context,
)
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    CompositionContext,
    FieldCompositionPolicy,
    compose_column,
    compose_key,
    compose_rows,
    field_policy_for,
    normalize_column,
    register_field_policy,
    resolve_column_collision,
)
from patchframe.dataset.fields import (
    DataField,
    DimensionField,
    DimensionedSliceField,
    Field,
    ForeignIndexField,
    IndexColumnField,
    IndexField,
    ValueField,
    dtype_compatible,
    to_nullable_dtype,
)
from patchframe.dataset.identity import (
    IndexIdentity,
    ensure_primary_index_identity,
    foreign_index_fields,
    mint_primary_index_identity,
    new_index_identity,
    primary_index_field,
    primary_index_identity,
    resolve_foreign_index_field,
)
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState

__all__ = [
    "BindDimensions",
    "BindSlice",
    "ColumnCollisionStrategy",
    "CompositionContext",
    "Coupling",
    "CouplingEngine",
    "CouplingSet",
    "DataField",
    "Dataset",
    "DatasetContext",
    "DatasetSourceInfo",
    "DatasetState",
    "DimensionField",
    "DimensionedSliceField",
    "Field",
    "FieldHandle",
    "FieldCompositionPolicy",
    "FieldRef",
    "ForeignIndexField",
    "IndexColumnField",
    "IndexField",
    "IndexIdentity",
    "Materialize",
    "Schema",
    "ValueField",
    "compose_column",
    "compose_key",
    "compose_rows",
    "dtype_compatible",
    "ensure_primary_index_identity",
    "field_policy_for",
    "foreign_index_fields",
    "get_active_dataset_context",
    "mint_primary_index_identity",
    "new_index_identity",
    "normalize_column",
    "primary_index_field",
    "primary_index_identity",
    "register_field_policy",
    "resolve_foreign_index_field",
    "resolve_column_collision",
    "to_nullable_dtype",
]
