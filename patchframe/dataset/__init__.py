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
    IndexColumnField,
    IndexField,
    ValueField,
    dtype_compatible,
    to_nullable_dtype,
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
    "DatasetSourceInfo",
    "DatasetState",
    "DimensionField",
    "DimensionedSliceField",
    "Field",
    "FieldCompositionPolicy",
    "FieldRef",
    "IndexColumnField",
    "IndexField",
    "Materialize",
    "Schema",
    "ValueField",
    "compose_column",
    "compose_key",
    "compose_rows",
    "dtype_compatible",
    "field_policy_for",
    "normalize_column",
    "register_field_policy",
    "resolve_column_collision",
    "to_nullable_dtype",
]
