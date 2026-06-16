"""Public data-layer exports."""

from patchframe.data.accessor import DataAccessor
from patchframe.data.array_source import ArrayDataSource, ResolvedSlice
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensioned_slice_array import DimensionedSliceArray
from patchframe.data.dimensions import (
    CategoricalDimension,
    Dimension,
    DimensionIndex,
    Dimensions,
    IndexDimension,
    TemporalDimension,
)
from patchframe.data.manager import (
    SourceLease,
    SourceManager,
    get_default_manager,
    reset_default_manager,
)
from patchframe.data.predicates import (
    MatchPredicate,
    Stage,
    assert_predicate_contract,
    equals,
    overlap,
)
from patchframe.data.source import DataSource
from patchframe.data.windows import AxisWindow

__all__ = [
    "ArrayDataSource",
    "AxisWindow",
    "CategoricalDimension",
    "MatchPredicate",
    "Stage",
    "assert_predicate_contract",
    "equals",
    "overlap",
    "DataAccessor",
    "DataSource",
    "Dimension",
    "DimensionIndex",
    "DimensionedSlice",
    "DimensionedSliceArray",
    "Dimensions",
    "IndexDimension",
    "ResolvedSlice",
    "SourceDescriptor",
    "SourceLease",
    "SourceManager",
    "TemporalDimension",
    "get_default_manager",
    "reset_default_manager",
]
