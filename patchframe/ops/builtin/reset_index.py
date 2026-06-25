"""patchframe.ops.builtin.reset_index"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.fields import IndexField
from patchframe.dataset.identity import maybe_primary_index_field
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    SourcesTransition,
    TransitionPlan,
)


class reset_index(DatasetOperator):
    """Move the index to data columns and mint a fresh positional index.

    The inverse of ``set_index``, and the sanctioned way *out* of a composite
    index (a composite is atomic, so to address a component you decompose it,
    not reach in). The decomposition is the index field's own knowledge
    (``to_data_fields``): a ``CompositeIndexField`` becomes its level columns —
    a level that was a ``ForeignIndexField`` stays one, so a subsequent
    ``partition``/rollup by that column re-aligns by identity — and a single
    index becomes one ``IndexColumnField`` column that remembers its namespace.
    A fresh positional index (default name ``"index"``) is minted; couplings are
    cleared (a structural reshape).

    Usage
    -----
    reset_index(ds)                       # eager -> Dataset (new index "index")
    reset_index(ds, index_name="row")     # name the new positional index
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.inherit(),
        index_identity=IndexIdentityTransition.mint(),
    )
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.DEPENDENT  # the new index is positional
    dataset = DatasetInput()
    out = FieldOutput()
    returns = FieldReturn()

    def apply_schema(
        self, state: DatasetState, *, index_name: str = "index", **_: Any
    ) -> Schema:
        old_index = maybe_primary_index_field(state.schema)
        if old_index is None:
            raise ValueError(f"{self.name}: the dataset has no index to reset.")
        if state.schema.has(index_name):
            raise ValueError(
                f"{self.name}: new index name {index_name!r} collides with an existing field."
            )
        output_fields = [IndexField(name=index_name)]
        output_fields.extend(old_index.to_data_fields())  # index -> its data columns
        output_fields.extend(f for f in state.schema if f is not old_index)
        return Schema(fields=tuple(output_fields))

    def apply_table(
        self, state: DatasetState, *, index_name: str = "index", **_: Any
    ) -> pd.DataFrame:
        # reset_index moves every index level to a column (single or MultiIndex,
        # uniformly) and leaves a default RangeIndex, which we name.
        return state.table.reset_index().rename_axis(index_name)
