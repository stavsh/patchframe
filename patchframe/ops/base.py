"""
patchframe.ops.base

Base operation model for patchframe.

Operations transform datasets aspect-by-aspect. The default behavior is to
preserve an aspect unless a subclass overrides the corresponding hook.
"""

from __future__ import annotations

from patchframe.dataset.dataset import Dataset
from patchframe.dataset.state import DatasetState
from patchframe.ops.effects import OperationEffects


class Operation:
    """Base class for all patchframe operations."""

    effects: OperationEffects = OperationEffects()

    def apply(self, dataset: Dataset) -> Dataset:
        """Apply this operation to a dataset and return a new dataset."""
        state = dataset.state
        state = self.apply_schema(state)
        state = self.apply_table(state)
        state = self.apply_bindings(state)
        state = self.apply_provenance(state)
        state = self.apply_accessors(state)
        return Dataset(state=state, source_manager=dataset.source_manager)

    def apply_schema(self, state: DatasetState) -> DatasetState:
        return state

    def apply_table(self, state: DatasetState) -> DatasetState:
        return state

    def apply_bindings(self, state: DatasetState) -> DatasetState:
        return state

    def apply_provenance(self, state: DatasetState) -> DatasetState:
        return state

    def apply_accessors(self, state: DatasetState) -> DatasetState:
        return state