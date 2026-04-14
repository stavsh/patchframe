"""
patchframe.dataset.dataset

Dataset facade for patchframe.

``Dataset`` is a thin wrapper around ``DatasetState`` that provides convenient
access to schema, table, bindings, provenance, and binding compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from patchframe.data.manager import SourceManager
from patchframe.dataset.bind_engine import BindEngine
from patchframe.dataset.state import DatasetState


@dataclass(slots=True)
class Dataset:
    """Thin dataset facade over ``DatasetState``."""

    state: DatasetState
    source_manager: SourceManager | None = None

    @property
    def schema(self):
        return self.state.schema

    @property
    def table(self) -> pd.DataFrame:
        return self.state.table

    @property
    def bindings(self):
        return self.state.bindings

    @property
    def provenance(self):
        return self.state.provenance

    def bind_engine(self) -> BindEngine:
        """Compile and validate a binding engine for this dataset."""
        engine = BindEngine(schema=self.schema, bindings=self.bindings)
        engine.validate()
        return engine

    def replace_state(self, **kwargs) -> "Dataset":
        """Return a new dataset with parts of the state replaced."""
        return Dataset(state=replace(self.state, **kwargs), source_manager=self.source_manager)

    def close(self) -> None:
        """Release runtime resources associated with this dataset."""
        # TODO: when datasets begin owning leases explicitly, release them here.
        return None