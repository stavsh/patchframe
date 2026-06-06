"""
patchframe.dataset.dataset

Dataset facade for patchframe.

``Dataset`` is a thin wrapper around ``DatasetState`` that provides convenient
access to schema, table, couplings, sources, and coupling compilation. The
``CouplingEngine`` is built lazily on first use and cached on the instance —
since ``DatasetState`` is frozen, replacing state via ``replace_state`` returns
a fresh ``Dataset`` with no cached engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import patchframe.dataset.extension  # noqa: F401 — registers finfo on pd.Series
from patchframe.data.manager import SourceManager
from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.extension import _FIELD
from patchframe.dataset.state import DatasetState

if TYPE_CHECKING:
    from patchframe.dataset.context import DatasetContext, FieldHandle, FieldSelection


@dataclass(slots=True)
class Dataset:
    """Thin dataset facade over ``DatasetState``."""

    state: DatasetState
    source_manager: SourceManager | None = None
    _engine: CouplingEngine | None = field(default=None, init=False, repr=False)
    _context: DatasetContext | None = field(default=None, init=False, repr=False)

    @property
    def schema(self):
        return self.state.schema

    @property
    def table(self) -> pd.DataFrame:
        return self.state.table

    @property
    def couplings(self):
        return self.state.couplings

    @property
    def sources(self):
        return self.state.sources

    def __len__(self) -> int:
        return len(self.table)

    def coupling_engine(self) -> CouplingEngine:
        """Return a cached, validated coupling engine for this dataset."""
        if self._engine is None:
            self._engine = CouplingEngine(schema=self.schema, couplings=self.couplings)
        return self._engine

    def context(self):
        """Return an explicit mutable authoring context rooted at this dataset."""

        from patchframe.dataset.context import DatasetContext

        return DatasetContext(self)

    def field(self, name: str) -> FieldHandle:
        """Return a context-bound handle for one field of this dataset.

        The entry bridge from the eager surface (``Dataset``) to the handle
        surface. Repeated ``field``/``fields`` calls on the same dataset share
        one context, so several handles can be passed to a single operator.
        """

        return self._authoring_context().field(name)

    def fields(self, names: Iterable[str]) -> FieldSelection:
        """Return a :class:`FieldSelection` of handles sharing one context."""

        from patchframe.dataset.context import FieldSelection

        context = self._authoring_context()
        return FieldSelection(tuple(context.field(name) for name in names))

    def _authoring_context(self) -> DatasetContext:
        """Return the authoring context for handles minted off this facade.

        Prefers an ambient ``DatasetContext`` already pointing at this dataset
        (so ``ds.field(...)`` agrees with ``ctx.field(...)`` inside a ``with``
        block); otherwise lazily builds one cached on the facade. The context
        threads forward through lazy operations (a lazy op propagates the
        context); eager operations fork by returning a new facade with its own
        fresh context. It is therefore not re-pinned to ``self``.
        """

        from patchframe.dataset.context import DatasetContext, get_active_dataset_context

        ambient = get_active_dataset_context()
        if ambient is not None and ambient.dataset is self:
            return ambient
        if self._context is None:
            self._context = DatasetContext(self)
        return self._context

    def __getitem__(self, key: Any) -> pd.Series | dict[str, Any]:
        """Column access or coupling-aware row access.

        - ``ds["col"]``     — returns a Series with field info in ``.attrs``.
        - ``ds[item_id]``   — returns a coupling-aware row dict; couplings are
                              applied in topo-sorted order via CouplingEngine.
        """
        if isinstance(key, str) and key in self.table.columns:
            series = self.table[key]
            try:
                series.attrs[_FIELD] = self.schema.get(key)
            except KeyError:
                pass
            return series

        if isinstance(key, (int, np.integer)) and key not in self.table.index:
            row = self.table.iloc[int(key)]
        else:
            row = self.table.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        result: dict[str, Any] = dict(row)
        for field_def in self.schema.fields:
            if field_def.logical_type == "index" and field_def.name not in result:
                result[field_def.name] = row.name

        if self.couplings.couplings:
            result = self.coupling_engine().apply_row(result, self.state)

        return result

    def replace_state(self, **kwargs) -> Dataset:
        """Return a new dataset with parts of the state replaced."""
        return Dataset(state=replace(self.state, **kwargs), source_manager=self.source_manager)

    def close(self) -> None:
        """Release runtime resources associated with this dataset."""
        return None
