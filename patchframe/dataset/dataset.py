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

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import patchframe.dataset.extension  # noqa: F401 — registers finfo on pd.Series
from patchframe.data.manager import SourceManager
from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.extension import _FIELD
from patchframe.dataset.fields import exit_value
from patchframe.dataset.state import DatasetState

if TYPE_CHECKING:
    from patchframe.dataset.context import DatasetContext, FieldHandle, FieldSelection
    from patchframe.dataset.fields import Field


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

    def new_field(self, field_def: Field) -> FieldHandle:
        """Declare a new null-filled field and return a handle to it.

        Sugar over the cached authoring cursor (like ``field``/``fields``): the
        field is added to the schema (filled with nulls) and the cursor advances,
        so repeated ``new_field`` calls share one context and their handles
        co-resolve — ready to pass together to ``assign``.
        """

        return self._authoring_context().new_field(field_def)

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
        """Column access (storage) or row access (evaluate + exit).

        - ``ds["col"]``     — the storage surface: returns the stored Series
                              (field info in ``.attrs``); pending couplings are
                              not run, and framework objects (fiber
                              ``Dataset``s, accessors) stay as they are — this
                              is where lazy navigation lives.
        - ``ds[item_id]``   — the exit point from the dataset world: the row's
                              pending couplings are *evaluated* in topo-sorted
                              order — ephemerally: nothing is persisted or
                              discharged (evaluation, not consumption) — and
                              every value then exits to plain Python through
                              its field's conversion (``Field.exit_value`` /
                              ``register_field_exit``): a ``BundleField``
                              fiber leaves as a list of records. Re-access
                              evaluates again.
        """
        if isinstance(key, str) and key in self.table.columns:
            series = self.table[key]
            try:
                series.attrs[_FIELD] = self.schema.get(key)
            except KeyError:
                pass
            return series

        if isinstance(key, (int, np.integer)) and key not in self.table.index:
            warnings.warn(
                "Dataset[int] positional fallback is deprecated: with integer "
                "row labels it silently changes meaning. Use Dataset.rows()[i] "
                "for positional access.",
                DeprecationWarning,
                stacklevel=2,
            )
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
            # Couplings evaluate over raw values (a fuse fn receives the fiber
            # Dataset, not its exported form); the exit pass runs after.
            result = self.coupling_engine().apply_row(result, self.state)

        return {
            name: exit_value(self.schema.get(name), value)
            if self.schema.has(name)
            else value
            for name, value in result.items()
        }

    def rows(self, field: str | Iterable[str] | None = None) -> RowSequence:
        """Positional view over evaluated, exited rows — DataLoader-pluggable.

        ``ds.rows()`` is a duck-typed map-style dataset (``__len__`` +
        ``__getitem__(int)`` + batched ``__getitems__``), so
        ``DataLoader(ds.rows(), ...)`` works directly, with no torch dependency
        in patchframe. Positional semantics is carried by the view's *type* —
        label-based row access stays on ``ds[item_id]``. ``field`` selects the
        per-item payload: ``None`` → the full exited row dict; a name → that
        field's exited value; several names → a sub-dict.
        """

        if field is None or isinstance(field, str):
            selected: str | tuple[str, ...] | None = field
        else:
            selected = tuple(field)
        names = (selected,) if isinstance(selected, str) else (selected or ())
        for name in names:
            if not self.schema.has(name):
                raise ValueError(f"rows: field {name!r} is not in the schema.")
        return RowSequence(dataset=self, field=selected)

    def replace_state(self, **kwargs) -> Dataset:
        """Return a new dataset with parts of the state replaced."""
        return Dataset(state=replace(self.state, **kwargs), source_manager=self.source_manager)

    def assign(self, **columns: Any) -> Dataset:
        """Pandas-style functional assignment: return a new dataset with columns set.

        Sugar over the ``assign`` operator — bare values infer a ``ValueField``;
        pass ``(field_def, values)`` for a typed field — matching pandas'
        ``df.assign`` convention (returns new, never mutates). The mutating
        conventions (``x[col] = values``, ``x.loc[ids] = values``) live on the
        authoring session types (``DatasetContext``, ``FieldHandle``), because
        ``Dataset`` itself is immutable.
        """

        from patchframe.ops.builtin.assign import assign as assign_op

        return assign_op(self, **columns)

    def close(self) -> None:
        """Release runtime resources associated with this dataset."""
        return None


class RowSequence:
    """Positional view over a dataset's evaluated, exited rows.

    The DataLoader face of a dataset (returned by :meth:`Dataset.rows`): a
    duck-typed map-style dataset — ``len(view)`` and ``view[i]`` (linear
    position → evaluated, exited row) are all a torch ``DataLoader`` needs.
    ``__getitems__`` implements the batched-fetch protocol: the batch's rows
    are taken into a transient dataset whose pending couplings are consumed
    once in bulk, then exited per row — amortizing evaluation across the
    batch without touching the source dataset (its couplings stay pending).
    """

    __slots__ = ("dataset", "field")

    def __init__(
        self,
        dataset: Dataset,
        field: str | tuple[str, ...] | None = None,
    ) -> None:
        self.dataset = dataset
        self.field = field

    def __len__(self) -> int:
        return len(self.dataset.table)

    def __getitem__(self, position: int) -> Any:
        # Out-of-range raises IndexError, which also terminates the legacy
        # iteration protocol, so list(view) / for-loops work.
        label = self.dataset.table.index[int(position)]
        return self._select(self.dataset[label])

    def __getitems__(self, positions: Iterable[int]) -> list[Any]:
        index = self.dataset.table.index
        labels = [index[int(position)] for position in positions]
        # Samplers may draw with replacement; the transient needs unique rows.
        unique = list(dict.fromkeys(labels))
        transient = self.dataset.replace_state(table=self.dataset.table.loc[unique])
        outputs = {c.output_field() for c in transient.couplings.couplings}
        if outputs:
            from patchframe.ops.builtin.consume import consume

            for output in outputs:
                transient = consume(transient, output)
        by_label = {label: self._select(transient[label]) for label in unique}
        return [by_label[label] for label in labels]

    def _select(self, row: dict[str, Any]) -> Any:
        if self.field is None:
            return row
        if isinstance(self.field, str):
            return row[self.field]
        return {name: row[name] for name in self.field}
