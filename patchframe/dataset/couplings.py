"""
patchframe.dataset.couplings

Class-based field couplings for patchframe datasets.

A Coupling is a frozen-dataclass declaration that:
- exposes which columns it reads (``input_fields``) and writes (``output_field``);
- provides a ``compute(state)`` hook for bulk materialization (used by the
  ``consume`` operator);
- provides an ``apply_row(row, state)`` hook for per-row dispatch (used by
  ``Dataset.__getitem__``).

Couplings live in ``CouplingSet`` and are interpreted by ``CouplingEngine``.
``FieldRef`` is a typed wrapper marking attributes that reference column names,
so generic helpers like ``rename_field_refs`` can rewrite references without
per-subclass boilerplate.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from patchframe.dataset.state import DatasetState


@dataclass(frozen=True, slots=True)
class FieldRef:
    """Typed value wrapper for a field name reference.

    Coupling subclasses declare any attribute that names a column as
    ``FieldRef`` (or ``tuple[FieldRef, ...]``). This lets generic helpers
    (rename, drop, project) rewrite references via dataclass introspection.

    FieldRef is a *value*, not a live handle: external holders do not auto-
    update across operator calls — same semantics as bare strings.
    """

    name: str

    def __str__(self) -> str:
        return self.name


def _coerce_field_ref(value: Any) -> Any:
    """Wrap a string in FieldRef; pass through tuple of strings as tuple of FieldRef."""
    if isinstance(value, str):
        return FieldRef(value)
    if isinstance(value, tuple) and value and all(isinstance(v, str) for v in value):
        return tuple(FieldRef(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class Coupling:
    """Abstract base for class-based field couplings.

    Subclasses are frozen dataclasses that declare input/output fields and
    implement ``compute`` (bulk) and ``apply_row`` (per-row dispatch).
    """

    def input_fields(self) -> tuple[str, ...]:
        raise NotImplementedError

    def output_field(self) -> str:
        raise NotImplementedError

    def compute(self, state: DatasetState) -> pd.Series:
        """Bulk materialization. Returns the new column as a Series indexed by table.index."""
        raise NotImplementedError

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        """Per-row dispatch. Default: return row unchanged."""
        return row


def _rewrite_field_refs(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, FieldRef):
        return FieldRef(mapping.get(value.name, value.name))
    if isinstance(value, tuple):
        return tuple(_rewrite_field_refs(v, mapping) for v in value)
    return value


def rename_field_refs(coupling: Coupling, mapping: dict[str, str]) -> Coupling:
    """Return a new Coupling with FieldRef-typed attributes renamed per mapping.

    Walks dataclass fields, finds ``FieldRef`` and ``tuple[FieldRef, ...]``
    instances, replaces names per mapping. Other attributes pass through.
    """
    updates: dict[str, Any] = {}
    for f in fields(coupling):
        value = getattr(coupling, f.name)
        new_value = _rewrite_field_refs(value, mapping)
        if new_value != value:
            updates[f.name] = new_value
    return replace(coupling, **updates) if updates else coupling


def _coerce_field_ref_tuple(value: Any) -> tuple[FieldRef, ...]:
    if isinstance(value, FieldRef):
        return (value,)
    if isinstance(value, str):
        return (FieldRef(value),)
    refs = tuple(FieldRef(v) if isinstance(v, str) else v for v in value)
    if not all(isinstance(ref, FieldRef) for ref in refs):
        raise TypeError("Coupling field bindings must contain field names or FieldRef values.")
    return refs


@dataclass(frozen=True, slots=True)
class CouplingSet:
    """Ordered collection of class-based couplings."""

    couplings: tuple[Coupling, ...] = field(default_factory=tuple)

    def add(self, *new_couplings: Coupling) -> CouplingSet:
        """Return a new coupling set with additional couplings appended."""
        return CouplingSet(couplings=self.couplings + tuple(new_couplings))

    def rewrite_field_names(self, mapping: dict[str, str]) -> CouplingSet:
        """Return a new coupling set with all FieldRef attributes renamed."""
        return CouplingSet(couplings=tuple(rename_field_refs(c, mapping) for c in self.couplings))

    def retain(self, field_names: set[str]) -> CouplingSet:
        """Return a new coupling set keeping only couplings whose fields are all in field_names."""
        valid = tuple(
            c
            for c in self.couplings
            if field_names.issuperset({c.output_field(), *c.input_fields()})
        )
        return CouplingSet(couplings=valid)


# ---------------------------------------------------------------------------
# Concrete couplings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindSlice(Coupling):
    """Apply a DimensionedSliceField column to a DataField column.

    Per-row: replaces the data accessor with a sliced accessor.
    Bulk: rebuilds the data column with sliced accessors.
    Output is written back to ``data_field`` in place.
    """

    slice_field: FieldRef
    data_field: FieldRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_field", _coerce_field_ref(self.slice_field))
        object.__setattr__(self, "data_field", _coerce_field_ref(self.data_field))

    def input_fields(self) -> tuple[str, ...]:
        return (self.slice_field.name, self.data_field.name)

    def output_field(self) -> str:
        return self.data_field.name

    def compute(self, state: DatasetState) -> pd.Series:
        from patchframe.data.accessor import DataAccessor
        from patchframe.data.dimensioned_slice import DimensionedSlice

        slices = state.table[self.slice_field.name]
        accs = state.table[self.data_field.name]
        return pd.Series(
            [
                a.slice(s) if isinstance(a, DataAccessor) and isinstance(s, DimensionedSlice) else a
                for a, s in zip(accs, slices, strict=True)
            ],
            index=state.table.index,
        )

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        from patchframe.data.accessor import DataAccessor
        from patchframe.data.dimensioned_slice import DimensionedSlice

        s = row.get(self.slice_field.name)
        a = row.get(self.data_field.name)
        if isinstance(a, DataAccessor) and isinstance(s, DimensionedSlice):
            row = dict(row)
            row[self.data_field.name] = a.slice(s)
        return row


@dataclass(frozen=True, slots=True)
class BindDimensions(Coupling):
    """Build a DimensionedSliceField from DimensionField table values.

    ``bindings`` may be a mapping of dimension name to field names, or an
    ordered collection of field-name tuples. Tuple bindings infer the dimension
    from the referenced DimensionField definitions.
    """

    slice_field: FieldRef
    bindings: Any
    dimension_names: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_field", _coerce_field_ref(self.slice_field))
        if self.dimension_names and self._is_normalized_bindings(self.bindings):
            dimension_names = self.dimension_names
            bindings = self.bindings
        else:
            dimension_names, bindings = self._normalize_bindings(self.bindings)
        object.__setattr__(self, "dimension_names", dimension_names)
        object.__setattr__(self, "bindings", bindings)

    @staticmethod
    def _is_normalized_bindings(value: Any) -> bool:
        return (
            isinstance(value, tuple)
            and all(isinstance(binding, tuple) for binding in value)
            and all(isinstance(ref, FieldRef) for binding in value for ref in binding)
        )

    @staticmethod
    def _normalize_bindings(
        value: Any,
    ) -> tuple[tuple[str | None, ...], tuple[tuple[FieldRef, ...], ...]]:
        if isinstance(value, Mapping):
            dimension_names = tuple(str(name) for name in value)
            bindings = tuple(_coerce_field_ref_tuple(fields_) for fields_ in value.values())
            return dimension_names, bindings

        raw_bindings = tuple(value)
        if raw_bindings and all(isinstance(v, str) for v in raw_bindings):
            raw_bindings = (raw_bindings,)
        return tuple(None for _ in raw_bindings), tuple(
            _coerce_field_ref_tuple(fields_) for fields_ in raw_bindings
        )

    def input_fields(self) -> tuple[str, ...]:
        return (
            self.slice_field.name,
            *(ref.name for binding in self.bindings for ref in binding),
        )

    def output_field(self) -> str:
        return self.slice_field.name

    def _dimensions(self, state: DatasetState) -> tuple[Any, ...]:
        from patchframe.dataset.fields import DimensionField

        dimensions = []
        for dimension_name, binding in zip(
            self.dimension_names,
            self.bindings,
            strict=True,
        ):
            if not binding:
                raise ValueError("BindDimensions bindings must contain at least one field.")

            field_defs = []
            for ref in binding:
                field_def = state.schema.get(ref.name)
                if not isinstance(field_def, DimensionField):
                    raise TypeError(f"BindDimensions field {ref.name!r} is not a DimensionField.")
                field_defs.append(field_def)

            dimension = field_defs[0].dimension
            if any(field_def.dimension != dimension for field_def in field_defs[1:]):
                names = [field_def.name for field_def in field_defs]
                raise ValueError(f"BindDimensions fields span multiple dimensions: {names}")
            if dimension_name is not None and dimension.name != dimension_name:
                raise ValueError(
                    f"BindDimensions mapping key {dimension_name!r} does not match "
                    f"DimensionField dimension {dimension.name!r}."
                )
            dimensions.append(dimension)
        return tuple(dimensions)

    def _build_slice(
        self,
        base: Any,
        dimensions: tuple[Any, ...],
        values: tuple[tuple[Any, ...], ...],
    ) -> Any:
        from patchframe.data.dimensioned_slice import DimensionedSlice

        if isinstance(base, DimensionedSlice):
            dims = dict(base.dims)
            metadata = dict(base.metadata)
        else:
            dims = {}
            metadata = {}

        for dimension, dimension_values in zip(dimensions, values, strict=True):
            fragment = dimension.spec(*dimension_values)
            dims.update(fragment.dims)
            metadata.update(fragment.metadata)
        return DimensionedSlice(dims=dims, metadata=metadata)

    def compute(self, state: DatasetState) -> pd.Series:
        from patchframe.data.dimensioned_slice_array import DimensionedSliceArray

        dimensions = self._dimensions(state)
        selector_columns = tuple(
            tuple(state.table[ref.name].to_numpy(copy=True) for ref in binding)
            for binding in self.bindings
        )
        array = DimensionedSliceArray.from_columns(
            dimensions=dimensions,
            selector_columns=selector_columns,
            base=state.table[self.slice_field.name],
        )
        return pd.Series(array, index=state.table.index)

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        dimensions = self._dimensions(state)
        values = tuple(tuple(row[ref.name] for ref in binding) for binding in self.bindings)
        row = dict(row)
        row[self.slice_field.name] = self._build_slice(
            row.get(self.slice_field.name),
            dimensions,
            values,
        )
        return row


@dataclass(frozen=True, slots=True)
class Materialize(Coupling):
    """Materialize a DataAccessor column into concrete arrays.

    Per-row: replaces the accessor with the result of ``accessor.materialize()``.
    Bulk: rebuilds the column with materialized values. Opt-in — lazy access
    remains the default unless this coupling is added to the dataset.
    """

    field: FieldRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _coerce_field_ref(self.field))

    def input_fields(self) -> tuple[str, ...]:
        return (self.field.name,)

    def output_field(self) -> str:
        return self.field.name

    def compute(self, state: DatasetState) -> pd.Series:
        from patchframe.data.accessor import DataAccessor

        col = state.table[self.field.name]
        return pd.Series(
            [v.materialize() if isinstance(v, DataAccessor) else v for v in col],
            index=state.table.index,
        )

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        from patchframe.data.accessor import DataAccessor

        v = row.get(self.field.name)
        if isinstance(v, DataAccessor):
            row = dict(row)
            row[self.field.name] = v.materialize()
        return row


class UnpicklableCallWarning(UserWarning):
    """A deferred operator call carries an argument (or operator) that cannot pickle.

    Raised when a deferred call is recorded (e.g. by ``defer_in_level``), so the
    problem surfaces *there* rather than later at ``collect``/save/worker dispatch.
    The call still replays in-process; escalate this category to an error
    (``warnings.filterwarnings("error", category=UnpicklableCallWarning)``) to
    enforce serializable couplings.
    """


@dataclass(frozen=True, slots=True)
class CallSpec:
    """The serializable core of a (deferred) operator call — enough to replay it.

    Captures the operator (class or configured instance, pickled by import
    reference) plus its normalized positional ``args``, keyword ``kwargs``, and
    dispatch ``variant``. This is the pickle-friendly half of an ``OperatorCall``
    (design-constraints §7); the live ``OperatorCall`` keeps the runtime
    datasets/states/contexts/effects, which are never serialized. Pickle-friendly
    *modulo the argument values themselves*: a ``where`` lambda predicate or a
    script-local operator class will not pickle — see ``picklability_error``.
    """

    # TODO(equality): proof CallSpec equality against real user couplings before
    # relying on it for array-bearing specs. The auto-generated dataclass __eq__
    # compares ``args``/``kwargs`` element-wise, which *raises* ("truth value of an
    # array is ambiguous") rather than returning False when a value is a numpy array
    # / pandas object / torch tensor. This is the common case, not an edge case —
    # users will routinely defer functions with array arguments (kernels, masks,
    # constants). It bites wherever couplings are compared by equality: the
    # derive-path dedup (ops/dispatch.py ``_apply_unary_derive`` and ops/base.py
    # ``_resolve_couplings``, which use ``in``/equality, not hashing) and any future
    # set/dedup over couplings. Fix = a safe structural-equality helper (per element:
    # identity, else element-wise ``all`` for array-likes). Unreachable *today* only
    # because the sole CallSpec-dedup producer (``map_fields``) builds an
    # empty-args/kwargs spec with a fresh output name, so the comparison
    # short-circuits before reaching ``args``/``kwargs``.
    operator: Any
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    variant: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    @property
    def operator_name(self) -> str:
        return getattr(self.operator, "__name__", None) or type(self.operator).__name__

    def replay(self, *prepend_args: Any) -> Any:
        """Re-invoke the operator: ``prepend_args`` (e.g. bundle cells), then the spec."""

        return self.operator(*prepend_args, *self.args, **self.kwargs)

    def picklability_error(self) -> Exception | None:
        """Return the exception this spec would raise on pickle, or ``None`` if it pickles."""

        try:
            pickle.dumps(self)
        except Exception as err:  # noqa: BLE001 — any pickling failure is the answer
            return err
        return None


def warn_if_unpicklable(call: CallSpec, *, stacklevel: int = 2) -> None:
    """Emit an ``UnpicklableCallWarning`` if ``call`` cannot pickle (early check)."""

    error = call.picklability_error()
    if error is None:
        return
    warnings.warn(
        f"Deferred operator {call.operator_name!r} cannot be pickled "
        f"({type(error).__name__}: {error}). It replays in-process at collect(), "
        "but the dataset cannot be persisted or sent to a worker while this "
        "coupling is present. Pass picklable arguments (a module-level function, "
        "not a lambda); filter this category to an error to require it.",
        UnpicklableCallWarning,
        stacklevel=stacklevel + 1,
    )


@dataclass(frozen=True, slots=True)
class ApplyOperator(Coupling):
    """Deferred operator application over ``BundleField`` cells.

    The base-level lift that lets an otherwise non-coupling-able operator
    (``merge``, ``concat``, ``join``) be recorded as a coupling: it reads input
    ``BundleField`` columns — each cell a whole ``Dataset`` — runs the *eager*
    operator on the cell datasets, and writes the result ``Dataset`` into the
    output cell. At the base level this is "read cell(s), write a cell, per
    row" (cardinality-preserving, field-filling), so it satisfies the coupling
    contract; the cell encapsulates the operator's internal cardinality change.

    The deferred call lives in a serializable ``CallSpec`` (``call``). Because the
    cells are plain ``Dataset``s, ``compute`` always hits the operator's eager arm
    via ``CallSpec.replay`` — no re-deferral. ``consume`` (the terminal) runs this
    coupling; the bundle's ``collect`` step then extracts the filled output cell.
    """

    inputs: tuple[FieldRef, ...]
    output: FieldRef
    call: CallSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _coerce_field_ref_tuple(self.inputs))
        object.__setattr__(self, "output", _coerce_field_ref(self.output))

    @property
    def operator(self) -> Any:
        """The deferred operator (read view onto ``call``)."""

        return self.call.operator

    @property
    def params(self) -> Mapping[str, Any]:
        """The deferred call's keyword params (read view onto ``call``)."""

        return self.call.kwargs

    def input_fields(self) -> tuple[str, ...]:
        return tuple(ref.name for ref in self.inputs)

    def output_field(self) -> str:
        return self.output.name

    def _apply_operator(self, cells: list[Any]) -> Any:
        return self.call.replay(*cells)

    def compute(self, state: DatasetState) -> pd.Series:
        out = pd.Series(index=state.table.index, dtype=object)
        for label in state.table.index:
            cells = [state.table.at[label, ref.name] for ref in self.inputs]
            out.at[label] = self._apply_operator(cells)
        return out

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        cells = [row.get(ref.name) for ref in self.inputs]
        row = dict(row)
        row[self.output.name] = self._apply_operator(cells)
        return row


@dataclass(frozen=True, slots=True)
class MapCoupling(Coupling):
    """Deferred per-row Python function over field *values*.

    The per-row sibling of ``ApplyOperator``. Where ``ApplyOperator`` runs a
    patchframe operator over ``BundleField`` cells (each cell a whole
    ``Dataset``), ``MapCoupling`` runs a plain callable over the *values* of one
    or more columns, per row, writing ``fn(*values)`` into ``output``. It is the
    coupling recorded by ``map_fields``: a per-row-independent, schema-extending
    add, so it is coupling-able and lives same-level (no bundle).

    The call lives in a serializable ``CallSpec`` whose ``operator`` is the user
    function, so ``replay(*values)`` invokes ``fn(*values, *args, **kwargs)``.
    Like the bundle arm, a lambda or script-local function will not pickle, so
    the recorder warns early via ``warn_if_unpicklable`` (design-constraints §7).

    Row values for ``input_fields`` are passed positionally in declaration order.
    When an input is a ``DataField``, place a ``Materialize`` coupling upstream so
    the function receives a concrete array rather than a ``DataAccessor`` — the
    engine orders ``Materialize`` before this coupling automatically, since its
    output feeds this coupling's input.
    """

    inputs: tuple[FieldRef, ...]
    output: FieldRef
    call: CallSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _coerce_field_ref_tuple(self.inputs))
        object.__setattr__(self, "output", _coerce_field_ref(self.output))

    @property
    def fn(self) -> Any:
        """The deferred per-row function (read view onto ``call``)."""

        return self.call.operator

    def input_fields(self) -> tuple[str, ...]:
        return tuple(ref.name for ref in self.inputs)

    def output_field(self) -> str:
        return self.output.name

    def compute(self, state: DatasetState) -> pd.Series:
        out = pd.Series(index=state.table.index, dtype=object)
        for label in state.table.index:
            values = [state.table.at[label, ref.name] for ref in self.inputs]
            out.at[label] = self.call.replay(*values)
        return out

    def apply_row(self, row: dict[str, Any], state: DatasetState) -> dict[str, Any]:
        values = [row.get(ref.name) for ref in self.inputs]
        row = dict(row)
        row[self.output.name] = self.call.replay(*values)
        return row
