"""Explicit mutable authoring context over immutable Dataset snapshots."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patchframe.dataset.dataset import Dataset
    from patchframe.dataset.fields import Field
    from patchframe.dataset.schema import Schema
    from patchframe.dataset.identity import FieldIdentity

_active_dataset_context: ContextVar[DatasetContext | None] = ContextVar(
    "patchframe_active_dataset_context",
    default=None,
)


@dataclass(slots=True)
class DatasetContext:
    """Mutable pipeline cursor over immutable Dataset snapshots.

    The context is process-local authoring state. It must not be stored inside
    DatasetState or serialized. Operators may resolve it explicitly through
    ``Operator.instance(dataset_context=...)`` or ambiently while it is active
    as a context manager.
    """

    dataset: Dataset
    _token: Token[DatasetContext | None] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __enter__(self) -> DatasetContext:
        if self._token is not None:
            raise RuntimeError("DatasetContext is already active.")
        self._token = _active_dataset_context.set(self)
        return self

    def __exit__(self, *_) -> None:
        if self._token is None:
            raise RuntimeError("DatasetContext is not active.")
        _active_dataset_context.reset(self._token)
        self._token = None

    def adopt(self, dataset: Dataset) -> Dataset:
        """Advance this cursor to a new immutable dataset snapshot."""

        self.dataset = dataset
        return dataset

    def branch(self) -> DatasetContext:
        """Return an independent cursor starting from the current snapshot."""

        return DatasetContext(self.dataset)

    def field(self, name: str) -> FieldHandle:
        """Return a context-bound handle for one field in the current snapshot."""

        field_def = self.dataset.schema.get(name)
        if field_def.field_identity is None:
            raise ValueError(f"Field {name!r} does not have a FieldIdentity.")
        return FieldHandle(
            dataset_context=self,
            field_identity=field_def.field_identity,
            name_hint=field_def.name,
        )


@dataclass(frozen=True, slots=True)
class FieldHandle:
    """Context-bound field handle that follows one FieldIdentity."""

    dataset_context: DatasetContext
    field_identity: FieldIdentity
    name_hint: str

    def resolve(self) -> Field:
        """Resolve this handle against its context's current dataset snapshot."""

        return _resolve_identity(
            self.dataset_context.dataset.schema,
            self.field_identity,
            name_hint=self.name_hint,
        )

    @property
    def name(self) -> str:
        """Return this field's current local name."""

        return self.resolve().name

    def collect(self) -> Dataset:
        """Materialize the pending coupling(s) producing this field; return a Dataset.

        The user-facing exit bridge (lazy-and-bundle.md §1): the nullary terminal,
        the one carve-out to "handles do not execute". Runs the couplings whose
        end node is this field (via the idempotent ``consume``). When the field is
        a ``BundleField`` the filled cell is itself a dataset, so it is extracted
        and returned; otherwise the container dataset (with this field
        materialized) is returned.
        """

        from patchframe.ops.bundle import _collect

        return _collect(self.dataset_context.dataset, self.name)


@dataclass(frozen=True, slots=True)
class FieldSelection:
    """An ordered selection of ``FieldHandle``s sharing one ``DatasetContext``.

    The multi-field authoring operand produced by ``Dataset.fields([...])``. It
    is deliberately *not* a ``FieldRef`` (the persisted coupling reference) and
    *not* an array (the multidimensional data layer); it is just a typed list of
    handles an operator can consume as a unit.
    """

    handles: tuple[FieldHandle, ...]

    def __post_init__(self) -> None:
        handles = tuple(self.handles)
        object.__setattr__(self, "handles", handles)
        contexts: list[DatasetContext] = []
        for handle in handles:
            if not isinstance(handle, FieldHandle):
                raise TypeError("FieldSelection accepts FieldHandle values only.")
            if all(handle.dataset_context is not existing for existing in contexts):
                contexts.append(handle.dataset_context)
        if len(contexts) > 1:
            raise ValueError("FieldSelection handles must share one DatasetContext.")

    def __iter__(self):
        return iter(self.handles)

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index):
        return self.handles[index]

    @property
    def dataset_context(self) -> DatasetContext | None:
        """Return the shared context, or ``None`` for an empty selection."""

        return self.handles[0].dataset_context if self.handles else None

    def names(self) -> tuple[str, ...]:
        """Return each handle's current local field name."""

        return tuple(handle.name for handle in self.handles)

    def resolve(self) -> tuple[Field, ...]:
        """Resolve each handle against the shared context's current snapshot."""

        return tuple(handle.resolve() for handle in self.handles)


def get_active_dataset_context() -> DatasetContext | None:
    """Return the ambient DatasetContext for the current execution context."""

    return _active_dataset_context.get()


def field_handle_contexts(*values: Any) -> tuple[DatasetContext, ...]:
    """Return distinct DatasetContexts referenced by nested FieldHandle values."""

    contexts: list[DatasetContext] = []
    for value in values:
        for handle in _iter_field_handles(value):
            if all(handle.dataset_context is not existing for existing in contexts):
                contexts.append(handle.dataset_context)
    return tuple(contexts)


def resolve_field_name(
    value: str | FieldHandle,
    schema: Schema,
    *,
    op_name: str,
) -> str:
    """Resolve a local field selector from a string or context-bound handle."""

    if isinstance(value, str):
        return value
    if not isinstance(value, FieldHandle):
        raise TypeError(f"{op_name}: expected a field name or FieldHandle.")
    return _resolve_identity(
        schema,
        value.field_identity,
        name_hint=value.name_hint,
        op_name=op_name,
    ).name


def resolve_field_selectors(
    value: Any,
    schema: Schema,
    *,
    op_name: str,
) -> Any:
    """Resolve nested FieldHandles while preserving selector container shape."""

    if isinstance(value, FieldHandle):
        return resolve_field_name(value, schema, op_name=op_name)
    if isinstance(value, FieldSelection):
        return tuple(
            resolve_field_name(handle, schema, op_name=op_name) for handle in value.handles
        )
    if isinstance(value, Mapping):
        return type(value)(
            (
                resolve_field_selectors(key, schema, op_name=op_name),
                resolve_field_selectors(item, schema, op_name=op_name),
            )
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(resolve_field_selectors(item, schema, op_name=op_name) for item in value)
    if isinstance(value, list):
        return [resolve_field_selectors(item, schema, op_name=op_name) for item in value]
    return value


def _iter_field_handles(value: Any):
    if isinstance(value, FieldHandle):
        yield value
        return
    if isinstance(value, FieldSelection):
        yield from value.handles
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_field_handles(key)
            yield from _iter_field_handles(item)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _iter_field_handles(item)


def _resolve_identity(
    schema: Schema,
    identity: FieldIdentity,
    *,
    name_hint: str,
    op_name: str = "FieldHandle",
) -> Field:
    matches = tuple(
        field_def
        for field_def in schema
        if field_def.field_identity == identity
    )
    if not matches:
        raise ValueError(
            f"{op_name}: field handle for {name_hint!r} no longer exists "
            "in the current dataset snapshot."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{op_name}: field handle for {name_hint!r} resolved to "
            f"{len(matches)} fields."
        )
    return matches[0]
