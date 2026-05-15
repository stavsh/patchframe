"""Higher-level array data-source base class."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import DimensionIndex, Dimensions
from patchframe.data.source import DataSource


@dataclass(frozen=True, slots=True)
class ResolvedSlice:
    """Dimension-name-preserving resolved slice.

    This is intentionally thin. It keeps dimension names available for source
    implementations while the broader dimensional slicing model is still being
    consolidated.
    """

    indexes: tuple[DimensionIndex, ...]

    def __bool__(self) -> bool:
        return bool(self.indexes)

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self) -> int:
        return len(self.indexes)

    def names(self) -> tuple[str, ...]:
        return tuple(index.name for index in self.indexes)

    def values(self) -> tuple[Any, ...]:
        return tuple(index.value for index in self.indexes)

    def by_name(self) -> dict[str, Any]:
        return {index.name: index.value for index in self.indexes}

    def get(self, name: str, default: Any = None) -> Any:
        return self.by_name().get(name, default)

    def apply(self, array: Any) -> Any:
        if not self.indexes:
            return array
        return array[self.values()]


class ArrayDataSource(DataSource):
    """Convenience base for sources that materialize array-like items.

    Subclasses declare serializable configuration fields through
    ``config_fields`` and usually implement ``read_full``. Sources that can read
    slices efficiently set ``supports_partial_read = True`` and implement
    ``read_partial``.
    """

    source_type: ClassVar[str] = "array"
    runtime: ClassVar[bool] = True
    reopenable: ClassVar[bool] = True
    portable: ClassVar[bool] = True
    config_fields: ClassVar[tuple[str, ...]] = ()
    identity_fields: ClassVar[tuple[str, ...] | None] = None
    supports_partial_read: ClassVar[bool] = False

    def __init__(
        self,
        *,
        dimensions: Dimensions | None = None,
        source_id: str | None = None,
        **config: Any,
    ) -> None:
        unknown = set(config) - set(self.config_fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown source configuration fields: {names}")

        self.dimensions = dimensions or Dimensions()
        self._source_id = source_id
        for name, value in config.items():
            setattr(self, name, value)

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> ArrayDataSource:
        """Reopen a source from the descriptor produced by ``describe``."""
        return cls(
            dimensions=descriptor.capabilities.get("dimensions", Dimensions()),
            source_id=descriptor.source_id,
            **dict(descriptor.open_config),
        )

    def describe(self) -> SourceDescriptor:
        """Return a durable descriptor generated from declared configuration."""
        return SourceDescriptor(
            source_type=self.source_type,
            source_id=self.source_id(),
            open_config=self.open_config(),
            capabilities=self.capabilities(),
        )

    def source_id(self) -> str:
        """Return the stable source identity used by ``SourceManager``."""
        if self._source_id is not None:
            return self._source_id

        fields = self.identity_fields
        if fields is None:
            fields = self.config_fields
        identity = {name: self._require_config_value(name) for name in fields}
        return f"{self.source_type}:{_stable_token(identity)}"

    def open_config(self) -> dict[str, Any]:
        """Return the descriptor configuration needed to reopen this source."""
        return {name: self._require_config_value(name) for name in self.config_fields}

    def capabilities(self) -> dict[str, Any]:
        """Return descriptor capabilities for this source."""
        return {"dimensions": self.dimensions}

    def materialize(self, accessor: DataAccessor) -> Any:
        """Materialize an accessor through partial or full-read paths."""
        dim_slice = accessor.dimensioned_slice
        if dim_slice is not None:
            resolved = self.resolve_slice(dim_slice)
            if self.supports_partial_read:
                return self.read_partial(accessor.item_id, resolved, accessor)
            return self.apply_resolved_slice(
                self.read_full(accessor.item_id, accessor),
                resolved,
            )
        return self.read_full(accessor.item_id, accessor)

    def read_full(self, item_id: Any, accessor: DataAccessor) -> Any:
        """Read the full array-like item for ``item_id``."""
        raise NotImplementedError

    def read_partial(
        self,
        item_id: Any,
        resolved_slice: ResolvedSlice,
        accessor: DataAccessor,
    ) -> Any:
        """Read a resolved slice without loading the full item."""
        raise NotImplementedError

    def inspect(self, accessor: DataAccessor) -> dict[str, Any]:
        """Return generic lightweight source/accessor metadata."""
        return {
            "item_id": accessor.item_id,
            "asset_id": accessor.asset_id,
            "view_id": accessor.view_id,
            "dimensions": self.dimensions.names(),
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def slice_accessor(self, accessor: DataAccessor, dim_slice: DimensionedSlice) -> DataAccessor:
        """Validate a slice against source dimensions and attach it lazily."""
        self.resolve_slice(dim_slice)
        return accessor.slice(dim_slice)

    def resolve_slice(self, dim_slice: DimensionedSlice | None) -> ResolvedSlice:
        """Resolve a natural-unit slice into dimension indexes."""
        return ResolvedSlice(self.dimensions.resolve(dim_slice or DimensionedSlice()))

    def apply_resolved_slice(self, array: Any, resolved_slice: ResolvedSlice) -> Any:
        """Apply resolved dimension indexes to an array-like object."""
        return resolved_slice.apply(array)

    def _require_config_value(self, name: str) -> Any:
        if not hasattr(self, name):
            raise ValueError(
                f"{type(self).__name__}: missing source configuration field {name!r}."
            )
        return getattr(self, name)


def _stable_token(value: Any) -> str:
    data = repr(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
