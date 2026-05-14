"""
patchframe.data.dimensions

Named dimension descriptors for array-backed data sources.

Dimensions describe the axis layout of arrays served by an ArrayStore or
DataSource. Each concrete Dimension subclass defines its own natural unit for
slice generation (spec()) and knows how to convert stored values to array
indices (to_index()).

Dimensions.resolve() translates a DimensionedSlice into a tuple of
DimensionIndex objects suitable for numpy array indexing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from patchframe.data.dimensioned_slice import DimensionedSlice


@dataclass(frozen=True, slots=True)
class DimensionIndex:
    """Resolved per-axis array index produced by Dimension.to_index().

    Backends use name for native lazy loading where supported; the default
    application is array[tuple(di.value for di in resolved_indices)].
    """

    name: str
    value: Any  # slice, int, list[int], np.ndarray, etc.


@dataclass(frozen=True, slots=True)
class Dimension(ABC):
    """Abstract base for a single named array axis.

    Defines what an axis means and how to convert natural-unit values to array
    indices. Carries no size or extent information — those are properties of
    specific arrays, not of the dimension definition.
    """

    name: str

    def spec(self, *values: Any) -> DimensionedSlice:
        """Return a DimensionedSlice for values interpreted by this dimension."""
        raise NotImplementedError

    @abstractmethod
    def to_index(self, value: Any) -> DimensionIndex:
        """Convert a stored natural-unit value to a resolved DimensionIndex."""
        ...


@dataclass(frozen=True, slots=True)
class IndexDimension(Dimension):
    """A dimension whose natural unit is raw array indices.

    Slice values are passed through unchanged to the underlying array.
    """

    def spec(self, *values: int) -> DimensionedSlice:
        """Return a DimensionedSlice covering [start, end) in raw indices."""
        if len(values) != 2:
            raise ValueError("IndexDimension expects exactly two selector values.")
        start, end = values
        return DimensionedSlice(dims={self.name: slice(start, end)})

    def to_index(self, value: Any) -> DimensionIndex:
        return DimensionIndex(name=self.name, value=value)


@dataclass(frozen=True, slots=True)
class TemporalDimension(Dimension):
    """A time axis whose natural unit is seconds.

    spec() accepts start/end in seconds and stores them as a float slice.
    to_index() converts the stored seconds-valued slice to sample indices
    using sample_rate.

    Example
    -------
    dim = TemporalDimension(name="time", sample_rate=16000)
    ds = dim.spec(1.5, 4.0)   # → DimensionedSlice(dims={"time": slice(1.5, 4.0)})
    di = dim.to_index(slice(1.5, 4.0))  # → DimensionIndex("time", slice(24000, 64000))
    """

    sample_rate: int = 1

    def spec(self, *values: float) -> DimensionedSlice:
        """Return a DimensionedSlice covering [start, end) in seconds."""
        if len(values) != 2:
            raise ValueError("TemporalDimension expects exactly two selector values.")
        start, end = values
        return DimensionedSlice(dims={self.name: slice(start, end)})

    def to_index(self, value: slice) -> DimensionIndex:
        """Convert a seconds-valued slice to a sample-index DimensionIndex."""
        sr = self.sample_rate
        return DimensionIndex(
            name=self.name,
            value=slice(int(value.start * sr), int(value.stop * sr)),
        )


@dataclass(frozen=True, slots=True)
class CategoricalDimension(Dimension):
    """A dimension whose natural unit is a category label.

    ``categories`` is optional. When provided, ``to_index`` maps category labels
    to integer positions. When omitted, labels pass through unchanged so sources
    can interpret categories directly.
    """

    categories: tuple[Any, ...] = ()

    def spec(self, *values: Any) -> DimensionedSlice:
        if not values:
            raise ValueError("CategoricalDimension expects at least one selector value.")
        value = values[0] if len(values) == 1 else tuple(values)
        return DimensionedSlice(dims={self.name: value})

    def to_index(self, value: Any) -> DimensionIndex:
        if not self.categories:
            return DimensionIndex(name=self.name, value=value)
        if _is_category_sequence(value):
            return DimensionIndex(name=self.name, value=[self._position(v) for v in value])
        return DimensionIndex(name=self.name, value=self._position(value))

    def _position(self, value: Any) -> int:
        try:
            return self.categories.index(value)
        except ValueError as err:
            raise ValueError(
                f"Category {value!r} is not present in dimension {self.name!r}."
            ) from err


def _is_category_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


@dataclass(frozen=True, slots=True)
class Dimensions:
    """Ordered axis layout for an array store or data source."""

    dims: tuple[Dimension, ...] = field(default_factory=tuple)

    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def resolve(self, dim_slice: DimensionedSlice) -> tuple[DimensionIndex, ...]:
        """Translate a DimensionedSlice into a tuple of resolved DimensionIndex objects.

        Each axis present in dim_slice is converted via Dimension.to_index(). Axes
        absent from dim_slice default to DimensionIndex(name, slice(None)) — full
        selection, analogous to numpy's implicit trailing-axis colon.

        Raises ValueError if dim_slice references any unknown dimension name.

        Apply the result as: array[tuple(di.value for di in resolved)]
        """
        unknown = set(dim_slice.dims) - set(self.names())
        if unknown:
            raise ValueError(f"DimensionedSlice references unknown dimensions: {sorted(unknown)}")
        return tuple(
            d.to_index(dim_slice.dims[d.name]) if d.name in dim_slice.dims
            else DimensionIndex(name=d.name, value=slice(None))
            for d in self.dims
        )
