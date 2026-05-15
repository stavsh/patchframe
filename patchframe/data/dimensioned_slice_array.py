"""Columnar pandas array for DimensionedSlice values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from pandas.api.extensions import take as take_array

from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimension, IndexDimension
from patchframe.data.windows import AxisWindow


def _missing_mask(values: np.ndarray) -> np.ndarray:
    mask = np.asarray(pd.isna(values), dtype=bool)
    if mask.shape == ():
        return np.full(values.shape, bool(mask.item()), dtype=bool)
    return mask


@register_extension_dtype
class DimensionedSliceDtype(ExtensionDtype):
    """Pandas dtype for columnar DimensionedSlice data."""

    name: ClassVar[str] = "dimensioned_slice"
    type: ClassVar[type] = DimensionedSlice
    kind: ClassVar[str] = "O"
    na_value: ClassVar[Any] = pd.NA

    @classmethod
    def construct_array_type(cls):
        return DimensionedSliceArray


class DimensionedSliceArray(ExtensionArray):
    """Columnar DimensionedSlice storage with scalar DimensionedSlice access."""

    def __init__(
        self,
        dimensions: Sequence[Dimension] = (),
        selector_columns: Any = (),
        base: Any = None,
        mask: Any = None,
        *,
        copy: bool = True,
    ) -> None:
        base_mask = None
        if isinstance(base, DimensionedSliceArray):
            dimensions = (*base._dimensions, *tuple(dimensions))
            selector_columns = (*base._selector_columns, *tuple(selector_columns))
            base_mask = base._mask
            base = base._base

        self._dimensions: tuple[Dimension, ...] = tuple(dimensions)
        self._selector_columns: tuple[tuple[np.ndarray, ...], ...] = tuple(
            tuple(self._coerce_column(col, copy=copy) for col in columns)
            for columns in selector_columns
        )
        self._base: np.ndarray | None = self._coerce_base(base, copy=copy)
        self._length: int = self._infer_length(mask)
        self._mask: np.ndarray = self._build_mask(mask=mask, base_mask=base_mask, copy=copy)

    @staticmethod
    def _coerce_column(column: Sequence[Any], *, copy: bool) -> np.ndarray:
        values = np.asarray(column, dtype=object)
        return values.copy() if copy else values

    @staticmethod
    def _coerce_base(base: Any, *, copy: bool) -> np.ndarray | None:
        if base is None:
            return None
        values = np.asarray(base, dtype=object)
        return values.copy() if copy else values

    @classmethod
    def from_columns(
        cls,
        *,
        dimensions: Sequence[Dimension],
        selector_columns: Any,
        base: Any = None,
    ) -> DimensionedSliceArray:
        if hasattr(base, "array") and isinstance(base.array, DimensionedSliceArray):
            base = base.array
        return cls(dimensions=dimensions, selector_columns=selector_columns, base=base)

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        dtype: ExtensionDtype | None = None,
        copy: bool = False,
    ) -> DimensionedSliceArray:
        values = np.asarray(scalars, dtype=object)
        return cls(base=values, mask=_missing_mask(values), copy=copy)

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[DimensionedSliceArray],
    ) -> DimensionedSliceArray:
        arrays = tuple(to_concat)
        if not arrays:
            return cls()

        non_null = tuple(arr for arr in arrays if not arr.isna().all())
        if not non_null:
            return cls(mask=np.ones(sum(len(arr) for arr in arrays), dtype=bool))

        signature = non_null[0]._signature()
        if any(arr._signature() != signature for arr in non_null[1:]):
            raise TypeError("Cannot concatenate incompatible DimensionedSliceArray values.")

        aligned = tuple(
            arr if arr._signature() == signature else cls._missing_like(non_null[0], len(arr))
            for arr in arrays
        )
        selector_columns = tuple(
            tuple(
                np.concatenate([arr._selector_columns[i][j] for arr in aligned])
                for j in range(len(aligned[0]._selector_columns[i]))
            )
            for i in range(len(aligned[0]._selector_columns))
        )
        base = None
        if any(arr._base is not None for arr in aligned):
            base = np.concatenate([
                arr._base if arr._base is not None else np.full(len(arr), None, dtype=object)
                for arr in aligned
            ])
        mask = np.concatenate([arr._mask for arr in aligned])
        return cls(
            dimensions=aligned[0]._dimensions,
            selector_columns=selector_columns,
            base=base,
            mask=mask,
            copy=False,
        )

    @classmethod
    def _missing_like(
        cls,
        template: DimensionedSliceArray,
        length: int,
    ) -> DimensionedSliceArray:
        selector_columns = tuple(
            tuple(np.full(length, None, dtype=object) for _ in columns)
            for columns in template._selector_columns
        )
        return cls(
            dimensions=template._dimensions,
            selector_columns=selector_columns,
            mask=np.ones(length, dtype=bool),
            copy=False,
        )

    @property
    def dtype(self) -> DimensionedSliceDtype:
        return DimensionedSliceDtype()

    @property
    def nbytes(self) -> int:
        selector_nbytes = sum(
            col.nbytes for columns in self._selector_columns for col in columns
        )
        base_nbytes = self._base.nbytes if self._base is not None else 0
        return selector_nbytes + base_nbytes + self._mask.nbytes

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item: int | slice | Sequence[int] | np.ndarray) -> Any:
        if isinstance(item, (int, np.integer)):
            pos = int(item)
            if pos < 0:
                pos += len(self)
            return self._scalar_at(pos)

        selector_columns = tuple(
            tuple(col[item] for col in columns)
            for columns in self._selector_columns
        )
        base = None if self._base is None else self._base[item]
        return type(self)(
            dimensions=self._dimensions,
            selector_columns=selector_columns,
            base=base,
            mask=self._mask[item],
            copy=False,
        )

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray([self[i] for i in range(len(self))], dtype=dtype or object)

    def isna(self) -> np.ndarray:
        return self._mask.copy()

    def take(
        self,
        indices: Sequence[int],
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> DimensionedSliceArray:
        selector_columns = tuple(
            tuple(
                take_array(
                    col,
                    indices,
                    allow_fill=allow_fill,
                    fill_value=fill_value,
                )
                for col in columns
            )
            for columns in self._selector_columns
        )
        base = None
        if self._base is not None:
            base = take_array(
                self._base,
                indices,
                allow_fill=allow_fill,
                fill_value=fill_value,
            )
        mask = take_array(
            self._mask,
            indices,
            allow_fill=allow_fill,
            fill_value=True,
        )
        return type(self)(
            dimensions=self._dimensions,
            selector_columns=selector_columns,
            base=base,
            mask=mask,
            copy=False,
        )

    def copy(self) -> DimensionedSliceArray:
        return type(self)(
            dimensions=self._dimensions,
            selector_columns=self._selector_columns,
            base=self._base,
            mask=self._mask,
            copy=True,
        )

    @property
    def dimensions(self) -> tuple[Dimension, ...]:
        """Return dimensions represented by vectorized selector columns."""

        return self._dimensions

    def dimension_names(self) -> tuple[str, ...]:
        """Return dimension names represented by vectorized selector columns."""

        return tuple(dim.name for dim in self._dimensions)

    def explode_windows(
        self,
        windows: Mapping[str, AxisWindow],
    ) -> tuple[np.ndarray, DimensionedSliceArray]:
        """Expand each bounded slice row into regular n-dimensional windows.

        Returns
        -------
        parent_positions:
            Integer positions in this array, repeated once for each generated
            window.
        slices:
            A columnar ``DimensionedSliceArray`` of generated tile slices.
        """

        if not windows:
            raise ValueError("explode_windows requires at least one window.")

        names = tuple(windows)
        extents = {
            name: self._extent_columns(name)
            for name in names
        }
        window_dimensions = tuple(self._dimension_for_name(name) for name in names)

        parent_positions = np.flatnonzero(~self._mask)
        selector_by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            starts_all, stops_all = extents[name]
            starts = starts_all[parent_positions]
            stops = stops_all[parent_positions]
            counts = windows[name].counts(starts, stops)

            selector_by_name = {
                existing_name: tuple(np.repeat(col, counts) for col in columns)
                for existing_name, columns in selector_by_name.items()
            }
            next_parent_positions = np.repeat(parent_positions, counts)
            selector_by_name[name] = windows[name].intervals(starts, stops, counts)
            parent_positions = next_parent_positions

        base = self.take(parent_positions) if len(parent_positions) else self.take([])
        slices = type(self).from_columns(
            dimensions=window_dimensions,
            selector_columns=tuple(selector_by_name[name] for name in names),
            base=base,
        )
        return parent_positions, slices

    def _signature(self) -> tuple[tuple[Dimension, ...], tuple[int, ...]]:
        return self._dimensions, tuple(len(columns) for columns in self._selector_columns)

    def _dimension_for_name(self, name: str) -> Dimension:
        for dimension in reversed(self._dimensions):
            if dimension.name == name:
                return dimension
        return IndexDimension(name=name)

    def _extent_columns(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        dimension_columns = tuple(zip(
            self._dimensions,
            self._selector_columns,
            strict=True,
        ))
        for dimension, columns in reversed(dimension_columns):
            if dimension.name != name:
                continue
            if len(columns) != 2:
                raise ValueError(
                    f"Dimension {name!r} is not interval-like and cannot be windowed."
                )
            return columns[0], columns[1]

        starts = np.full(len(self), None, dtype=object)
        stops = np.full(len(self), None, dtype=object)
        for pos in np.flatnonzero(~self._mask):
            scalar = self._scalar_at(int(pos))
            if not isinstance(scalar, DimensionedSlice):
                raise TypeError("explode_windows requires DimensionedSlice rows.")
            if name not in scalar.dims:
                raise ValueError(
                    f"DimensionedSlice row {pos} does not contain dimension {name!r}."
                )
            value = scalar.dims[name]
            if not isinstance(value, slice) or value.stop is None:
                raise ValueError(
                    f"Dimension {name!r} must be represented by a bounded slice."
                )
            starts[pos] = 0 if value.start is None else value.start
            stops[pos] = value.stop
        return starts, stops

    def _infer_length(self, mask: Sequence[bool] | None) -> int:
        lengths = [len(col) for columns in self._selector_columns for col in columns]
        if self._base is not None:
            lengths.append(len(self._base))
        if mask is not None:
            lengths.append(len(mask))
        if not lengths:
            return 0

        length = lengths[0]
        if any(candidate != length for candidate in lengths[1:]):
            raise ValueError("DimensionedSliceArray columns must all have the same length.")
        return length

    def _build_mask(
        self,
        *,
        mask: Sequence[bool] | None,
        base_mask: np.ndarray | None,
        copy: bool,
    ) -> np.ndarray:
        if mask is None:
            result = np.zeros(self._length, dtype=bool)
        else:
            result = np.asarray(mask, dtype=bool)
            result = result.copy() if copy else result

        if base_mask is not None:
            result = result | base_mask
        elif self._base is not None and not self._selector_columns:
            result = result | _missing_mask(self._base)

        for columns in self._selector_columns:
            for col in columns:
                result = result | _missing_mask(col)
        return result

    def _scalar_at(self, pos: int) -> DimensionedSlice | Any:
        if self._mask[pos]:
            return pd.NA

        base = self._base[pos] if self._base is not None else None
        if isinstance(base, DimensionedSlice):
            dims = dict(base.dims)
            metadata = dict(base.metadata)
        else:
            dims = {}
            metadata = {}

        for dimension, columns in zip(
            self._dimensions,
            self._selector_columns,
            strict=True,
        ):
            fragment = dimension.spec(*(col[pos] for col in columns))
            dims.update(fragment.dims)
            metadata.update(fragment.metadata)

        if not self._dimensions and isinstance(base, DimensionedSlice):
            return base
        return DimensionedSlice(dims=dims, metadata=metadata)
