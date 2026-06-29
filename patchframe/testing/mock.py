"""
patchframe.testing.mock

Dimension-agnostic mock data source and dataset factory for testing.

MockDataSource generates random float32 arrays on demand — no files, no stored
arrays. Shape is derived from per-item extents (DimensionedSlice) merged with
any applied accessor slice.

make_mock_dataset is the corresponding CreationOperator. It accepts either a
single DimensionedSlice (broadcast to all items) or a per-item mapping.

Usage
-----
    from patchframe.data.dimensions import Dimensions, IndexDimension, TemporalDimension
    from patchframe.testing.mock import make_mock_dataset

    # All items share the same extent
    dim = IndexDimension(name="x")
    ds = make_mock_dataset(["a", "b", "c"], Dimensions((dim,)), dim.spec(0, 100), seed=42)

    arr = ds.table["data"].iloc[0].materialize()           # shape (100,)
    crop = ds.table["data"].iloc[0].slice(dim.spec(10, 50)).materialize()  # shape (40,)

    # Per-item extents
    time_dim = TemporalDimension(name="time", sample_rate=16000)
    extents = {"clip_a": time_dim.spec(0.0, 10.0), "clip_b": time_dim.spec(0.0, 3.0)}
    ds = make_mock_dataset(list(extents), Dimensions((time_dim,)), extents, seed=0)
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from patchframe.data.accessor import DataAccessor

if TYPE_CHECKING:
    from patchframe.dataset.dataset import Dataset

from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import DimensionIndex, Dimensions
from patchframe.data.source import DataSource
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.fields import DataField, IndexField
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, CreationOperator, Parameter


def _axis_size(di: DimensionIndex) -> int:
    """Compute the number of elements along one resolved axis.

    Expects fully bounded slices — callers must merge item extent before
    resolving so that slice(None) never appears here.
    """
    v = di.value
    if isinstance(v, slice):
        start = v.start if v.start is not None else 0
        if v.stop is None:
            raise ValueError(
                f"Dimension '{di.name}' has an unbounded stop after extent merge. "
                f"Ensure the item extent covers all dimensions."
            )
        return v.stop - start
    if isinstance(v, (int, np.integer)):
        return 1
    return len(v)  # type: ignore[arg-type]


def _normalize_extents(
    item_ids: Sequence[Any],
    extents: DimensionedSlice | Mapping[Any, DimensionedSlice],
) -> dict[Any, DimensionedSlice]:
    if isinstance(extents, DimensionedSlice):
        return {iid: extents for iid in item_ids}
    return dict(extents)


@dataclass(slots=True)
class MockDataSource(DataSource):
    """Dimension-agnostic source that generates random float32 arrays on demand.

    Each item has its own extent (DimensionedSlice in natural units). Shape for
    a materialization request is computed by merging the item's extent with any
    DimensionedSlice attached to the accessor — the extent provides the full
    bounds for axes not mentioned in the accessor slice.

    Seeded per item as (seed, item_id) so each item produces a different but
    reproducible array. Pass seed=None for non-deterministic output.
    """

    source_type: str = "mock"
    thread_safe: bool = True
    fork_safe: bool = True
    dimensions: Dimensions = field(default_factory=Dimensions)
    _extents: dict[Any, DimensionedSlice] = field(default_factory=dict)
    seed: int | None = None
    _source_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> MockDataSource:
        return cls(
            dimensions=descriptor.capabilities.get("dimensions", Dimensions()),
            _extents=descriptor.open_config.get("extents", {}),
            seed=descriptor.open_config.get("seed"),
            _source_id=descriptor.source_id,
        )

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_type="mock",
            source_id=self._source_id,
            open_config={"extents": self._extents, "seed": self.seed},
            capabilities={"dimensions": self.dimensions},
        )

    def extent_for(self, item_id: Any) -> DimensionedSlice | None:
        return self._extents.get(item_id)

    def _effective_slice(self, accessor: DataAccessor) -> DimensionedSlice:
        item_extent = self._extents[accessor.item_id]
        if accessor.dimensioned_slice is None:
            return item_extent
        return DimensionedSlice(dims={**item_extent.dims, **accessor.dimensioned_slice.dims})

    def _shape_for(self, accessor: DataAccessor) -> tuple[int, ...]:
        resolved = self.dimensions.resolve(self._effective_slice(accessor))
        return tuple(_axis_size(di) for di in resolved)

    def shape(self, accessor: DataAccessor) -> tuple[int, ...]:
        return self._shape_for(accessor)

    def _rng(self, item_id: Any) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, hash(item_id) & 0xFFFFFFFF])

    def materialize(self, accessor: DataAccessor) -> np.ndarray:
        return self._rng(accessor.item_id).random(self._shape_for(accessor)).astype(np.float32)

    def inspect(self, accessor: DataAccessor) -> dict[str, Any]:
        return {
            "shape": self._shape_for(accessor),
            "item_id": accessor.item_id,
            "extent": self._extents.get(accessor.item_id),
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def slice_accessor(self, accessor: DataAccessor, dim_slice: DimensionedSlice) -> DataAccessor:
        unknown = set(dim_slice.dims) - set(self.dimensions.names())
        if unknown:
            raise ValueError(f"DimensionedSlice references unknown dimensions: {sorted(unknown)}")
        return accessor.slice(dim_slice)


class make_mock_dataset(CreationOperator):
    """Create a dataset backed by MockDataSource.

    Parameters
    ----------
    item_ids:
        Sequence of item identifiers.
    dimensions:
        Shared Dimensions describing what each axis means.
    extents:
        Full-array extent for each item, in natural units. Either a single
        DimensionedSlice (broadcast to all items) or a mapping of
        ``{item_id: DimensionedSlice}``.
    data_fields:
        Names of the DataField columns to create. Defaults to ``("data",)``.
    seed:
        Optional integer seed for reproducible array generation.
    """

    seed = Parameter(default=None)

    def make_source(
        self,
        item_ids: Sequence[Any],
        dimensions: Dimensions,
        extents: DimensionedSlice | Mapping[Any, DimensionedSlice],
        *,
        data_fields: tuple[str, ...] = ("data",),
        seed: int | None | object = MISSING,
        **_: Any,
    ) -> MockDataSource:
        return MockDataSource(
            dimensions=dimensions,
            _extents=_normalize_extents(item_ids, extents),
            seed=self.resolve_param("seed", seed),
            _source_id=str(uuid.uuid4()),
        )

    def generate_source_info(
        self,
        item_ids: Sequence[Any],
        dimensions: Dimensions,
        extents: DimensionedSlice | Mapping[Any, DimensionedSlice],
        *,
        data_fields: tuple[str, ...] = ("data",),
        seed: int | None | object = MISSING,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> DatasetSourceInfo:
        return DatasetSourceInfo(source_uri="mock://", source_type="mock", source_name="mock")

    def build(
        self,
        item_ids: Sequence[Any],
        dimensions: Dimensions,
        extents: DimensionedSlice | Mapping[Any, DimensionedSlice],
        *,
        data_fields: tuple[str, ...] = ("data",),
        seed: int | None | object = MISSING,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> DatasetState:
        ids = list(item_ids)
        df = pd.DataFrame(index=pd.Index(ids, name="item_id"))
        for field_name in data_fields:
            df[field_name] = [
                DataAccessor(source_desc_id=source_desc_id, item_id=iid) for iid in ids
            ]

        schema = Schema(
            fields=(
                IndexField(name="item_id"),
                *(DataField(name=f) for f in data_fields),
            )
        )

        return DatasetState(schema=schema, table=df, couplings=CouplingSet())


def make_mock_dataset_from_dims(
    ds: Dataset,
    *,
    data_field: str = "data",
    seed: int | None = None,
) -> Dataset:
    """Create a mock dataset whose array extents are derived from DimensionField columns.

    Scans the schema for DimensionField entries, groups fields by their
    ``.dimension`` object (in schema order), and calls ``dimension.spec(*values)``
    per row to build a per-item DimensionedSlice extent.  The resulting extents
    are passed to ``make_mock_dataset``, which returns a fresh dataset containing
    ``item_id`` and a single data column.

    Parameters
    ----------
    ds:
        Dataset with DimensionField columns encoding per-row extents.
    data_field:
        Name of the single data column in the returned dataset. Defaults to
        ``"data"``.
    seed:
        Optional seed for reproducible array generation.
    """
    from patchframe.dataset.fields import DimensionField

    dim_to_fields: dict[Any, list[Any]] = {}
    for f in ds.schema:
        if isinstance(f, DimensionField):
            dim_to_fields.setdefault(f.dimension, []).append(f)

    if not dim_to_fields:
        raise ValueError("make_mock_dataset_from_dims: no DimensionField columns in schema.")

    dimensions = Dimensions(tuple(dim_to_fields.keys()))

    extents: dict[Any, DimensionedSlice] = {}
    for item_id, row in ds.table.iterrows():
        dims: dict[str, Any] = {}
        for dim, fields in dim_to_fields.items():
            values = tuple(row[f.name] for f in fields)
            fragment = dim.spec(*values)
            dims.update(fragment.dims)
        extents[item_id] = DimensionedSlice(dims=dims)

    return make_mock_dataset(
        list(ds.table.index),
        dimensions,
        extents,
        data_fields=(data_field,),
        seed=seed,
    )
