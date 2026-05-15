"""Tests for ArrayDataSource and source contract checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from patchframe.data.accessor import DataAccessor
from patchframe.data.array_source import ArrayDataSource
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions, IndexDimension
from patchframe.sources.memory.source import MemoryDataSource, _MemoryArrayEntry
from patchframe.testing import assert_source_contract


class FullReadSource(ArrayDataSource):
    source_type = "full_read_test"
    config_fields = ("name", "arrays")
    identity_fields = ("name",)

    def read_full(self, item_id: Any, accessor: DataAccessor) -> np.ndarray:
        return self.arrays[item_id]


class PartialReadSource(FullReadSource):
    source_type = "partial_read_test"
    supports_partial_read = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.full_reads = 0
        self.partial_reads = 0

    def read_full(self, item_id: Any, accessor: DataAccessor) -> np.ndarray:
        self.full_reads += 1
        return self.arrays[item_id]

    def read_partial(
        self,
        item_id: Any,
        resolved_slice,
        accessor: DataAccessor,
    ) -> np.ndarray:
        self.partial_reads += 1
        return self.arrays[item_id][resolved_slice.values()]


class BrokenRoundtripSource(FullReadSource):
    source_type = "broken_roundtrip_test"

    @classmethod
    def open(cls, descriptor: SourceDescriptor):
        return cls(
            dimensions=descriptor.capabilities["dimensions"],
            name="different",
            arrays=descriptor.open_config["arrays"],
        )


class RuntimeOnlySource(FullReadSource):
    source_type = "runtime_only_test"
    reopenable = False
    portable = False

    @classmethod
    def open(cls, descriptor: SourceDescriptor):
        raise AssertionError("Runtime-only sources should not be reopened by the contract helper.")


def _dims() -> Dimensions:
    return Dimensions((IndexDimension(name="y"), IndexDimension(name="x")))


def _array() -> np.ndarray:
    return np.arange(20).reshape(4, 5)


def _source(cls=FullReadSource):
    return cls(name="source-a", arrays={"tile": _array()}, dimensions=_dims())


def _slice() -> DimensionedSlice:
    return DimensionedSlice(dims={"y": slice(1, 3), "x": slice(2, 5)})


def test_full_read_source_applies_resolved_slice_after_reading_full_item():
    source = _source()
    accessor = DataAccessor(source_desc_id=1, item_id="tile", dimensioned_slice=_slice())

    result = source.materialize(accessor)

    np.testing.assert_array_equal(result, _array()[1:3, 2:5])


def test_partial_read_source_uses_partial_path_for_sliced_accessors():
    source = _source(PartialReadSource)
    accessor = DataAccessor(source_desc_id=1, item_id="tile", dimensioned_slice=_slice())

    result = source.materialize(accessor)

    np.testing.assert_array_equal(result, _array()[1:3, 2:5])
    assert source.partial_reads == 1
    assert source.full_reads == 0


def test_partial_read_source_uses_full_path_for_unsliced_accessors():
    source = _source(PartialReadSource)
    accessor = DataAccessor(source_desc_id=1, item_id="tile")

    result = source.materialize(accessor)

    np.testing.assert_array_equal(result, _array())
    assert source.partial_reads == 0
    assert source.full_reads == 1


def test_descriptor_roundtrip_preserves_source_identity_and_dimensions():
    source = _source()
    descriptor = source.describe()
    reopened = FullReadSource.open(descriptor)

    assert reopened.describe().source_id == descriptor.source_id
    assert reopened.dimensions == source.dimensions
    np.testing.assert_array_equal(
        reopened.materialize(DataAccessor(source_desc_id=1, item_id="tile")),
        _array(),
    )


def test_source_id_is_stable_for_identity_fields():
    left = _source()
    right = FullReadSource(name="source-a", arrays={"tile": _array() + 1}, dimensions=_dims())

    assert left.describe().source_id == right.describe().source_id


def test_slice_accessor_rejects_unknown_dimensions():
    source = _source()
    accessor = DataAccessor(source_desc_id=1, item_id="tile")

    with pytest.raises(ValueError, match="unknown dimensions"):
        source.slice_accessor(
            accessor,
            DimensionedSlice(dims={"z": slice(0, 1)}),
        )


def test_resolved_slice_keeps_name_and_tuple_access():
    source = _source()
    resolved = source.resolve_slice(_slice())

    assert resolved.names() == ("y", "x")
    assert resolved.get("y") == slice(1, 3)
    assert resolved.by_name()["x"] == slice(2, 5)
    assert resolved.values() == (slice(1, 3), slice(2, 5))


def test_source_contract_passes_for_full_read_source():
    assert_source_contract(_source(), item_id="tile", dim_slice=_slice())


def test_source_contract_passes_for_partial_read_source():
    assert_source_contract(
        _source(PartialReadSource),
        item_id="tile",
        dim_slice=_slice(),
        compare_partial=True,
    )


def test_source_contract_detects_broken_descriptor_roundtrip():
    with pytest.raises(AssertionError, match="source_id"):
        assert_source_contract(
            _source(BrokenRoundtripSource),
            item_id="tile",
            dim_slice=_slice(),
        )


def test_source_contract_allows_runtime_only_sources_without_reopen_roundtrip():
    assert_source_contract(
        _source(RuntimeOnlySource),
        item_id="tile",
        dim_slice=_slice(),
    )


def test_source_contract_passes_for_memory_data_source():
    source = MemoryDataSource(
        dimensions=_dims(),
        entries={
            "tile": {
                "data": _MemoryArrayEntry(
                    array=_array(),
                    extent=DimensionedSlice(dims={"y": slice(0, 4), "x": slice(0, 5)}),
                )
            }
        },
    )

    assert_source_contract(source, item_id="tile", dim_slice=_slice())
