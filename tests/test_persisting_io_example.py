"""Guards the runnable persistence IO usage example against regressions."""

from __future__ import annotations

import pandas as pd

import patchframe as pf
from examples.persisting_io_usage import (
    make_events,
    offload_in_memory,
    round_trip_whole_with_extract,
    round_trip_with_flatten,
    stream_one_chunk,
)


def test_offload_example_streams_one_chunk() -> None:
    ds = make_events()
    bundle, _ = offload_in_memory(ds, chunk_size=2)

    assert len(bundle.table) == 3
    assert isinstance(bundle.schema.get("fiber"), pf.BundleField)

    chunk = stream_one_chunk(bundle, position=1)
    assert [row["clip"] for row in chunk] == ["c", "d"]
    assert [row["score"] for row in chunk] == [0.73, 0.66]


def test_offload_example_flatten_round_trips() -> None:
    ds = make_events()
    flat = round_trip_with_flatten(ds)

    pd.testing.assert_frame_equal(flat.table[ds.table.columns], ds.table)
    assert list(flat.table.index) == list(ds.table.index)
    assert all(name in flat.schema.names() for name in ds.schema.names())


def test_offload_example_extract_round_trips_whole_dataset() -> None:
    ds = make_events()
    whole = round_trip_whole_with_extract(ds)

    pd.testing.assert_frame_equal(whole.table, ds.table)
    assert whole.schema.names() == ds.schema.names()
