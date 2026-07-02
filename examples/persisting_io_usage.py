"""Runnable example for the currently built persistence IO seam.

The disk ``save``/``load`` container is still design-stage. The built path today
is ``offload``: persist a resident ``Dataset`` into a ``DatasetStore`` and read
it back through lazy ``DatasetAccessor`` bundle cells.

Run:

    python examples/persisting_io_usage.py
"""

from __future__ import annotations

import pandas as pd

import patchframe as pf
from patchframe.sources.memory.dataset_source import MemoryDatasetStore


def make_events() -> pf.Dataset:
    """Create a small scalar dataset with explicit row identity."""

    table = pd.DataFrame(
        {
            "clip": ["a", "b", "c", "d", "e", "f"],
            "score": [0.91, 0.20, 0.73, 0.66, 0.12, 0.84],
        },
        index=pd.Index([f"event-{i}" for i in range(6)], name="event_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="event_id"),
            pf.ValueField(name="clip", dtype=str),
            pf.ValueField(name="score", dtype=float),
        )
    )
    return pf.make_from_dataframe(table, schema)


def offload_in_memory(
    ds: pf.Dataset,
    *,
    chunk_size: int = 2,
) -> tuple[pf.Dataset, pf.SourceManager]:
    """Persist ``ds`` into the in-memory store and return a lazy chunk bundle."""

    manager = pf.SourceManager()
    store = MemoryDatasetStore(manager=manager)
    bundle = pf.offload(ds, store=store, chunk_size=chunk_size)
    return bundle, manager


def stream_one_chunk(bundle: pf.Dataset, position: int = 1) -> list[dict[str, object]]:
    """Read one offloaded chunk through row exit.

    ``bundle.rows()[position]["fiber"]`` resolves only that ``DatasetAccessor``.
    The ``BundleField`` exits as a list of row dictionaries, which is the shape a
    dataloader-style consumer would see.
    """

    return bundle.rows()[position]["fiber"]


def round_trip_with_flatten(ds: pf.Dataset) -> pf.Dataset:
    """Offload then flatten all chunks back to the original row space.

    ``flatten`` row-stacks fibers via ``concat_rows``. The original row labels
    are preserved as the index; depending on concat policy, the index field can
    also appear as a normal trace column in the flattened table.
    """

    bundle, _ = offload_in_memory(ds, chunk_size=2)
    return pf.flatten(bundle)


def round_trip_whole_with_extract(ds: pf.Dataset) -> pf.Dataset:
    """Offload as one whole-dataset fiber, then extract that fiber."""

    bundle, _ = offload_in_memory(ds, chunk_size=None)
    return pf.extract(bundle)


def main() -> None:
    ds = make_events()
    bundle, _ = offload_in_memory(ds, chunk_size=2)

    print("Original dataset:")
    print(ds.table.to_string())

    print("\nOffloaded bundle table:")
    print(bundle.table.to_string())
    print(f"\nBundle rows: {len(bundle.table)} chunks")

    chunk = stream_one_chunk(bundle, position=1)
    print("\nRead only chunk 1 through row exit:")
    for row in chunk:
        print(row)

    flattened = pf.flatten(bundle)
    print("\nFlattened read-back:")
    print(flattened.table.to_string())

    whole = round_trip_whole_with_extract(ds)
    assert list(flattened.table.index) == list(ds.table.index)
    pd.testing.assert_frame_equal(flattened.table[ds.table.columns], ds.table)
    pd.testing.assert_frame_equal(whole.table, ds.table)
    print("\nRound-trip checks passed.")


if __name__ == "__main__":
    main()
