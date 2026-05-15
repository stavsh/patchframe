"""Tests for the AudioSet example workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from examples.audioset import (
    AUDIO_FIELD,
    SEGMENT_END_FIELD,
    SEGMENT_FIELD,
    SEGMENT_START_FIELD,
    SOURCE_AUDIO_ID_FIELD,
    WavDataSource,
    bind_audio_segments,
    make_audio_files,
    make_audioset,
    make_audioset_labels,
    merge_audio_labels,
    parse_labels,
)
from patchframe.data.dimensioned_slice_array import DimensionedSliceArray
from patchframe.data.dimensions import CategoricalDimension
from patchframe.data.manager import reset_default_manager
from patchframe.dataset.fields import DimensionField
from patchframe.ops.builtin.consume import consume
from patchframe.testing import assert_source_contract


@pytest.fixture(autouse=True)
def fresh_manager():
    reset_default_manager()


@pytest.fixture(autouse=True)
def fake_wav_reads(monkeypatch):
    def _read_full(self, item_id, accessor):
        return np.zeros(self.sample_rate * 20, dtype=np.float32)

    def _read_partial(self, item_id, resolved_slice, accessor):
        time_slice = resolved_slice.get("time")
        duration = 8
        if (
            isinstance(time_slice, slice)
            and time_slice.stop is not None
            and time_slice.start is not None
        ):
            duration = max(time_slice.stop - time_slice.start, 1)
        return np.zeros(duration, dtype=np.float32)

    monkeypatch.setattr("examples.audioset.WavDataSource.read_full", _read_full)
    monkeypatch.setattr("examples.audioset.WavDataSource.read_partial", _read_partial)


def _metadata_csv(tmp_path):
    path = tmp_path / "audioset.csv"
    path.write_text(
        "\n".join(
            [
                "# ytid,start_seconds,end_seconds,positive_labels",
                'clip_a,10.0,20.0,"/m/a,/m/b"',
                'clip_b,30.0,33.5,"/m/b"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_make_audioset_binds_segments_by_default(tmp_path):
    ds = make_audioset(_metadata_csv(tmp_path), tmp_path)

    assert ds.schema.has(SOURCE_AUDIO_ID_FIELD)
    assert ds.schema.has(SEGMENT_FIELD)
    assert isinstance(ds.schema.get(SEGMENT_START_FIELD), DimensionField)
    assert isinstance(ds.schema.get(SEGMENT_END_FIELD), DimensionField)
    assert ds.table[SEGMENT_START_FIELD].tolist() == [0.0, 0.0]
    assert ds.table[SEGMENT_END_FIELD].tolist() == [10.0, 3.5]
    assert ds.table.index.name == "clip_id"

    row = ds["clip_a"]
    assert isinstance(row[AUDIO_FIELD], np.ndarray)
    assert ds[0]["clip_id"] == "clip_a"
    assert row[SEGMENT_FIELD].dims["time"] == slice(0.0, 10.0)


def test_audio_files_and_labels_are_separate_sources_before_merge(tmp_path):
    labels = make_audioset_labels(_metadata_csv(tmp_path))
    audio_files = make_audio_files(
        labels.table[SOURCE_AUDIO_ID_FIELD].drop_duplicates(),
        tmp_path,
    )

    assert labels.schema.has(SOURCE_AUDIO_ID_FIELD)
    assert audio_files.schema.has(SOURCE_AUDIO_ID_FIELD)
    assert isinstance(labels.schema.get(SOURCE_AUDIO_ID_FIELD), DimensionField)
    assert isinstance(audio_files.schema.get(SOURCE_AUDIO_ID_FIELD), DimensionField)
    assert labels.schema.get(SOURCE_AUDIO_ID_FIELD).dimension == CategoricalDimension(
        name="audio_id"
    )

    merged = merge_audio_labels(labels, audio_files)
    row = merged["clip_a"]

    assert row[SOURCE_AUDIO_ID_FIELD] == "clip_a"
    assert isinstance(row[AUDIO_FIELD], np.ndarray)
    assert row[SEGMENT_FIELD].dims["time"] == slice(0.0, 10.0)
    assert merged.table.index.name == "clip_id"
    assert merged[0]["clip_id"] == "clip_a"


def test_wav_data_source_satisfies_array_source_contract(tmp_path):
    source = WavDataSource(base_dir=str(tmp_path), sample_rate=16_000)
    dim_slice = source.dimensions.dims[0].spec(0.0, 1.0)

    assert_source_contract(
        source,
        item_id="clip_a",
        dim_slice=dim_slice,
        compare_partial=True,
    )


def test_full_audio_layout_uses_csv_times_as_slice_coordinates(tmp_path):
    ds = make_audioset(_metadata_csv(tmp_path), tmp_path, audio_layout="full")

    row = ds["clip_a"]
    assert row[SEGMENT_FIELD].dims["time"] == slice(10.0, 20.0)
    assert isinstance(row[AUDIO_FIELD], np.ndarray)


def test_bind_audio_segments_can_be_applied_once_after_creation(tmp_path):
    ds = make_audioset(_metadata_csv(tmp_path), tmp_path, bind_segments=False)

    assert not ds.schema.has(SEGMENT_FIELD)
    rebound = bind_audio_segments(ds)

    row = rebound["clip_b"]
    assert row[SEGMENT_FIELD].dims["time"] == slice(0.0, 3.5)
    assert isinstance(row[AUDIO_FIELD], np.ndarray)


def test_consume_segments_keeps_dimensioned_slice_array_columnar(tmp_path):
    ds = make_audioset(_metadata_csv(tmp_path), tmp_path)

    consumed = consume(ds, SEGMENT_FIELD)

    assert isinstance(consumed.table[SEGMENT_FIELD].array, DimensionedSliceArray)
    assert consumed.table[SEGMENT_FIELD].iloc[0].dims["time"] == slice(0.0, 10.0)


def test_parse_labels_handles_audioset_strings():
    assert parse_labels('"/m/a,/m/b"') == ("/m/a", "/m/b")
    assert parse_labels("/m/a, /m/b") == ("/m/a", "/m/b")
    assert parse_labels(pd.NA) == ()


def test_categorical_dimension_can_map_labels_to_positions():
    dim = CategoricalDimension(name="audio_id", categories=("clip_a", "clip_b"))

    spec = dim.spec("clip_b")
    index = dim.to_index(spec.dims["audio_id"])

    assert spec.dims == {"audio_id": "clip_b"}
    assert index.value == 1


def test_torch_dataloader_uses_dataset_rows(tmp_path):
    torch = pytest.importorskip("torch")
    ds = make_audioset(_metadata_csv(tmp_path), tmp_path)

    from examples.audioset import make_torch_dataloader

    loader = make_torch_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch["audio"].ndim == 3
    assert batch["lengths"].shape == torch.Size([2])
    assert batch["item_id"][0] == "clip_a"
