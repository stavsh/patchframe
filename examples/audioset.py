"""AudioSet example built around separate label and audio sources.

The source-specific work lives in the makers:

- ``make_audio_files`` parses the WAV side
- ``make_audioset_labels`` parses the label/segment side
- ``merge_audio_labels`` composes them through ``join``/``merge``

The conventional usage path stays short:

    ds = make_audioset(csv_path, wav_dir)
    row = ds["clip_id"]
    audio = row["audio"]

By default ``audio`` is already materialized through a coupling, so the row
access path can be used directly for visualization and batch loading.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import patchframe as pf

AudioLayout = Literal["segments", "full"]
ChannelLayout = Literal["channels_first", "channels_last"]

AUDIO_FIELD = "audio"
SOURCE_AUDIO_ID_FIELD = "audio_id"
SEGMENT_FIELD = "segment"
SEGMENT_START_FIELD = "segment_start_seconds"
SEGMENT_END_FIELD = "segment_end_seconds"
LABELS_FIELD = "labels"
_AUDIOSET_CSV_COLUMNS = ("ytid", "start_seconds", "end_seconds", "labels")


class WavDataSource(pf.ArrayDataSource):
    """DataSource that reads WAV files on demand."""

    source_type = "wav"
    thread_safe: bool = True
    fork_safe: bool = True
    config_fields = ("base_dir", "sample_rate", "file_template", "channel_layout")
    supports_partial_read = True

    def __init__(
        self,
        *,
        base_dir: str,
        sample_rate: int = 16_000,
        file_template: str = "{item_id}.wav",
        channel_layout: ChannelLayout = "channels_first",
        dimensions: pf.Dimensions | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions or _audio_dimensions(sample_rate),
            source_id=source_id,
            base_dir=os.path.abspath(base_dir),
            sample_rate=sample_rate,
            file_template=file_template,
            channel_layout=channel_layout,
        )

    def read_full(self, item_id: Any, accessor: pf.DataAccessor) -> np.ndarray:
        return self._read_wav(item_id)

    def read_partial(
        self,
        item_id: Any,
        resolved_slice: pf.ResolvedSlice,
        accessor: pf.DataAccessor,
    ) -> np.ndarray:
        start, stop = self._sample_bounds(resolved_slice)
        return self._read_wav(item_id, start=start, stop=stop)

    def _read_wav(
        self,
        item_id: Any,
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        sf = _soundfile()
        read_kwargs: dict[str, Any] = {"dtype": "float32", "always_2d": True}
        if start is not None or stop is not None:
            read_kwargs["start"] = start or 0
            read_kwargs["stop"] = stop
        audio, _ = sf.read(self._path_for(item_id), **read_kwargs)
        if self.channel_layout == "channels_first":
            audio = audio.T
        return audio

    def inspect(self, accessor: pf.DataAccessor) -> dict[str, Any]:
        sf = _soundfile()
        info = sf.info(self._path_for(accessor.item_id))
        return {
            "frames": info.frames,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "item_id": accessor.item_id,
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def _path_for(self, item_id: Any) -> str:
        return os.path.join(self.base_dir, self.file_template.format(item_id=item_id))

    def _sample_bounds(self, resolved: pf.ResolvedSlice) -> tuple[int | None, int | None]:
        if not resolved:
            return None, None
        value = resolved.get("time")
        if isinstance(value, slice):
            return value.start, value.stop
        if isinstance(value, int | np.integer):
            pos = int(value)
            return pos, pos + 1
        return None, None


class make_audio_files(pf.CreationOperator):
    """Build a dataset over WAV files without parsing labels."""

    sample_rate = pf.Parameter(default=16_000)
    file_template = pf.Parameter(default="{item_id}.wav")
    channel_layout = pf.Parameter(default="channels_first")

    def make_source(
        self,
        audio_ids: Any,
        base_dir: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        file_template: str | object = pf.MISSING,
        channel_layout: ChannelLayout | object = pf.MISSING,
        **_: Any,
    ) -> WavDataSource:
        sr, ft, layout = self._resolve(sample_rate, file_template, channel_layout)
        return WavDataSource(
            dimensions=_audio_dimensions(sr),
            base_dir=str(base_dir),
            sample_rate=sr,
            file_template=ft,
            channel_layout=layout,
        )

    def generate_source_info(
        self,
        audio_ids: Any,
        base_dir: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        file_template: str | object = pf.MISSING,
        channel_layout: ChannelLayout | object = pf.MISSING,
        source_desc_id: int | None = None,
        **_: Any,
    ) -> pf.DatasetSourceInfo:
        return pf.DatasetSourceInfo(
            source_uri=f"file://{os.path.abspath(str(base_dir))}",
            source_type="wav_files",
            source_name="Audio files",
        )

    def build(
        self,
        audio_ids: Any,
        base_dir: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        file_template: str | object = pf.MISSING,
        channel_layout: ChannelLayout | object = pf.MISSING,
        audio_field: str = AUDIO_FIELD,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> pf.DatasetState:
        sr, ft, _ = self._resolve(sample_rate, file_template, channel_layout)
        ids = tuple(str(audio_id) for audio_id in audio_ids)
        table = pd.DataFrame(index=pd.Index(ids, name="audio_file_id"))
        table[SOURCE_AUDIO_ID_FIELD] = pd.Series(ids, index=table.index, dtype="string")
        table["audio_path"] = pd.Series(
            [os.path.join(str(base_dir), ft.format(item_id=audio_id)) for audio_id in ids],
            index=table.index,
            dtype="string",
        )
        table["sample_rate"] = pd.Series(sr, index=table.index, dtype="Int64")
        table[audio_field] = [
            pf.DataAccessor(
                source_desc_id=source_desc_id,
                item_id=audio_id,
                manager_hint=source_manager,
            )
            for audio_id in ids
        ]

        schema = pf.Schema(
            fields=(
                pf.IndexField(name="audio_file_id"),
                pf.DimensionField.from_dim(_audio_id_dimension(), SOURCE_AUDIO_ID_FIELD, dtype=str),
                pf.ValueField(name="audio_path", dtype=str),
                pf.ValueField(name="sample_rate", dtype=int),
                pf.DataField(name=audio_field),
            )
        )
        return pf.DatasetState(schema=schema, table=table)

    def _resolve(
        self,
        sample_rate: Any,
        file_template: Any,
        channel_layout: Any,
    ) -> tuple[int, str, ChannelLayout]:
        sr = self.resolve_param("sample_rate", sample_rate)
        ft = self.resolve_param("file_template", file_template)
        layout = self.resolve_param("channel_layout", channel_layout)
        if layout not in {"channels_first", "channels_last"}:
            raise ValueError("channel_layout must be either 'channels_first' or 'channels_last'.")
        return sr, ft, layout


class make_audioset_labels(pf.CreationOperator):
    """Build the label/segment side of an AudioSet dataset."""

    sample_rate = pf.Parameter(default=16_000)
    audio_layout = pf.Parameter(default="segments")

    def make_source(
        self,
        metadata_path: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        audio_layout: AudioLayout | object = pf.MISSING,
        **_: Any,
    ) -> None:
        return None

    def generate_source_info(
        self,
        metadata_path: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        audio_layout: AudioLayout | object = pf.MISSING,
        source_desc_id: int | None = None,
        **_: Any,
    ) -> pf.DatasetSourceInfo:
        return pf.DatasetSourceInfo(
            source_uri=f"file://{os.path.abspath(str(metadata_path))}",
            source_type="audioset_csv",
            source_name="AudioSet",
        )

    def build(
        self,
        metadata_path: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        audio_layout: AudioLayout | object = pf.MISSING,
        source_desc_id: int | None = None,
        **_: Any,
    ) -> pf.DatasetState:
        sr, layout = self._resolve(sample_rate, audio_layout)
        time_dim = pf.TemporalDimension(name="time", sample_rate=sr)
        table = _read_audioset_csv(metadata_path)
        table[SOURCE_AUDIO_ID_FIELD] = table["ytid"].astype("string")
        _add_segment_columns(table, layout=layout)
        table.index = _clip_index(table)
        table.index.name = "clip_id"

        schema = pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.DimensionField.from_dim(_audio_id_dimension(), SOURCE_AUDIO_ID_FIELD, dtype=str),
                pf.ValueField(name="start_seconds", dtype=float),
                pf.ValueField(name="end_seconds", dtype=float),
                pf.ValueField(name=LABELS_FIELD, dtype=str),
                pf.DimensionField.from_dim(time_dim, SEGMENT_START_FIELD, dtype=float),
                pf.DimensionField.from_dim(time_dim, SEGMENT_END_FIELD, dtype=float),
            )
        )
        return pf.DatasetState(schema=schema, table=table)

    def _resolve(
        self,
        sample_rate: Any,
        audio_layout: Any,
    ) -> tuple[int, AudioLayout]:
        sr = self.resolve_param("sample_rate", sample_rate)
        layout = self.resolve_param("audio_layout", audio_layout)
        if layout not in {"segments", "full"}:
            raise ValueError("audio_layout must be either 'segments' or 'full'.")
        return sr, layout


class make_audioset(pf.CreationOperator):
    """Convenience wrapper that builds labels, audio files, then composes them."""

    sample_rate = pf.Parameter(default=16_000)
    file_template = pf.Parameter(default="{item_id}.wav")
    audio_layout = pf.Parameter(default="segments")
    channel_layout = pf.Parameter(default="channels_first")

    def __call__(
        self,
        metadata_path: str | Path,
        base_dir: str | Path,
        *,
        sample_rate: int | object = pf.MISSING,
        file_template: str | object = pf.MISSING,
        audio_layout: AudioLayout | object = pf.MISSING,
        channel_layout: ChannelLayout | object = pf.MISSING,
        bind_segments: bool = True,
        materialize_audio: bool = True,
        audio_field: str = AUDIO_FIELD,
        segment_field: str = SEGMENT_FIELD,
    ) -> pf.Dataset:
        sr, ft, layout, channel_layout = self._resolve(
            sample_rate,
            file_template,
            audio_layout,
            channel_layout,
        )
        labels = make_audioset_labels(
            metadata_path,
            sample_rate=sr,
            audio_layout=layout,
        )
        audio_files = make_audio_files(
            _unique_audio_ids(labels),
            base_dir,
            sample_rate=sr,
            file_template=ft,
            channel_layout=channel_layout,
            audio_field=audio_field,
        )
        return merge_audio_labels(
            labels,
            audio_files,
            bind_segments=bind_segments,
            materialize_audio=materialize_audio,
            audio_field=audio_field,
            segment_field=segment_field,
        )

    def generate_source_info(self, *args: Any, **kwargs: Any) -> pf.DatasetSourceInfo:
        raise NotImplementedError("make_audioset dispatches in __call__.")

    def build(self, *args: Any, **kwargs: Any) -> pf.DatasetState:
        raise NotImplementedError("make_audioset dispatches in __call__.")

    def _resolve(
        self,
        sample_rate: Any,
        file_template: Any,
        audio_layout: Any,
        channel_layout: Any,
    ) -> tuple[int, str, AudioLayout, ChannelLayout]:
        sr = self.resolve_param("sample_rate", sample_rate)
        ft = self.resolve_param("file_template", file_template)
        layout = self.resolve_param("audio_layout", audio_layout)
        ch_layout = self.resolve_param("channel_layout", channel_layout)
        if layout not in {"segments", "full"}:
            raise ValueError("audio_layout must be either 'segments' or 'full'.")
        if ch_layout not in {"channels_first", "channels_last"}:
            raise ValueError("channel_layout must be either 'channels_first' or 'channels_last'.")
        return sr, ft, layout, ch_layout


def bind_audio_segments(
    dataset: pf.Dataset,
    *,
    slice_field: str = SEGMENT_FIELD,
    audio_field: str = AUDIO_FIELD,
    start_field: str = SEGMENT_START_FIELD,
    end_field: str = SEGMENT_END_FIELD,
) -> pf.Dataset:
    """Bind AudioSet segment columns so row access returns sliced audio arrays."""
    ds = pf.compose_slice(
        dataset,
        slice_field=slice_field,
        bindings={"time": (start_field, end_field)},
    )
    return pf.slice_data(ds, slice_field=slice_field, data_field=audio_field)


def merge_audio_labels(
    labels: pf.Dataset,
    audio_files: pf.Dataset,
    *,
    how: str = "inner",
    bind_segments: bool = True,
    materialize_audio: bool = True,
    audio_id_field: str = SOURCE_AUDIO_ID_FIELD,
    audio_field: str = AUDIO_FIELD,
    segment_field: str = SEGMENT_FIELD,
) -> pf.Dataset:
    """Compose separately parsed label and audio-file datasets."""
    plan = pf.join(labels, audio_files, on=audio_id_field, how=how)
    merged = pf.merge(
        labels,
        audio_files,
        plan,
        collision=pf.ColumnCollisionStrategy(mode="keep", side="left"),
    )
    clip_id_name = labels.schema.names()[0]
    merged = pf.set_index(merged, "left_index", index_name=clip_id_name)
    if bind_segments:
        merged = bind_audio_segments(
            merged,
            slice_field=segment_field,
            audio_field=audio_field,
        )
    if materialize_audio:
        merged = pf.materialize(merged, audio_field)
    return merged


def parse_labels(value: Any) -> tuple[str, ...]:
    """Parse AudioSet positive label strings into a stable tuple."""
    if value is None or pd.isna(value):
        return ()
    return tuple(part.strip().strip('"') for part in str(value).split(",") if part.strip())


def plot_waveform(
    dataset: pf.Dataset,
    item_id: Any,
    *,
    audio_field: str = AUDIO_FIELD,
    ax: Any = None,
) -> Any:
    """Plot one implicitly sliced audio segment."""
    plt = _pyplot()
    row = dataset[item_id]
    audio = _mono(np.asarray(row[audio_field]))
    sample_rate = int(row["sample_rate"])
    if ax is None:
        _, ax = plt.subplots()
    time = np.arange(audio.shape[-1]) / sample_rate
    ax.plot(time, audio)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(str(item_id))
    return ax


def plot_spectrogram(
    dataset: pf.Dataset,
    item_id: Any,
    *,
    audio_field: str = AUDIO_FIELD,
    n_fft: int = 1024,
    ax: Any = None,
) -> Any:
    """Plot a spectrogram for one implicitly sliced audio segment."""
    plt = _pyplot()
    row = dataset[item_id]
    audio = _mono(np.asarray(row[audio_field]))
    sample_rate = int(row["sample_rate"])
    if ax is None:
        _, ax = plt.subplots()
    ax.specgram(audio, NFFT=n_fft, Fs=sample_rate, noverlap=n_fft // 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(str(item_id))
    return ax


def plot_label_counts(
    dataset: pf.Dataset,
    *,
    labels_field: str = LABELS_FIELD,
    top_n: int = 20,
    ax: Any = None,
) -> Any:
    """Plot the most frequent AudioSet labels in a dataset."""
    plt = _pyplot()
    counts: Counter[str] = Counter()
    for value in dataset.table[labels_field]:
        counts.update(parse_labels(value))
    top = counts.most_common(top_n)
    labels = [label for label, _ in top]
    values = [count for _, count in top]
    if ax is None:
        _, ax = plt.subplots()
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Clips")
    ax.set_title("AudioSet label counts")
    return ax


def pad_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length audio rows into a padded batch."""
    torch = _torch()
    audio_tensors = [
        torch.from_numpy(np.ascontiguousarray(np.atleast_2d(np.asarray(sample[AUDIO_FIELD]))))
        for sample in batch
    ]
    lengths = torch.tensor([tensor.shape[-1] for tensor in audio_tensors], dtype=torch.long)
    channels = max(tensor.shape[0] for tensor in audio_tensors)
    max_length = int(lengths.max().item()) if len(batch) else 0
    audio = audio_tensors[0].new_zeros((len(batch), channels, max_length))
    for i, tensor in enumerate(audio_tensors):
        audio[i, : tensor.shape[0], : tensor.shape[-1]] = tensor
    return {
        "item_id": [
            sample["clip_id"] if "clip_id" in sample else sample["audio_file_id"]
            for sample in batch
        ],
        "audio": audio,
        "lengths": lengths,
        "labels": [sample.get(LABELS_FIELD) for sample in batch],
    }


def make_torch_dataloader(
    dataset: pf.Dataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Any = pad_audio_batch,
    **kwargs: Any,
) -> Any:
    """Create a basic PyTorch DataLoader over patchframe rows.

    ``Dataset.rows()`` is the positional, duck-typed map-style view (len +
    int indexing + batched fetch), so it plugs into ``DataLoader`` directly:
    the sampler's linear indices resolve to row labels inside the view, and
    each item is the evaluated, exited row dict.
    """
    torch = _torch()
    return torch.utils.data.DataLoader(
        dataset.rows(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def _read_audioset_csv(metadata_path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(
        metadata_path,
        comment="#",
        header=None,
        names=list(_AUDIOSET_CSV_COLUMNS),
        skipinitialspace=True,
    )
    table["ytid"] = table["ytid"].astype("string")
    table["start_seconds"] = pd.to_numeric(table["start_seconds"]).astype("Float64")
    table["end_seconds"] = pd.to_numeric(table["end_seconds"]).astype("Float64")
    table[LABELS_FIELD] = table[LABELS_FIELD].astype("string")
    return table


def _add_segment_columns(table: pd.DataFrame, *, layout: AudioLayout) -> None:
    if layout == "segments":
        table[SEGMENT_START_FIELD] = pd.Series(0.0, index=table.index, dtype="Float64")
        table[SEGMENT_END_FIELD] = (
            table["end_seconds"] - table["start_seconds"]
        ).astype("Float64")
        return
    table[SEGMENT_START_FIELD] = table["start_seconds"].astype("Float64")
    table[SEGMENT_END_FIELD] = table["end_seconds"].astype("Float64")


def _audio_dimensions(sample_rate: int) -> pf.Dimensions:
    return pf.Dimensions((pf.TemporalDimension(name="time", sample_rate=sample_rate),))


def _audio_id_dimension() -> pf.CategoricalDimension:
    return pf.CategoricalDimension(name="audio_id")


def _clip_index(table: pd.DataFrame) -> pd.Index:
    ytid = table["ytid"].astype(str)
    if ytid.is_unique:
        return pd.Index(ytid, name="clip_id")
    start = table["start_seconds"].astype(str)
    end = table["end_seconds"].astype(str)
    return pd.Index(
        [
            f"{audio_id}:{start_s}:{end_s}:{i}"
            for i, (audio_id, start_s, end_s) in enumerate(
                zip(ytid, start, end, strict=True)
            )
        ],
        name="clip_id",
    )


def _unique_audio_ids(labels: pf.Dataset) -> tuple[str, ...]:
    values = labels.table[SOURCE_AUDIO_ID_FIELD].astype("string")
    return tuple(str(value) for value in pd.unique(values))


def _mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        return values.mean(axis=0)
    return values


def _soundfile() -> Any:
    try:
        import soundfile as sf
    except ImportError as err:
        raise ImportError("soundfile is required for WavDataSource.") from err
    return sf


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required for AudioSet visualization helpers.") from err
    return plt


def _torch() -> Any:
    try:
        import torch
    except ImportError as err:
        raise ImportError("torch is required for AudioSet DataLoader support.") from err
    return torch
