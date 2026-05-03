"""
AudioSet dataset creation example.

Demonstrates how to build a patchframe Dataset from an external file-backed
source without copying any audio into memory.

Key concepts
------------
- WavDataSource  : a minimal DataSource that reads WAV files on demand.
                   describe() returns the SourceDescriptor used for
                   registration and re-opening.
- make_audioset  : a CreationOperator. make_source() returns a live
                   WavDataSource; build() constructs the DatasetState.
- TemporalDimension (from patchframe.data.dimensions): spec() accepts
                   start/end in seconds; to_index() converts to sample indices.

Usage
-----
    from examples.audioset import make_audioset
    from patchframe.data.dimensions import TemporalDimension

    ds = make_audioset("audioset_balanced_train.csv", "/data/audioset/wavs/")

    # lazy access — nothing is loaded yet
    accessor = ds.table["audio"].iloc[0]

    # full clip
    audio = accessor.materialize()

    # lazy sub-clip: 1.0 s to 3.5 s
    time_dim = TemporalDimension(name="time", sample_rate=16000)
    clipped = accessor.slice(time_dim.spec(1.0, 3.5)).materialize()

Requirements
------------
    pip install soundfile
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from patchframe.data.accessor import DataAccessor
from patchframe.data.descriptor import SourceDescriptor
from patchframe.data.dimensioned_slice import DimensionedSlice
from patchframe.data.dimensions import Dimensions, TemporalDimension
from patchframe.data.source import DataSource
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.fields import DataField, IndexField, ValueField
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import MISSING, CreationOperator, Parameter


# ---------------------------------------------------------------------------
# WavDataSource
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WavDataSource(DataSource):
    """DataSource that reads WAV files on demand."""

    source_type: str = "wav"
    thread_safe: bool = True
    fork_safe: bool = True
    dimensions: Dimensions = field(default_factory=lambda: Dimensions((TemporalDimension(name="time"),)))
    base_dir: str = ""
    sample_rate: int | None = None
    file_template: str = "{item_id}.wav"

    @classmethod
    def open(cls, descriptor: SourceDescriptor) -> "WavDataSource":
        return cls(
            dimensions=descriptor.capabilities.get(
                "dimensions",
                Dimensions((TemporalDimension(name="time"),)),
            ),
            base_dir=descriptor.open_config["base_dir"],
            sample_rate=descriptor.open_config.get("sample_rate"),
            file_template=descriptor.open_config.get("file_template", "{item_id}.wav"),
        )

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_type="wav",
            source_id=f"wav:{os.path.abspath(self.base_dir)}",
            open_config={
                "base_dir": self.base_dir,
                "sample_rate": self.sample_rate,
                "file_template": self.file_template,
            },
            capabilities={"dimensions": self.dimensions},
        )

    def _path_for(self, item_id: Any) -> str:
        return os.path.join(self.base_dir, self.file_template.format(item_id=item_id))

    def materialize(self, accessor: DataAccessor) -> np.ndarray:
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for WavDataSource. Install with: pip install soundfile")
        audio, _ = sf.read(self._path_for(accessor.item_id), dtype="float32", always_2d=False)
        if accessor.dimensioned_slice is not None:
            resolved = self.dimensions.resolve(accessor.dimensioned_slice)
            audio = audio[tuple(di.value for di in resolved)]
        return audio

    def inspect(self, accessor: DataAccessor) -> Mapping[str, Any]:
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for WavDataSource. Install with: pip install soundfile")
        info = sf.info(self._path_for(accessor.item_id))
        return {
            "frames": info.frames,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "item_id": accessor.item_id,
            "dimensioned_slice": accessor.dimensioned_slice,
        }

    def slice_accessor(self, accessor: DataAccessor, dim_slice: DimensionedSlice) -> DataAccessor:
        unknown = set(dim_slice.dims) - set(self.dimensions.names())
        if unknown:
            raise ValueError(f"DimensionedSlice references unknown dimensions: {sorted(unknown)}")
        return accessor.slice(dim_slice)


# ---------------------------------------------------------------------------
# make_audioset
# ---------------------------------------------------------------------------

_AUDIOSET_CSV_COLUMNS = ["ytid", "start_seconds", "end_seconds", "labels"]


class make_audioset(CreationOperator):
    """Build an AudioSet dataset from a metadata CSV and a WAV file directory.

    Parameters
    ----------
    metadata_path:
        Path to the AudioSet metadata CSV. Lines starting with ``#`` are
        treated as comments. Expected columns (no header in file):
        ytid, start_seconds, end_seconds, positive_labels.
    base_dir:
        Directory containing one WAV file per clip.
    sample_rate:
        Expected sample rate of the WAV files. Defaults to 16000.
    file_template:
        Filename pattern. ``{item_id}`` is replaced with the ytid.
        Defaults to ``"{item_id}.wav"``.
    """

    sample_rate = Parameter(default=16000)
    file_template = Parameter(default="{item_id}.wav")

    def _resolve(self, sample_rate: Any, file_template: Any) -> tuple[int, str]:
        return (
            self.resolve_param("sample_rate", sample_rate),
            self.resolve_param("file_template", file_template),
        )

    def make_source(
        self,
        metadata_path: str | Path,
        base_dir: str | Path,
        *,
        sample_rate: int | object = MISSING,
        file_template: str | object = MISSING,
        **_: Any,
    ) -> WavDataSource:
        sr, ft = self._resolve(sample_rate, file_template)
        return WavDataSource(
            dimensions=Dimensions((TemporalDimension(name="time", sample_rate=sr),)),
            base_dir=str(base_dir),
            sample_rate=sr,
            file_template=ft,
        )

    def generate_source_info(
        self,
        metadata_path: str | Path,
        base_dir: str | Path,
        *,
        sample_rate: int | object = MISSING,
        file_template: str | object = MISSING,
        source_desc_id: int | None = None,
        **_: Any,
    ) -> DatasetSourceInfo:
        return DatasetSourceInfo(
            source_uri=f"file://{os.path.abspath(str(metadata_path))}",
            source_type="audioset_csv",
            source_name="AudioSet",
        )

    def build(
        self,
        metadata_path: str | Path,
        base_dir: str | Path,
        *,
        sample_rate: int | object = MISSING,
        file_template: str | object = MISSING,
        source_desc_id: int | None = None,
        **_: Any,
    ) -> DatasetState:
        df = pd.read_csv(
            metadata_path,
            comment="#",
            header=None,
            names=_AUDIOSET_CSV_COLUMNS,
            skipinitialspace=True,
        ).set_index("ytid")

        df["audio"] = [
            DataAccessor(source_desc_id=source_desc_id, item_id=ytid)
            for ytid in df.index
        ]

        schema = Schema(fields=(
            IndexField(name="ytid"),
            ValueField(name="start_seconds", dtype=float),
            ValueField(name="end_seconds", dtype=float),
            ValueField(name="labels", dtype=str),
            DataField(name="audio"),
        ))

        return DatasetState(schema=schema, table=df)
