"""
AudioSet usage example.

Demonstrates the full workflow from dataset creation through filtering to
lazy audio loading with temporal slicing.

Run (requires AudioSet CSVs and WAV files):
    python examples/audioset_usage.py \\
        --csv  /data/audioset/balanced_train_segments.csv \\
        --wavs /data/audioset/wavs/

The script works end-to-end only when the WAV files are present. All steps
up to materialization can be verified without them.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from examples.audioset import make_audioset
from patchframe.data.dimensions import TemporalDimension
from patchframe.ops.builtin.where import where


def main(csv_path: str, wav_dir: str, clip_seconds: float = 3.0) -> None:
    # ------------------------------------------------------------------
    # 1. Create the dataset — no audio is loaded yet.
    # ------------------------------------------------------------------
    ds = make_audioset(csv_path, wav_dir)
    print(f"Loaded {len(ds.table)} clips")
    print(ds.table[["start_seconds", "end_seconds", "labels"]].head())

    # ------------------------------------------------------------------
    # 2. Filter: keep only clips whose duration covers our crop window.
    # ------------------------------------------------------------------
    duration = ds.table["end_seconds"] - ds.table["start_seconds"]
    ds_filtered = where(ds, duration >= clip_seconds)
    print(f"\nAfter duration filter (>= {clip_seconds}s): {len(ds_filtered.table)} clips")

    # ------------------------------------------------------------------
    # 3. Filter: keep a specific label (speech = /m/09x0r).
    # ------------------------------------------------------------------
    speech_label = "/m/09x0r"
    ds_speech = where(ds_filtered, ds_filtered.table["labels"].str.contains(speech_label))
    print(f"After label filter ({speech_label!r}): {len(ds_speech.table)} clips")

    if ds_speech.table.empty:
        print("No clips matched — exiting.")
        return

    # ------------------------------------------------------------------
    # 4. Build a temporal slice spec for [0, clip_seconds).
    # ------------------------------------------------------------------
    time_dim = TemporalDimension(name="time", sample_rate=16000)
    crop_spec = time_dim.spec(0.0, clip_seconds)

    # ------------------------------------------------------------------
    # 5. Inspect the first matching clip without loading audio.
    # ------------------------------------------------------------------
    first_accessor = ds_speech.table["audio"].iloc[0]
    info = first_accessor.inspect(ds_speech.source_manager)
    print(f"\nFirst clip info: {info}")

    # ------------------------------------------------------------------
    # 6. Materialize a cropped clip.
    #    accessor.slice() is lazy — no I/O until .materialize().
    # ------------------------------------------------------------------
    audio = first_accessor.slice(crop_spec).materialize(ds_speech.source_manager)
    assert isinstance(audio, np.ndarray)
    print(f"Loaded audio shape: {audio.shape}  dtype: {audio.dtype}")

    # ------------------------------------------------------------------
    # 7. Iterate over the first few clips and crop each one.
    # ------------------------------------------------------------------
    print("\nFirst 5 crops:")
    for ytid, row in ds_speech.table.head(5).iterrows():
        clip = row["audio"].slice(crop_spec).materialize(ds_speech.source_manager)
        print(f"  {ytid}: {clip.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to AudioSet metadata CSV")
    parser.add_argument("--wavs", required=True, help="Directory containing WAV files")
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    args = parser.parse_args()
    main(args.csv, args.wavs, args.clip_seconds)
