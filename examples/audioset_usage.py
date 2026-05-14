"""AudioSet usage example.

Run with real AudioSet-style metadata and WAV files:

    python examples/audioset_usage.py --csv balanced_train_segments.csv --wavs wavs/

The script parses labels and audio files as separate datasets, then composes
them with ``join``/``merge`` through ``merge_audio_labels``. Audio is
materialized through a coupling, so row access stays simple. Use
``--audio-layout full`` when WAV files are full source media and CSV start/end
values should be used directly for slicing.
"""

from __future__ import annotations

import argparse

from examples.audioset import (
    SOURCE_AUDIO_ID_FIELD,
    make_audio_files,
    make_audioset_labels,
    make_torch_dataloader,
    merge_audio_labels,
    plot_label_counts,
    plot_spectrogram,
    plot_waveform,
)
from patchframe.ops.builtin.where import where


def main(
    csv_path: str,
    wav_dir: str,
    *,
    label: str | None,
    audio_layout: str,
    plot: bool,
    torch_batch: bool,
) -> None:
    labels = make_audioset_labels(csv_path, audio_layout=audio_layout)
    audio_files = make_audio_files(
        labels.table[SOURCE_AUDIO_ID_FIELD].drop_duplicates(),
        wav_dir,
    )
    ds = merge_audio_labels(labels, audio_files)
    print(f"Loaded {len(ds.table)} clips")
    print(ds.table[["start_seconds", "end_seconds", "labels"]].head())

    if label is not None:
        ds = where(ds, ds.table["labels"].str.contains(label, regex=False))
        print(f"\nAfter label filter ({label!r}): {len(ds.table)} clips")

    if ds.table.empty:
        print("No clips matched.")
        return

    item_id = ds.table.index[0]
    row = ds[item_id]
    print(f"\nFirst row id: {item_id}")
    print(f"Loaded implicit segment shape: {row['audio'].shape} dtype={row['audio'].dtype}")
    print(f"Labels: {row['labels']}")

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        plot_waveform(ds, item_id, ax=axes[0])
        plot_spectrogram(ds, item_id, ax=axes[1])
        plot_label_counts(ds, ax=axes[2], top_n=15)
        fig.tight_layout()
        plt.show()

    if torch_batch:
        loader = make_torch_dataloader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        print("\nTorch batch:")
        print(f"  audio:   {tuple(batch['audio'].shape)}")
        print(f"  lengths: {batch['lengths'].tolist()}")
        print(f"  labels:  {batch['labels'][:2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to AudioSet metadata CSV")
    parser.add_argument("--wavs", required=True, help="Directory containing WAV files")
    parser.add_argument("--label", default="/m/09x0r", help="Optional AudioSet label filter")
    parser.add_argument(
        "--audio-layout",
        choices=("segments", "full"),
        default="segments",
        help="Use 'segments' for pre-extracted WAV clips, 'full' for full media files.",
    )
    parser.add_argument("--plot", action="store_true", help="Show waveform/spectrogram plots")
    parser.add_argument(
        "--torch-batch",
        action="store_true",
        help="Load one padded PyTorch batch using implicit slicing.",
    )
    args = parser.parse_args()
    main(
        args.csv,
        args.wavs,
        label=args.label,
        audio_layout=args.audio_layout,
        plot=args.plot,
        torch_batch=args.torch_batch,
    )
