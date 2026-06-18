"""Guards the synthetic multimodal fusion example against regressions.

The example is the Phase-4 forcing function: three modalities with
non-overlapping dimension sets fused per window through couplings, with the
data format/structure/idiosyncrasies modeled on real datasets (SRT captions
with approximate timing, NTSC frame rate, A/V stream duration mismatch).
These tests pin its load-bearing claims — shared-clock windowing across
heterogeneous rates, full-extent defaulting for unmentioned dimensions (the
Phase-5 check), the realism artifacts, identity-alignment carry of plan
columns, per-row laziness on the handle arm, eager-arm/handle-arm parity, and
picklable deferred state.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import patchframe as pf
from patchframe.testing.source_contract import assert_source_contract

from examples.multimodal_fusion import (
    AUDIO_DURATIONS,
    AUDIO_FIELD,
    SAMPLE_FIELD,
    SEGMENTS_FIELD,
    SOURCE_INDEX_FIELD,
    VIDEO_DURATIONS,
    VIDEO_FIELD,
    WINDOW_FIELD,
    RampAudioSource,
    RampVideoSource,
    attach_matched_segments,
    combine_sample,
    effective_durations,
    expected_sample,
    expected_windows,
    fuse_windows,
    make_fusion_clips,
    parse_srt,
    window_clips,
)


def _windows() -> pf.Dataset:
    # The interval join is build-time now: window the clips, then match cues to
    # each window (match -> implode) so every window carries its matched cues.
    return attach_matched_segments(window_clips(make_fusion_clips()))


@pytest.fixture(scope="module")
def samples() -> pf.FieldHandle:
    return fuse_windows(_windows())


def _carrier(samples: pf.FieldHandle) -> pf.Dataset:
    return samples.dataset_context.dataset


def _samples_by_position(samples: pf.FieldHandle) -> list[dict]:
    return [sample for _, sample in samples.items()]


def test_video_source_contract_with_time_only_slice():
    # The Phase-5 check: a time-only slice against a {time, y, x, channel}
    # source must resolve with full extent on the unmentioned dimensions, and
    # the partial read must equal full-read-then-slice.
    assert_source_contract(
        RampVideoSource(durations=VIDEO_DURATIONS),
        item_id="clip_a",
        dim_slice=pf.DimensionedSlice(dims={"time": slice(1.0, 3.0)}),
        compare_partial=True,
    )


def test_audio_source_contract_with_time_only_slice():
    assert_source_contract(
        RampAudioSource(durations=AUDIO_DURATIONS),
        item_id="clip_b",
        dim_slice=pf.DimensionedSlice(dims={"time": slice(0.5, 1.5)}),
        compare_partial=True,
    )


def test_parse_srt_format():
    cues = parse_srt(
        """\
1
00:00:00,640 --> 00:00:01,420
the quick

2
00:01:01,980 --> 00:01:03,310
brown
fox
"""
    )
    # Millisecond-quantized stamps with comma decimals; multi-line cue text
    # joins with a space.
    assert cues == ((0.64, 1.42, "the quick"), (61.98, 63.31, "brown fox"))


def test_av_duration_mismatch_trims_clip_extent():
    # The usable extent is the per-clip minimum across streams; with these
    # constants the trim changes the window count (clip_a loses a window
    # versus its video duration).
    usable = effective_durations()
    assert usable == {"clip_a": 4.98, "clip_b": 3.07, "clip_c": 6.46}
    clips = make_fusion_clips()
    assert clips.table["clip_stop"].tolist() == [4.98, 3.07, 6.46]


def test_window_layout_and_source_traceability(samples: pf.FieldHandle):
    # The window slice and the source_index mapping are plan columns attached
    # by identity alignment (explode inherits the plan's index identity), so
    # every window row knows its clip directly.
    carrier = _carrier(samples)
    layout = expected_windows()
    assert len(carrier.table) == len(layout) == 10
    assert carrier.table[SOURCE_INDEX_FIELD].tolist() == [
        clip_id for clip_id, _, _ in layout
    ]
    assert isinstance(carrier.schema.get(SOURCE_INDEX_FIELD), pf.ForeignIndexField)
    for position, (_, start, stop) in enumerate(layout):
        window = carrier.table[WINDOW_FIELD].iloc[position]
        assert window.dims["time"] == slice(start, stop)


def test_pipeline_is_lazy_until_access(samples: pf.FieldHandle):
    # The handle arm defers everything: the carrier still holds raw full-clip
    # accessors and a null sample column — slice_data / materialize /
    # map_fields only recorded couplings.
    carrier = _carrier(samples)
    assert all(
        isinstance(cell, pf.DataAccessor) for cell in carrier.table[VIDEO_FIELD]
    )
    assert all(
        isinstance(cell, pf.DataAccessor) for cell in carrier.table[AUDIO_FIELD]
    )
    assert carrier.table[SAMPLE_FIELD].isna().all()


def test_fused_samples_match_reference(samples: pf.FieldHandle):
    # Per-row access fuses one window at a time; every sample is exactly the
    # value computed straight from the synthetic content functions, which
    # verifies the seconds -> frame-index and seconds -> sample-index
    # resolution per source as well as the slack-padded caption filtering.
    layout = expected_windows()
    for (_, sample), (clip_id, start, stop) in zip(
        samples.items(), layout, strict=True
    ):
        reference = expected_sample(clip_id, start, stop)
        np.testing.assert_array_equal(sample["video"], reference["video"])
        np.testing.assert_array_equal(sample["audio"], reference["audio"])
        assert sample["transcript"] == reference["transcript"]
        assert sample["window_seconds"] == reference["window_seconds"]
        assert sample["audio"].shape == (32_000, 2)  # 2.0s @ 16 kHz, fixed


def test_ntsc_rate_makes_window_frame_counts_vary(samples: pf.FieldHandle):
    # 2.0s @ 30000/1001 fps is 59.94 frames: index truncation gives the first
    # window of each clip 59 frames and offset windows 60 — equal-duration
    # windows do NOT have equal shapes, so collation cannot assume fixed sizes.
    frame_counts = [
        sample["video"].shape[0] for sample in _samples_by_position(samples)
    ]
    assert frame_counts == [59, 60, 60, 59, 60, 59, 60, 60, 60, 60]


def test_caption_idiosyncrasies_in_fused_text(samples: pf.FieldHandle):
    values = _samples_by_position(samples)
    # Hardcoded against the SRT literals (independent of the shared helpers).
    # clip_a [2,4): slack padding pulls in the lagging 'jumps' cue (4.15s start
    # vs window stop 4.0 + 0.25 slack).
    assert values[2]["transcript"] == "brown fox jumps"
    # clip_b: the '[Music]' annotation cue is stripped; the multi-line cue
    # parsed to one string.
    assert values[3]["transcript"] == "hello world"
    # clip_c [0,2): rolling captions repeat the previous line — the duplicate
    # survives into the fused text (dedup is a downstream concern).
    assert values[5]["transcript"] == "lorem lorem ipsum dolor"
    # clip_c [4,6): the 'sit amet' cue ends past the clip's usable extent but
    # still matches by overlap.
    assert values[9]["transcript"] == "lorem ipsum dolor sit amet"


def test_eager_arm_computes_now_and_matches_handle_collect():
    # The operand-dispatch law: map_fields on the Dataset computes every
    # sample immediately (the coupling stays as the recipe); the handle arm
    # defers until collect. Both arms produce identical samples.
    windows = _windows()
    handle = fuse_windows(windows)
    assert isinstance(handle, pf.FieldHandle)
    assert _carrier(handle).table[SAMPLE_FIELD].isna().all()

    eager = pf.map_fields(
        windows,
        [VIDEO_FIELD, AUDIO_FIELD, SEGMENTS_FIELD, WINDOW_FIELD],
        combine_sample,
        out=SAMPLE_FIELD,
    )
    assert isinstance(eager, pf.Dataset)
    assert not eager.table[SAMPLE_FIELD].isna().any()

    collected = handle.collect()
    for eager_sample, lazy_sample in zip(
        eager.table[SAMPLE_FIELD], collected.table[SAMPLE_FIELD], strict=True
    ):
        np.testing.assert_array_equal(eager_sample["video"], lazy_sample["video"])
        np.testing.assert_array_equal(eager_sample["audio"], lazy_sample["audio"])
        assert eager_sample["transcript"] == lazy_sample["transcript"]


def test_deferred_pipeline_is_picklable():
    # The fusion fn is module-level, so building the whole deferred pipeline
    # must not trip UnpicklableCallWarning, and the recorded coupling state
    # (Materialize + the MapCoupling with its CallSpec; slice_data is eager now
    # and discharged) must round-trip.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pf.UnpicklableCallWarning)
        handle = fuse_windows(_windows())

    couplings = _carrier(handle).state.couplings
    restored = pickle.loads(pickle.dumps(couplings))
    assert isinstance(restored, pf.CouplingSet)
    assert len(restored.couplings) == len(couplings.couplings)
