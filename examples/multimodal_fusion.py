"""Synthetic multimodal fusion: video + audio + transcript over one clock.

Three modalities with genuinely non-overlapping dimension sets, sharing only
the clip index and a seconds-valued ``time`` dimension:

- video        ``{time, y, x, channel}``   dense frames @ ~29.97 fps (NTSC)
- audio        ``{time, audio_channel}``   dense waveform @ 16 kHz
- transcript   ``{time}``                  sparse SRT caption cues

Time is represented three ways (frame index / sample index / continuous
interval), and two alignment styles fall out — that is the lesson:

1. video<->audio — *shared-clock dense windowing*. Windows are planned once in
   seconds; the same time-only ``DimensionedSlice`` is bound to both data
   fields, and each source's ``Dimensions.resolve`` turns seconds into its own
   native indices (frames vs samples). No join is needed — the natural-unit ->
   backend-index abstraction does the rate reconciliation, and dimensions the
   slice does not mention (y/x/channel) default to full extent.

2. transcript<->window — *interval overlap*. Caption cues live at arbitrary
   times. Grouping cues under their clip is the resolved partition/aggregate
   fork (partition-aggregate.md): ``attach_transcript_segments`` types the
   clip reference (``link``) and groups cue rows into per-clip fibers
   (``partition``), attached by identity alignment. Matching cues to
   *windows* remains a genuine dimensional/interval join, done inside the
   fuse function (``combine_sample``) — the deliberate marker for where the
   real interval-join operator belongs (roadmap #3).

The array *values* are synthetic (ramp-valued: every video pixel equals its
absolute frame index, every audio sample its absolute sample index, plus a
per-clip offset — so every fused sample is exactly verifiable), but the data
**format, structure, and idiosyncrasies are modeled on real datasets**:

- Transcripts arrive as literal **SRT text** (index lines, ``HH:MM:SS,mmm``
  stamps, multi-line cues) and must be parsed. Caption timing is approximate,
  the way auto-captions are: cues lag speech onset by a few hundred ms,
  rolling cues overlap and repeat the previous line, a cue runs past the end
  of the clip, and non-speech annotations (``[Music]``) are interleaved.
  Window matching therefore pads by ``CAPTION_SLACK_SECONDS`` and strips
  annotation cues — and fused text can still contain rolling-caption
  duplicates, which is a downstream concern, exactly as in real pipelines.
- Video runs at the NTSC rate **30000/1001 ≈ 29.97 fps**, so equal-length
  windows do **not** contain equal frame counts (59 vs 60 frames per 2 s
  window, by index truncation) — batch collation cannot assume fixed shapes.
  ``TemporalDimension.sample_rate`` is ``int | float | None``, so the rational
  rate is first-class; the seconds→index truncation policy is still implicit
  in ``to_index`` (an open boundary-rounding question).
- The audio and video streams of a clip report **different durations** (as
  real containers do); the clip's usable extent is trimmed to the shared
  minimum before windowing.

Everything is dependency-light and deterministic, so the pipeline is
CI-runnable:

    python examples/multimodal_fusion.py
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import patchframe as pf

#: Per-modality stream durations in seconds. Real containers routinely report
#: slightly different lengths for the audio and video streams of one clip.
VIDEO_DURATIONS: dict[str, float] = {"clip_a": 5.04, "clip_b": 3.07, "clip_c": 6.5}
AUDIO_DURATIONS: dict[str, float] = {"clip_a": 4.98, "clip_b": 3.12, "clip_c": 6.46}

#: NTSC frame rate. Deliberately not an integer: real video is 30000/1001 or
#: 24000/1001 far more often than a clean 30. TemporalDimension.sample_rate is
#: int | float | None, so the rational rate is a first-class sampling.
VIDEO_FPS = 30000 / 1001
FRAME_HEIGHT = 6
FRAME_WIDTH = 8
FRAME_CHANNELS = 3
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 2

WINDOW_SECONDS = 2.0
WINDOW_STRIDE_SECONDS = 1.0

#: Caption timestamps are approximate (auto-captions lag speech onset by
#: hundreds of ms), so window membership is tested against a padded window
#: rather than the exact interval.
CAPTION_SLACK_SECONDS = 0.25

VIDEO_FIELD = "video"
AUDIO_FIELD = "audio"
SEGMENTS_FIELD = "segments"
SEG_SPAN_FIELD = "seg_span"  # the cue interval composed into a time-slice for the join
WINDOW_FIELD = "window"
SAMPLE_FIELD = "sample"
CLIP_START_FIELD = "clip_start"
CLIP_STOP_FIELD = "clip_stop"
PLAN_INDEX_FIELD = "plan_id"
SOURCE_INDEX_FIELD = "source_index"

#: Transcripts in the format they actually arrive in: SRT blocks. The cue
#: structure models real auto-caption artifacts (see the parenthetical notes):
TRANSCRIPTS_SRT: dict[str, str] = {
    # Cues lag true speech onset by ~0.3-0.6s; the last cue runs past the end
    # of the clip (4.98s usable extent).
    "clip_a": """\
1
00:00:00,640 --> 00:00:01,420
the quick

2
00:00:01,980 --> 00:00:03,310
brown fox

3
00:00:04,150 --> 00:00:05,260
jumps
""",
    # A non-speech annotation cue overlapping the speech cue, and a cue whose
    # text spans two lines — both standard SRT.
    "clip_b": """\
1
00:00:00,000 --> 00:00:02,800
[Music]

2
00:00:00,330 --> 00:00:02,710
hello
world
""",
    # Rolling captions: cue 2 starts before cue 1 ends and repeats its text —
    # the classic auto-caption artifact. Cue 3 ends past the clip.
    "clip_c": """\
1
00:00:01,070 --> 00:00:02,140
lorem

2
00:00:02,020 --> 00:00:04,560
lorem ipsum dolor

3
00:00:05,900 --> 00:00:06,840
sit amet
""",
}


def parse_srt(srt: str) -> tuple[tuple[float, float, str], ...]:
    """Parse SRT text into (start_seconds, end_seconds, text) cues.

    Handles the format as it arrives: blank-line-separated blocks of
    ``index`` / ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` / one-or-more text lines
    (joined with a space). Timestamps are millisecond-quantized by format.
    """

    cues = []
    for block in srt.strip().split("\n\n"):
        lines = [line.strip() for line in block.strip().splitlines()]
        if len(lines) < 3:
            raise ValueError(f"Malformed SRT block: {block!r}")
        start_stamp, separator, end_stamp = lines[1].partition(" --> ")
        if not separator:
            raise ValueError(f"Malformed SRT timing line: {lines[1]!r}")
        cues.append(
            (_srt_seconds(start_stamp), _srt_seconds(end_stamp), " ".join(lines[2:]))
        )
    return tuple(cues)


def _srt_seconds(stamp: str) -> float:
    """``HH:MM:SS,mmm`` -> seconds (note the comma decimal separator)."""

    clock, _, millis = stamp.strip().partition(",")
    hours, minutes, seconds = clock.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def effective_durations(
    video_durations: Mapping[str, float] = VIDEO_DURATIONS,
    audio_durations: Mapping[str, float] = AUDIO_DURATIONS,
) -> dict[str, float]:
    """Usable clip extent: the per-clip minimum across streams.

    The standard preprocessing idiom for containers whose streams disagree —
    windows are planned only over time that every modality can serve.
    """

    return {
        clip_id: min(video_durations[clip_id], audio_durations[clip_id])
        for clip_id in video_durations
    }


def _clock_dimension() -> pf.TemporalDimension:
    """The shared clock: seconds-valued time, independent of any modality rate.

    A **continuous** axis (``sample_rate=None``) — it is the comparison frame,
    never a sampled source. Windows are planned against it in seconds; each
    modality's source carries its *own* ``time`` dimension at its native rate
    and converts the same seconds-valued slice to frame/sample indices at
    resolve time. (Previously this carried a fake ``sample_rate=1`` only to
    satisfy a required field; the honest shape is no sampling at all.)
    """

    return pf.TemporalDimension(name="time", sample_rate=None)


def clip_offset(item_id: Any, durations: Mapping[str, float]) -> float:
    """Per-clip value offset so different clips are numerically distinguishable."""

    return 1000.0 * list(durations).index(item_id)


def video_frames(
    item_id: Any,
    durations: Mapping[str, float],
    start_frame: int,
    stop_frame: int,
) -> np.ndarray:
    """Frames ``[start_frame, stop_frame)``: every pixel of frame f equals offset+f."""

    values = clip_offset(item_id, durations) + np.arange(
        start_frame, stop_frame, dtype=np.float32
    )
    shape = (stop_frame - start_frame, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
    return np.broadcast_to(values[:, None, None, None], shape).copy()


def audio_samples(
    item_id: Any,
    durations: Mapping[str, float],
    start_sample: int,
    stop_sample: int,
) -> np.ndarray:
    """Samples ``[start_sample, stop_sample)``: value = offset + sample + channel/10."""

    base = clip_offset(item_id, durations) + np.arange(
        start_sample, stop_sample, dtype=np.float32
    )
    channels = np.arange(AUDIO_CHANNELS, dtype=np.float32) / 10.0
    return base[:, None] + channels[None, :]


class RampVideoSource(pf.ArrayDataSource):
    """Synthetic video decoder: dims (time @ ~29.97 fps, y, x, channel).

    ``read_partial`` generates only the requested frame range — the synthetic
    stand-in for "decode only the windowed frames". The window slice mentions
    only ``time``; y/x/channel arrive as full-extent selectors from
    ``Dimensions.resolve``. At the NTSC rate, seconds -> frame-index
    conversion truncates, so equal-length windows yield 59 or 60 frames.
    """

    source_type = "synthetic_video"
    thread_safe: bool = True
    fork_safe: bool = True
    config_fields = ("durations", "fps")
    supports_partial_read = True

    def __init__(
        self,
        *,
        durations: Mapping[str, float],
        fps: float = VIDEO_FPS,
        dimensions: pf.Dimensions | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions
            or pf.Dimensions(
                (
                    pf.TemporalDimension(name="time", sample_rate=fps),
                    pf.IndexDimension(name="y"),
                    pf.IndexDimension(name="x"),
                    pf.IndexDimension(name="channel"),
                )
            ),
            source_id=source_id,
            durations=dict(durations),
            fps=fps,
        )

    def read_full(self, item_id: Any, accessor: pf.DataAccessor) -> np.ndarray:
        return video_frames(item_id, self.durations, 0, self._frame_count(item_id))

    def read_partial(
        self,
        item_id: Any,
        resolved_slice: pf.ResolvedSlice,
        accessor: pf.DataAccessor,
    ) -> np.ndarray:
        start, stop = _bounded(resolved_slice.get("time"), self._frame_count(item_id))
        block = video_frames(item_id, self.durations, start, stop)
        spatial = tuple(
            resolved_slice.get(name, slice(None)) for name in ("y", "x", "channel")
        )
        return block[(slice(None), *spatial)]

    def extent_for(self, item_id: Any) -> pf.DimensionedSlice:
        return pf.DimensionedSlice(
            dims={
                "time": slice(0.0, self.durations[item_id]),
                "y": slice(0, FRAME_HEIGHT),
                "x": slice(0, FRAME_WIDTH),
                "channel": slice(0, FRAME_CHANNELS),
            }
        )

    def _frame_count(self, item_id: Any) -> int:
        return int(round(self.durations[item_id] * self.fps))


class RampAudioSource(pf.ArrayDataSource):
    """Synthetic waveform reader: dims (time @ 16 kHz, audio_channel)."""

    source_type = "synthetic_audio"
    thread_safe: bool = True
    fork_safe: bool = True
    config_fields = ("durations", "sample_rate")
    supports_partial_read = True

    def __init__(
        self,
        *,
        durations: Mapping[str, float],
        sample_rate: int = AUDIO_SAMPLE_RATE,
        dimensions: pf.Dimensions | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions
            or pf.Dimensions(
                (
                    pf.TemporalDimension(name="time", sample_rate=sample_rate),
                    pf.IndexDimension(name="audio_channel"),
                )
            ),
            source_id=source_id,
            durations=dict(durations),
            sample_rate=sample_rate,
        )

    def read_full(self, item_id: Any, accessor: pf.DataAccessor) -> np.ndarray:
        return audio_samples(item_id, self.durations, 0, self._sample_count(item_id))

    def read_partial(
        self,
        item_id: Any,
        resolved_slice: pf.ResolvedSlice,
        accessor: pf.DataAccessor,
    ) -> np.ndarray:
        start, stop = _bounded(resolved_slice.get("time"), self._sample_count(item_id))
        block = audio_samples(item_id, self.durations, start, stop)
        return block[(slice(None), resolved_slice.get("audio_channel", slice(None)))]

    def extent_for(self, item_id: Any) -> pf.DimensionedSlice:
        return pf.DimensionedSlice(
            dims={
                "time": slice(0.0, self.durations[item_id]),
                "audio_channel": slice(0, AUDIO_CHANNELS),
            }
        )

    def _sample_count(self, item_id: Any) -> int:
        return int(round(self.durations[item_id] * self.sample_rate))


def _bounded(value: Any, full_stop: int) -> tuple[int, int]:
    """Bound a resolved time selector (an index slice, possibly open) to [0, full_stop)."""

    if not isinstance(value, slice):
        raise TypeError(f"Synthetic sources expect slice time selectors; got {value!r}.")
    start = 0 if value.start is None else int(value.start)
    stop = full_stop if value.stop is None else int(value.stop)
    return start, stop


class make_video_clips(pf.CreationOperator):
    """Video side: lazy frame accessors plus the reported stream duration."""

    def make_source(self, durations: Mapping[str, float], **_: Any) -> RampVideoSource:
        return RampVideoSource(durations=durations)

    def generate_source_info(
        self, durations: Mapping[str, float], **_: Any
    ) -> pf.DatasetSourceInfo:
        return pf.DatasetSourceInfo(
            source_uri="synthetic://video",
            source_type="synthetic_video",
            source_name="Synthetic ramp video",
        )

    def build(
        self,
        durations: Mapping[str, float],
        *,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> pf.DatasetState:
        ids = list(durations)
        index = pd.Index(ids, name="clip_id")
        table = pd.DataFrame(index=index)
        table["video_duration"] = pd.Series(
            [durations[clip_id] for clip_id in ids], index=index, dtype="Float64"
        )
        table[VIDEO_FIELD] = [
            pf.DataAccessor(
                source_desc_id=source_desc_id,
                item_id=clip_id,
                manager_hint=source_manager,
            )
            for clip_id in ids
        ]
        schema = pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.ValueField(name="video_duration", dtype=float),
                pf.DataField(name=VIDEO_FIELD),
            )
        )
        return pf.DatasetState(schema=schema, table=table)


class make_audio_clips(pf.CreationOperator):
    """Audio side: lazy waveform accessors over the same clip ids."""

    def make_source(self, durations: Mapping[str, float], **_: Any) -> RampAudioSource:
        return RampAudioSource(durations=durations)

    def generate_source_info(
        self, durations: Mapping[str, float], **_: Any
    ) -> pf.DatasetSourceInfo:
        return pf.DatasetSourceInfo(
            source_uri="synthetic://audio",
            source_type="synthetic_audio",
            source_name="Synthetic ramp audio",
        )

    def build(
        self,
        durations: Mapping[str, float],
        *,
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> pf.DatasetState:
        ids = list(durations)
        index = pd.Index(ids, name="clip_id")
        table = pd.DataFrame(index=index)
        table["audio_duration"] = pd.Series(
            [durations[clip_id] for clip_id in ids], index=index, dtype="Float64"
        )
        table[AUDIO_FIELD] = [
            pf.DataAccessor(
                source_desc_id=source_desc_id,
                item_id=clip_id,
                manager_hint=source_manager,
            )
            for clip_id in ids
        ]
        schema = pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.ValueField(name="audio_duration", dtype=float),
                pf.DataField(name=AUDIO_FIELD),
            )
        )
        return pf.DatasetState(schema=schema, table=table)


def make_transcript(
    transcripts_srt: Mapping[str, str] = TRANSCRIPTS_SRT,
) -> pf.Dataset:
    """Transcript side: one row per parsed SRT cue — sparse in time.

    Parses the raw SRT (format idiosyncrasies included) into cue rows. The
    timestamps stay approximate; nothing downstream may assume they bracket
    the speech exactly.
    """

    clock = _clock_dimension()
    rows = [
        (clip_id, start, end, text)
        for clip_id, srt in transcripts_srt.items()
        for start, end, text in parse_srt(srt)
    ]
    index = pd.Index(
        [f"seg_{position:03d}" for position in range(len(rows))], name="segment_id"
    )
    table = pd.DataFrame(
        {
            "clip_id": pd.Series([r[0] for r in rows], index=index, dtype="string"),
            "seg_start": pd.Series([r[1] for r in rows], index=index, dtype="Float64"),
            "seg_end": pd.Series([r[2] for r in rows], index=index, dtype="Float64"),
            "text": pd.Series([r[3] for r in rows], index=index, dtype="string"),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="segment_id"),
            pf.ValueField(name="clip_id", dtype=str),
            pf.DimensionField.from_dim(clock, "seg_start", dtype=float),
            pf.DimensionField.from_dim(clock, "seg_end", dtype=float),
            pf.ValueField(name="text", dtype=str),
        )
    )
    return pf.make_from_dataframe(table, schema)


def attach_transcript_segments(clips: pf.Dataset, transcript: pf.Dataset) -> pf.Dataset:
    """Group transcript cues *by clip* into per-clip fibers (the aggregation arm).

    A standalone illustration of ``partition``'s aggregation arm, distinct from
    the window-level interval *join* (``attach_matched_segments``): ``link``
    types ``clip_id`` as a reference into the clips namespace, ``partition``
    groups cue rows into per-clip sub-datasets (``domain=clips`` makes the base
    total — a clip with no cues gets an *empty fiber*), and the fiber attaches
    by identity alignment (the base inherits the clips identity, so
    ``concat_columns`` needs no collision strategy). Grouping one dataset by a
    key is ``partition``; matching two datasets by overlapping intervals is the
    join — the two are different operations, shown side by side.
    """

    transcript = pf.link(transcript, clips, "clip_id")
    groups = pf.partition(transcript, "clip_id", domain=clips, into=SEGMENTS_FIELD)
    return pf.concat_columns(clips, pf.keep(groups, [SEGMENTS_FIELD]))


def make_fusion_clips(
    video_durations: Mapping[str, float] = VIDEO_DURATIONS,
    audio_durations: Mapping[str, float] = AUDIO_DURATIONS,
) -> pf.Dataset:
    """Compose the dense modalities into one clip-indexed dataset.

    Video and audio are separate sources (each ``make_*`` registers its own
    ``DataSource``); ``concat_columns`` aligns them on the shared clip index, so
    the composed dataset carries accessors into two different sources side by
    side. The clip's usable clock extent is the per-clip minimum of the two
    stream durations (real containers disagree). The sparse transcript is
    *not* attached here — it is matched to windows by interval join after
    windowing (see ``attach_matched_segments``).
    """

    video = make_video_clips(video_durations)
    audio = make_audio_clips(audio_durations)
    # Both sides carry the same-named clip_id IndexField, which concat_columns
    # treats as an ordinary field collision — aligning same-index datasets
    # currently needs an explicit keep strategy for the index field.
    clips = pf.concat_columns(
        video,
        audio,
        collision=pf.ColumnCollisionStrategy(mode="keep", side="left"),
    )

    clock = _clock_dimension()
    usable = effective_durations(video_durations, audio_durations)
    # Pandas-style column assignment lives on the mutable cursor (Dataset
    # itself is immutable): each statement desugars to the assign operator and
    # advances the cursor. A Field key carries its own name, so a typed
    # assignment states the name once.
    ctx = clips.context()
    ctx[pf.DimensionField.from_dim(clock, CLIP_START_FIELD, dtype=float)] = [
        0.0
    ] * len(clips.table)
    ctx[pf.DimensionField.from_dim(clock, CLIP_STOP_FIELD, dtype=float)] = [
        usable[clip_id] for clip_id in clips.table.index
    ]
    return ctx.dataset


def window_clips(
    clips: pf.Dataset,
    *,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_STRIDE_SECONDS,
) -> pf.Dataset:
    """Tile every clip into windows and bind lazy windowed reads to row access.

    The plan is built once, in seconds, from the clock-dimension clip bounds
    (already trimmed to the shared A/V extent). ``explode`` is a pure gather;
    the plan's own columns — the ``window`` slice and the ``source_index``
    mapping back to the clip — attach afterwards by **identity alignment**
    (the exploded rows inherit the plan's index identity, so ``concat_columns``
    needs no collision strategy; join-dimensions-identity.md §5). The same
    time-only window slice is then bound to *both* data fields: each source
    resolves seconds to its native indices at materialization. Nothing is
    decoded here — slice_data/materialize only record couplings.
    """

    plan = pf.window_expansion_plan(
        clips,
        bindings={"time": (CLIP_START_FIELD, CLIP_STOP_FIELD)},
        windows={"time": pf.AxisWindow(window_seconds, stride_seconds)},
        slice_field=WINDOW_FIELD,
    )
    windows = pf.explode(clips, plan)
    windows = pf.concat_columns(
        windows,
        pf.keep(plan, [PLAN_INDEX_FIELD, SOURCE_INDEX_FIELD, WINDOW_FIELD]),
    )
    windows = pf.slice_data(windows, slice_field=WINDOW_FIELD, data_field=VIDEO_FIELD)
    windows = pf.slice_data(windows, slice_field=WINDOW_FIELD, data_field=AUDIO_FIELD)
    return pf.materialize(windows, (VIDEO_FIELD, AUDIO_FIELD))


def attach_matched_segments(
    windows: pf.Dataset,
    transcript: pf.Dataset = None,  # type: ignore[assignment]
) -> pf.Dataset:
    """Match transcript cues to windows by clip + time overlap (the real join).

    This is the interval join the example was built to motivate — it replaces
    the per-window interval filter that used to live inside ``combine_sample``.
    The matching is now *build-time*, dimensional, and declarative:

    1. ``compose_slice`` composes each cue's ``(seg_start, seg_end)`` into a
       seconds-valued ``time`` slice (the interval-predicate operand); ``consume``
       materializes it so the join can read it.
    2. ``match`` correlates windows and cues on the **clip scope** plus a
       **padded time overlap** — the caption slack is now predicate vocabulary
       (``overlap(pad=...)``), not hand-rolled Python — emitting a window↔cue
       correspondence plan.
    3. ``implode`` collapses that correspondence into one **fiber of matched
       cues per window** (``domain=windows`` makes it total — a window with no
       captions gets an empty fiber). The fiber attaches by identity alignment.

    A cue overlapping two windows lands in both (``implode`` replicates); the
    overlap test is symmetric with the old window-padded filter, so the fused
    text is unchanged — only the interval reasoning moved out of the fuse fn.
    """

    if transcript is None:
        transcript = make_transcript()
    transcript = pf.compose_slice(
        transcript,
        slice_field=SEG_SPAN_FIELD,
        bindings={"time": ("seg_start", "seg_end")},
    )
    transcript = pf.consume(transcript, SEG_SPAN_FIELD)

    correspondence = pf.match(
        windows,
        transcript,
        on=[(SOURCE_INDEX_FIELD, "clip_id")],
        predicates={"time": pf.overlap(pad=CAPTION_SLACK_SECONDS)},
    )
    groups = pf.implode(transcript, correspondence, windows, into=SEGMENTS_FIELD)
    return pf.concat_columns(windows, pf.keep(groups, [SEGMENTS_FIELD]))


def captions_for_window(
    segments: tuple[tuple[float, float, str], ...],
    start_seconds: float,
    stop_seconds: float,
    *,
    slack_seconds: float = CAPTION_SLACK_SECONDS,
) -> tuple[str, ...]:
    """Caption texts overlapping the padded window, annotations stripped.

    The consumer-side idiom for approximate caption timing: pad the window by
    ``slack_seconds`` before the overlap test (cue stamps lag speech), and
    drop non-speech annotation cues (``[Music]``). Rolling captions can still
    contribute repeated text — deduplication is a downstream choice, not done
    here.
    """

    lo = start_seconds - slack_seconds
    hi = stop_seconds + slack_seconds
    return tuple(
        text
        for start, end, text in segments
        if start < hi and end > lo and not _is_annotation(text)
    )


def _is_annotation(text: str) -> bool:
    return text.startswith("[") and text.endswith("]")


def speech_texts(segments: pf.Dataset) -> tuple[str, ...]:
    """Caption texts of a window's matched cues, non-speech annotations stripped.

    The cues arrive already interval-matched (``attach_matched_segments`` did
    the overlap), so the only consumer-side filter left is *content*: drop
    ``[Music]``-style annotations. Rolling captions still contribute repeated
    text — the fiber preserves cue (SRT) order, and deduplication is a
    downstream choice, not done here.
    """

    return tuple(
        str(text) for text in segments.table["text"] if not _is_annotation(str(text))
    )


def cue_count(segments: pf.Dataset) -> int:
    """Cues per clip — the aggregation pattern: ``map_fields`` over fibers."""

    return len(segments.table)


def combine_sample(
    video: Any,
    audio: Any,
    segments: pf.Dataset,
    window: pf.DimensionedSlice,
) -> dict[str, Any]:
    """Fuse one window row into a training sample (module-level, so picklable).

    By the time this runs, the coupling engine has already ordered
    slice -> materialize -> map for the row, so ``video``/``audio`` arrive as
    windowed arrays and ``segments`` as the window's **matched** cue fiber —
    the interval join (``attach_matched_segments``) already selected the cues
    overlapping this window with caption slack. The only work left here is
    content: strip non-speech annotations and join the text.
    """

    interval = window.dims["time"]
    return {
        "video": np.asarray(video),
        "audio": np.asarray(audio),
        "transcript": " ".join(speech_texts(segments)),
        "window_seconds": (float(interval.start), float(interval.stop)),
    }


def fuse_windows(windows: pf.Dataset) -> pf.FieldHandle:
    """Record the per-row fusion as a MapCoupling; nothing runs until access.

    Deferral is opt-in via the handle arm (the operand-dispatch law):
    ``map_fields`` on a field *selection* records the coupling and returns a
    chaining ``FieldHandle``; ``map_fields`` on the ``Dataset`` itself would
    compute every sample immediately. A training pipeline wants the handle —
    samples materialize one row at a time through ``.items()``/``.loc``.
    """

    return pf.map_fields(
        windows.fields([VIDEO_FIELD, AUDIO_FIELD, SEGMENTS_FIELD, WINDOW_FIELD]),
        combine_sample,
        out=SAMPLE_FIELD,
    )


def expected_sample(
    clip_id: str,
    start_seconds: float,
    stop_seconds: float,
    *,
    video_durations: Mapping[str, float] = VIDEO_DURATIONS,
    audio_durations: Mapping[str, float] = AUDIO_DURATIONS,
    transcripts_srt: Mapping[str, str] = TRANSCRIPTS_SRT,
) -> dict[str, Any]:
    """Reference fused sample, computed directly from the synthetic functions.

    Mirrors the per-source seconds -> native-index conversion: truncation via
    ``int()``, per ``TemporalDimension.to_index`` — at the NTSC video rate this
    is what makes window frame counts vary (59 vs 60).
    """

    segments = parse_srt(transcripts_srt[clip_id])
    return {
        "video": video_frames(
            clip_id,
            video_durations,
            int(start_seconds * VIDEO_FPS),
            int(stop_seconds * VIDEO_FPS),
        ),
        "audio": audio_samples(
            clip_id,
            audio_durations,
            int(start_seconds * AUDIO_SAMPLE_RATE),
            int(stop_seconds * AUDIO_SAMPLE_RATE),
        ),
        "transcript": " ".join(
            captions_for_window(segments, start_seconds, stop_seconds)
        ),
        "window_seconds": (start_seconds, stop_seconds),
    }


def expected_windows(
    durations: Mapping[str, float] | None = None,
    *,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_STRIDE_SECONDS,
) -> tuple[tuple[str, float, float], ...]:
    """(clip_id, start, stop) per plan row, in plan order.

    The window rows carry their clip id directly (``source_index``, attached
    by identity alignment in ``window_clips``); this helper exists to pair
    each row with its expected start/stop seconds for verification.
    """

    durations = effective_durations() if durations is None else dict(durations)
    rows = []
    for clip_id, duration in durations.items():
        count = int((duration - window_seconds) // stride_seconds) + 1
        for position in range(count):
            start = position * stride_seconds
            rows.append((clip_id, start, start + window_seconds))
    return tuple(rows)


def main() -> None:
    clips = make_fusion_clips()
    transcript = make_transcript()

    print("=== clips: two dense array sources; transcript grouped by clip ===")
    print("(audio/video stream durations disagree; clip_stop is the trimmed extent)")
    # partition's *aggregation* arm: group the sparse transcript by clip and
    # count cues per clip — distinct from the window-level interval *join* below.
    clip_groups = attach_transcript_segments(clips, transcript)
    overview = pf.map_fields(clip_groups, [SEGMENTS_FIELD], cue_count, out="n_cues")
    print(
        overview.table[
            ["video_duration", "audio_duration", CLIP_STOP_FIELD, "n_cues"]
        ].to_string()
    )

    # The interval *join*: tile clips into windows, then match transcript cues
    # to each window by clip + padded time overlap (match -> implode), so every
    # window carries its own matched cues.
    windows = attach_matched_segments(window_clips(clips), transcript)
    samples = fuse_windows(windows)  # the lazy handle arm: nothing runs yet
    layout = expected_windows()
    print(
        f"\nPlanned {len(windows.table)} windows of {WINDOW_SECONDS}s "
        f"(stride {WINDOW_STRIDE_SECONDS}s) across {len(clips.table)} clips, "
        f"each carrying its interval-matched cues"
    )

    # Laziness: the carrier still holds raw accessors and an all-null sample
    # column — the pipeline so far has only recorded couplings.
    assert isinstance(samples, pf.FieldHandle)
    carrier = samples.dataset_context.dataset
    assert isinstance(carrier.table[VIDEO_FIELD].iloc[0], pf.DataAccessor)
    assert carrier.table[SAMPLE_FIELD].isna().all()

    print("\n=== one sample, loaded through coupling-aware row access ===")
    window_id = carrier.table.index[1]  # clip_a, seconds [1.0, 3.0)
    sample = samples.loc[window_id]
    clip_id, start, stop = layout[1]
    assert carrier.table[SOURCE_INDEX_FIELD].loc[window_id] == clip_id
    print(f"window {window_id}: {clip_id} [{start}, {stop})s")
    print(f"  video  {sample['video'].shape}  (frames @ ~29.97 fps NTSC)")
    print(f"  audio  {sample['audio'].shape}  (samples @ {AUDIO_SAMPLE_RATE} Hz)")
    print(f"  transcript: {sample['transcript']!r}")

    reference = expected_sample(clip_id, start, stop)
    assert np.array_equal(sample["video"], reference["video"])
    assert np.array_equal(sample["audio"], reference["audio"])
    assert sample["transcript"] == reference["transcript"]
    print("  matches the directly computed reference sample")

    print("\n=== dataloader loop: per-row materialization via items() ===")
    print("(note the 59 vs 60 frame windows from NTSC truncation, and the rolling-")
    print(" caption duplicate 'lorem lorem ipsum dolor' in clip_c)")
    for (window_id, sample), (clip_id, start, stop) in zip(
        samples.items(), layout, strict=True
    ):
        reference = expected_sample(clip_id, start, stop)
        assert np.array_equal(sample["video"], reference["video"])
        assert np.array_equal(sample["audio"], reference["audio"])
        assert sample["transcript"] == reference["transcript"]
        print(
            f"  {window_id}: {clip_id} [{start:.1f}, {stop:.1f})s  "
            f"video{sample['video'].shape} audio{sample['audio'].shape}  "
            f"text={sample['transcript']!r}"
        )

    print("\n=== the eager arm: a Dataset operand means compute now ===")
    eager = pf.map_fields(
        windows,
        [VIDEO_FIELD, AUDIO_FIELD, SEGMENTS_FIELD, WINDOW_FIELD],
        combine_sample,
        out=SAMPLE_FIELD,
    )
    assert isinstance(eager, pf.Dataset)
    assert not eager.table[SAMPLE_FIELD].isna().any()  # already computed
    collected = samples.collect()
    for eager_sample, lazy_sample in zip(
        eager.table[SAMPLE_FIELD], collected.table[SAMPLE_FIELD], strict=True
    ):
        assert np.array_equal(eager_sample["video"], lazy_sample["video"])
        assert np.array_equal(eager_sample["audio"], lazy_sample["audio"])
        assert eager_sample["transcript"] == lazy_sample["transcript"]
    print("  eager map_fields(dataset) == handle.collect(), sample for sample")

    print("\nAll multimodal fusion checks passed.")


if __name__ == "__main__":
    main()
