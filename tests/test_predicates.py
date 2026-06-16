"""Single-dimension match predicates (dimension-join-execution.md).

A predicate supplies semantics (``matches`` — the pairwise oracle) and an
optional bulk strategy (``correspond`` — vectorized; defaults to brute-forcing
``matches``). ``assert_predicate_contract`` pins strategy==semantics offline.
v1 carries the correspondence as expanded pairs throughout; partitioners
(``equals``) produce a compact candidate set via hash join, pairers
(``overlap``) narrow within it.
"""

from __future__ import annotations

import numpy as np

import patchframe as pf
from examples.multimodal_fusion import (
    CAPTION_SLACK_SECONDS,
    SOURCE_INDEX_FIELD,
    TRANSCRIPTS_SRT,
    WINDOW_FIELD,
    make_fusion_clips,
    make_transcript,
    window_clips,
)
from patchframe.data.predicates import Stage


# -- predicate contracts (strategy == semantics) -------------------------------


def test_equals_contract_and_stage():
    left = ["a", "b", "a", "c"]
    right = ["b", "a", "a", "d"]
    pf.assert_predicate_contract(pf.equals(), left, right)
    assert pf.equals().stage is Stage.PARTITIONER


def test_equals_hash_join_matches_brute_force():
    left = ["a", "b", "a", "c", "b"]
    right = ["b", "a", "a", "b"]
    li, ri = pf.equals().correspond(left, right, None)
    got = {(int(i), int(j)) for i, j in zip(li, ri)}
    want = {(i, j) for i in range(len(left)) for j in range(len(right)) if left[i] == right[j]}
    assert got == want


def test_overlap_contract_with_and_without_pad():
    left = [(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)]
    right = [(1.5, 1.8), (2.1, 2.4), (10.0, 11.0)]
    pf.assert_predicate_contract(pf.overlap(), left, right)
    pf.assert_predicate_contract(pf.overlap(pad=0.25), left, right)
    assert pf.overlap().stage is Stage.BLOCK_PAIRER


def test_overlap_pad_widens_matches():
    # [0,2) and [2.1,2.4) do not overlap, but pad=0.25 brings them together.
    left = [(0.0, 2.0)]
    right = [(2.1, 2.4)]
    assert pf.overlap(pad=0.0).correspond(left, right, None)[0].size == 0
    li, ri = pf.overlap(pad=0.25).correspond(left, right, None)
    assert (int(li[0]), int(ri[0])) == (0, 0)


def test_overlap_narrows_candidates():
    left = [(0.0, 2.0), (5.0, 7.0)]
    right = [(1.0, 1.5), (6.0, 6.5)]
    # A candidate set that includes a non-overlapping pair (0, 1) → dropped.
    candidates = (np.array([0, 0, 1]), np.array([0, 1, 1]))
    li, ri = pf.overlap().correspond(left, right, candidates)
    assert {(int(i), int(j)) for i, j in zip(li, ri)} == {(0, 0), (1, 1)}


def test_overlap_is_picklable():
    import pickle

    term = pf.overlap(pad=0.25)
    assert pickle.loads(pickle.dumps(term)) == term


# -- the chain reproduces the spike oracle on real fusion data -----------------


def _operands():
    clips = make_fusion_clips()
    windows = window_clips(clips)
    segments = make_transcript(TRANSCRIPTS_SRT)
    w_clip = windows.table[SOURCE_INDEX_FIELD].to_numpy()
    w_interval = [
        (c.dims["time"].start, c.dims["time"].stop) for c in windows.table[WINDOW_FIELD]
    ]
    s_clip = segments.table["clip_id"].to_numpy()
    s_interval = list(
        zip(segments.table["seg_start"], segments.table["seg_end"], strict=True)
    )
    return windows, segments, w_clip, w_interval, s_clip, s_interval


def test_equals_then_overlap_chain_reproduces_oracle():
    windows, segments, w_clip, w_interval, s_clip, s_interval = _operands()
    pad = CAPTION_SLACK_SECONDS

    # The chain: equals (clip) produces candidates; overlap (time) narrows them.
    candidates = pf.equals().correspond(w_clip, s_clip, None)
    li, ri = pf.overlap(pad=pad).correspond(w_interval, s_interval, candidates)

    got: dict = {}
    for i, j in zip(li, ri, strict=True):
        got.setdefault(int(i), set()).add(int(j))

    # Oracle: the same clip-equality + padded-overlap test, brute force.
    want: dict = {}
    for i in range(len(w_clip)):
        wc, (ws, we) = w_clip[i], w_interval[i]
        lo, hi = ws - pad, we + pad
        for j in range(len(s_clip)):
            ss, se = s_interval[j]
            if s_clip[j] == wc and ss < hi and se > lo:
                want.setdefault(i, set()).add(j)

    assert got == want
    assert sum(len(v) for v in got.values()) > 0  # the join is non-trivial
