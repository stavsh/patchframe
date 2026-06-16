"""Dimension comparability + optional sample_rate (dimension-join-execution.md §5).

Commensurability — the term-validity judgment for dimensional joins — is a
dimension-type-owned judgment (``comparable_with``): same concrete type and
same name, *excluding* per-source resolution params (a TemporalDimension's
rate, a CategoricalDimension's categories). Distinct from ``__eq__`` (which is
structural identity including those params). ``sample_rate`` is optional —
``None`` is a continuous, unsampled axis that joins but cannot resolve to
backend indices.
"""

from __future__ import annotations

import pytest

from patchframe.data.dimensions import (
    CategoricalDimension,
    Dimensions,
    IndexDimension,
    TemporalDimension,
)


# -- comparable_with -----------------------------------------------------------


def test_same_type_and_name_are_comparable():
    assert TemporalDimension(name="time").comparable_with(TemporalDimension(name="time"))
    assert IndexDimension(name="x").comparable_with(IndexDimension(name="x"))


def test_different_name_is_not_comparable():
    assert not TemporalDimension(name="time").comparable_with(
        TemporalDimension(name="t2")
    )


def test_different_type_same_name_is_not_comparable():
    # An IndexDimension and a TemporalDimension named "x" are not commensurable
    # — comparability must see type, not name alone.
    assert not IndexDimension(name="x").comparable_with(TemporalDimension(name="x"))
    assert not TemporalDimension(name="x").comparable_with(IndexDimension(name="x"))


def test_rate_is_excluded_from_comparability():
    # The forcing case: honest rates must join. Video at NTSC vs a transcript
    # in seconds are the same axis.
    video = TemporalDimension(name="time", sample_rate=30000 / 1001)
    transcript = TemporalDimension(name="time", sample_rate=None)
    audio = TemporalDimension(name="time", sample_rate=16000)

    assert video.comparable_with(transcript)
    assert video.comparable_with(audio)
    assert transcript.comparable_with(audio)


def test_categories_are_excluded_from_comparability():
    # Categories are a resolution param (label→position), like sample_rate —
    # divergent vocabularies still compare by label.
    a = CategoricalDimension(name="label", categories=("cat", "dog"))
    b = CategoricalDimension(name="label", categories=("cat", "dog", "bird"))
    c = CategoricalDimension(name="label")

    assert a.comparable_with(b)
    assert a.comparable_with(c)


def test_comparability_is_distinct_from_equality():
    # __eq__ is structural identity (rate included); comparable_with is
    # commensurability (rate excluded).
    fast = TemporalDimension(name="time", sample_rate=48000)
    slow = TemporalDimension(name="time", sample_rate=16000)

    assert fast != slow
    assert fast.comparable_with(slow)


# -- optional sample_rate (continuous axis) ------------------------------------


def test_sample_rate_defaults_to_continuous():
    assert TemporalDimension(name="time").sample_rate is None


def test_continuous_axis_cannot_resolve_to_indices():
    clock = TemporalDimension(name="time", sample_rate=None)
    with pytest.raises(ValueError, match="continuous"):
        clock.to_index(slice(1.0, 2.0))


def test_rational_rate_resolves():
    # NTSC: 2.0 s at 30000/1001 fps truncates to frame 59 (the 59-vs-60 case).
    ntsc = TemporalDimension(name="time", sample_rate=30000 / 1001)
    resolved = ntsc.to_index(slice(0.0, 2.0))
    assert resolved.value == slice(0, 59)


def test_continuous_axis_still_resolves_full_extent_when_unmentioned():
    # A continuous axis in a Dimensions layout: absent from the slice → full
    # extent (no to_index call), so it composes with resolvable siblings.
    dims = Dimensions(
        (
            TemporalDimension(name="time", sample_rate=None),
            IndexDimension(name="x"),
        )
    )
    resolved = dims.resolve(IndexDimension(name="x").spec(0, 10))
    assert resolved[0].value == slice(None)  # time: full extent, unresolved
    assert resolved[1].value == slice(0, 10)
