"""partition: the tall-bundle group-by (partition-aggregate.md).

partition splits one dataset into key-fibers: one base row per key, indexed by
the key labels, with a BundleField column of member sub-datasets. It takes no
aggregation function — per-group computation is map_fields over the fiber
column, under the operand-dispatch law. by= dispatches on field type: a
ForeignIndexField key inherits the referenced identity (identity scope, with
domain= totality and empty fibers); a plain value key mints (categorical).
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _collect_texts(fiber):
    return tuple(fiber.table["text"])


def _members() -> pf.Dataset:
    """Member rows keyed by a plain (un-linked) value column."""

    return pf.make_from_dataframe(
        pd.DataFrame(
            {"clip": ["b", "a", "b", "c"], "text": ["t0", "t1", "t2", "t3"]},
            index=pd.Index(["s0", "s1", "s2", "s3"], name="seg_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="seg_id"),
                pf.ValueField(name="clip", dtype=str),
                pf.ValueField(name="text", dtype=str),
            )
        ),
    )


def _clips() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"duration": [10.0, 20.0, 30.0, 40.0]},
            index=pd.Index(["a", "b", "c", "d"], name="clip_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.ValueField(name="duration", dtype=float),
            )
        ),
    )


def _segments(
    clips: pf.Dataset, labels: tuple[str, ...] = ("a", "b", "a", "c", "a")
) -> pf.Dataset:
    """Member rows whose key is a typed reference into the clips namespace."""

    table = pd.DataFrame(
        {
            "clip_id": list(labels),
            "text": [f"t{i}" for i in range(len(labels))],
        },
        index=pd.Index([f"s{i}" for i in range(len(labels))], name="seg_id"),
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name="seg_id"),
            pf.ForeignIndexField(
                name="clip_id",
                index_identity=pf.primary_index_identity(clips),
            ),
            pf.ValueField(name="text", dtype=str),
        )
    )
    return pf.make_from_dataframe(table, schema)


# -- categorical arm ---------------------------------------------------------


def test_categorical_arm_mints_and_orders_first_appearance():
    ds = _members()

    groups = pf.partition(ds, "clip", into="members")

    assert isinstance(groups, pf.Dataset)
    assert list(groups.table.index) == ["b", "a", "c"]
    assert groups.table.index.name == "clip"
    assert isinstance(groups.schema.get("members"), pf.BundleField)
    # Fresh key-namespace identity, not the member dataset's row identity.
    assert pf.primary_index_identity(groups) != pf.primary_index_identity(ds)


def test_fibers_are_unmodified_row_subsets():
    ds = _members()

    groups = pf.partition(ds, "clip", into="members")
    fiber = groups.table.loc["b", "members"]

    assert isinstance(fiber, pf.Dataset)
    # Original row order within the fiber, full member schema (key included).
    assert list(fiber.table.index) == ["s0", "s2"]
    assert fiber.schema.names() == ds.schema.names()
    assert fiber.table["text"].tolist() == ["t0", "t2"]
    # Fibers keep the member dataset's row identity and couplings.
    assert pf.primary_index_identity(fiber) == pf.primary_index_identity(ds)
    assert fiber.couplings == ds.couplings


def test_partition_is_deterministic():
    ds = _members()

    first = pf.partition(ds, "clip", into="members")
    second = pf.partition(ds, "clip", into="members")

    assert list(first.table.index) == list(second.table.index)
    for label in first.table.index:
        assert (
            first.table.loc[label, "members"].table.index.tolist()
            == second.table.loc[label, "members"].table.index.tolist()
        )


# -- foreign-key arm (identity scope) ----------------------------------------


def test_foreign_key_arm_inherits_target_identity():
    clips = _clips()
    segments = _segments(clips)

    groups = pf.partition(segments, "clip_id", into="segments")

    # Observed keys only (no domain), first-appearance order.
    assert list(groups.table.index) == ["a", "b", "c"]
    assert pf.primary_index_identity(groups) == pf.primary_index_identity(clips)


def test_domain_makes_base_total_with_empty_fibers():
    clips = _clips()
    segments = _segments(clips)

    groups = pf.partition(segments, "clip_id", domain=clips, into="segments")

    assert list(groups.table.index) == ["a", "b", "c", "d"]  # domain order
    empty = groups.table.loc["d", "segments"]
    assert isinstance(empty, pf.Dataset)
    assert len(empty.table) == 0
    # The member schema survives into the empty fiber (no sentinel value).
    assert empty.schema.names() == segments.schema.names()
    assert pf.primary_index_identity(groups) == pf.primary_index_identity(clips)


def test_attach_to_domain_by_identity_alignment():
    clips = _clips()
    segments = _segments(clips)
    groups = pf.partition(segments, "clip_id", domain=clips, into="segments")

    # Inherited identity: same-name index fields unify with no collision
    # strategy (join-dimensions-identity.md §5).
    out = pf.concat_columns(clips, pf.keep(groups, ["segments"]))

    assert list(out.table.index) == ["a", "b", "c", "d"]
    assert out.table.loc["a", "segments"].table["text"].tolist() == ["t0", "t2", "t4"]
    assert len(out.table.loc["d", "segments"].table) == 0


def test_flatten_round_trips_rows_and_identity():
    clips = _clips()
    segments = _segments(clips)
    groups = pf.partition(segments, "clip_id", domain=clips, into="segments")

    flat = pf.flatten(groups)

    assert sorted(flat.table.index) == sorted(segments.table.index)
    assert pf.primary_index_identity(flat) == pf.primary_index_identity(segments)


# -- aggregation via map_fields ----------------------------------------------


def test_map_fields_aggregates_fibers_eagerly():
    clips = _clips()
    segments = _segments(clips)
    groups = pf.partition(segments, "clip_id", domain=clips, into="segments")

    fused = pf.map_fields(groups, ["segments"], _collect_texts, out="texts")

    assert fused.table["texts"].tolist() == [
        ("t0", "t2", "t4"),
        ("t1",),
        ("t3",),
        (),
    ]


def test_map_fields_over_fibers_defers_on_handle_arm():
    clips = _clips()
    segments = _segments(clips)
    groups = pf.partition(segments, "clip_id", domain=clips, into="segments")

    handle = pf.map_fields(groups.fields(["segments"]), _collect_texts, out="texts")

    assert isinstance(handle, pf.FieldHandle)
    carrier = handle.dataset_context.dataset
    assert carrier.table["texts"].isna().all()
    assert handle.collect().table["texts"].tolist() == [
        ("t0", "t2", "t4"),
        ("t1",),
        ("t3",),
        (),
    ]


# -- validation ----------------------------------------------------------------


def test_null_keys_error():
    ds = pf.make_from_dataframe(
        pd.DataFrame(
            {"clip": ["a", None], "text": ["t0", "t1"]},
            index=pd.Index(["s0", "s1"], name="seg_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="seg_id"),
                pf.ValueField(name="clip", dtype=str),
                pf.ValueField(name="text", dtype=str),
            )
        ),
    )

    with pytest.raises(ValueError, match="null"):
        pf.partition(ds, "clip")


def test_dangling_foreign_label_errors():
    clips = _clips()
    segments = _segments(clips, labels=("a", "zz"))

    with pytest.raises(ValueError, match="not present in the domain index"):
        pf.partition(segments, "clip_id", domain=clips)


def test_domain_requires_foreign_key():
    with pytest.raises(TypeError, match="requires a ForeignIndexField"):
        pf.partition(_members(), "clip", domain=_clips())


def test_domain_identity_mismatch_errors():
    clips = _clips()
    other = _clips()  # same labels, freshly minted namespace
    segments = _segments(clips)

    with pytest.raises(ValueError, match="identity does not match"):
        pf.partition(segments, "clip_id", domain=other)


def test_key_validation_errors():
    clips = _clips()
    segments = _segments(clips)

    with pytest.raises(ValueError, match="not in the schema"):
        pf.partition(segments, "missing")
    with pytest.raises(TypeError, match="primary index"):
        pf.partition(segments, "seg_id")
    with pytest.raises(ValueError, match="collides with the base index"):
        pf.partition(segments, "clip_id", into="clip_id")
