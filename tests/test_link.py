"""link: typing a value column as a reference into another dataset's index.

link(ds, target, field) upgrades a label-valued column to a ForeignIndexField
carrying the target's IndexIdentity (join-dimensions-identity.md §4) — the
dual of set_index, and the entry point into identity-scoped operations
(partition's identity arm, the future validated join scope). Schema-typing
only: the table is untouched and the field keeps its values, dtype, and field
identity.

Operand-dispatch law: a FieldHandle operand means the lazy arm — link rewrites
schema, so it lifts onto a BundleField carrier (bundle both sides first);
there is no eager handle resolution.
"""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _clips() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {"duration": [10.0, 20.0]},
            index=pd.Index(["a", "b"], name="clip_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="clip_id"),
                pf.ValueField(name="duration", dtype=float),
            )
        ),
    )


def _transcript(labels: tuple[str, ...] = ("a", "b", "a")) -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {
                "clip_id": list(labels),
                "text": [f"t{i}" for i in range(len(labels))],
            },
            index=pd.Index([f"s{i}" for i in range(len(labels))], name="seg_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="seg_id"),
                pf.ValueField(name="clip_id", dtype=str),
                pf.ValueField(name="text", dtype=str),
            )
        ),
    )


def test_link_upgrades_value_field_to_foreign_index():
    clips = _clips()
    transcript = _transcript()

    linked = pf.link(transcript, clips, "clip_id")

    field = linked.schema.get("clip_id")
    assert isinstance(field, pf.ForeignIndexField)
    assert field.target_identity == pf.primary_index_identity(clips)
    # A retype, not a new field: values, dtype, and lineage survive.
    assert field.field_identity == transcript.schema.get("clip_id").field_identity
    assert field.dtype == transcript.schema.get("clip_id").dtype
    assert linked.table["clip_id"].tolist() == transcript.table["clip_id"].tolist()
    # Row identity and the rest of the schema are untouched.
    assert pf.primary_index_identity(linked) == pf.primary_index_identity(transcript)
    assert linked.schema.names() == transcript.schema.names()


def test_link_deferred_arm_over_bundle():
    """The lazy arm: bundle both sides, defer the link, collect runs it."""

    clips = _clips()
    transcript = _transcript()
    b = pf.bundle(transcript=transcript, clips=clips)

    handle = pf.link(
        b.field("transcript"), b.field("clips"), "clip_id", out="linked"
    )

    assert isinstance(handle, pf.FieldHandle)
    linked = handle.collect()
    field = linked.schema.get("clip_id")
    assert isinstance(field, pf.ForeignIndexField)
    assert field.target_identity == pf.primary_index_identity(clips)
    assert linked.table["clip_id"].tolist() == transcript.table["clip_id"].tolist()


def test_link_regular_field_handle_routes_lazy_not_eager():
    """The law: a handle input selects the lazy arm — never eager resolution.

    link is not coupling-able (schema rewrite), so its lazy form needs bundle
    cells; a regular field handle is rejected, not silently resolved.
    """

    clips = _clips()
    transcript = _transcript()

    with pytest.raises(TypeError, match="bundle FieldHandles"):
        pf.link(transcript.field("clip_id"), clips, "clip_id", out="linked")


def test_link_validates_labels_against_target_index():
    clips = _clips()
    transcript = _transcript(labels=("a", "zz"))

    with pytest.raises(ValueError, match="not present in the target index"):
        pf.link(transcript, clips, "clip_id")

    linked = pf.link(transcript, clips, "clip_id", allow_dangling=True)
    assert isinstance(linked.schema.get("clip_id"), pf.ForeignIndexField)


def test_link_field_validation():
    clips = _clips()
    transcript = _transcript()

    with pytest.raises(ValueError, match="not in the schema"):
        pf.link(transcript, clips, "missing")
    with pytest.raises(TypeError, match="primary index"):
        pf.link(transcript, clips, "seg_id")

    linked = pf.link(transcript, clips, "clip_id")
    with pytest.raises(TypeError, match="already a ForeignIndexField"):
        pf.link(linked, clips, "clip_id")


def test_link_then_partition_identity_arm():
    """The composed flow the fork resolves to (partition-aggregate.md §6)."""

    clips = _clips()
    transcript = pf.link(_transcript(), clips, "clip_id")

    groups = pf.partition(transcript, "clip_id", domain=clips, into="segments")
    attached = pf.concat_columns(clips, pf.keep(groups, ["segments"]))

    assert list(attached.table.index) == ["a", "b"]
    assert attached.table.loc["a", "segments"].table["text"].tolist() == ["t0", "t2"]
    assert attached.table.loc["b", "segments"].table["text"].tolist() == ["t1"]
