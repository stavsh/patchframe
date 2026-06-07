"""Runnable demonstration of patchframe's lazy <-> eager operator duality.

Every transform operator has two call forms, selected by operand type — there is
no separate lazy API and no ``.lazy()`` switch:

    op(dataset)         # eager:    apply now, return a Dataset
    op(handle, out=...)  # deferred: record the op, return a chaining FieldHandle

A deferred chain records its operators as couplings on a one-row ``bundle``
carrier and materializes only at ``collect()``. Coupling-able ops (``bind_*``)
defer *in place* without a bundle. This script builds the same work both ways and
checks they agree. It needs no external data:

    python examples/lazy_eager_duality_usage.py
"""

from __future__ import annotations

import pandas as pd

import patchframe as pf
from patchframe.data.dimensions import IndexDimension


def _scores() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"score": [10, 20, 30, 40]}, index=["a", "b", "c", "d"]),
        pf.Schema(
            fields=(pf.IndexField(name="item_id"), pf.ValueField(name="score", dtype=int))
        ),
    )


def _labels() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"label": ["cat", "dog", "cat", "fish"]}, index=["a", "b", "c", "d"]),
        pf.Schema(
            fields=(pf.IndexField(name="item_id"), pf.ValueField(name="label", dtype=str))
        ),
    )


def eager_pipeline(scores: pf.Dataset, labels: pf.Dataset) -> pf.Dataset:
    """Each step materializes immediately and returns a ``Dataset``."""

    plan = pf.join(scores, labels, how="inner")
    merged = pf.merge(scores, labels, plan)
    kept = pf.where(merged, lambda df: df["score"] >= 20)
    return pf.drop(kept, ["right_index"])


def deferred_pipeline(scores: pf.Dataset, labels: pf.Dataset) -> pf.Dataset:
    """The same pipeline, recorded as deferred couplings on one bundle carrier.

    ``merge``/``where``/``drop`` are not coupling-able (they compose, filter, and
    narrow the schema), so each lifts onto the carrier as an ``ApplyOperator``
    coupling. The ``out=`` field of one step is the handle the next step consumes,
    so the chain accretes a small coupling graph that the engine topo-sorts and
    runs once, per fiber, at ``collect()``.
    """

    plan = pf.join(scores, labels, how="inner")  # a plan is an ordinary dataset
    b = pf.bundle(scores=scores, labels=labels, plan=plan)

    merged = pf.merge(b.field("scores"), b.field("labels"), b.field("plan"), out="merged")
    kept = pf.where(merged, lambda df: df["score"] >= 20, out="kept")
    trimmed = pf.drop(kept, ["right_index"], out="trimmed")

    # Nothing has executed yet: the carrier holds unmaterialized cells and the
    # three couplings that produce them.
    assert isinstance(trimmed, pf.FieldHandle)
    carrier = trimmed.dataset_context.dataset
    assert pd.isna(carrier.table.at[0, "trimmed"])
    assert len(carrier.couplings.couplings) == 3

    return trimmed.collect()  # materialize the whole chain in one pass


def same_level_deferral() -> pf.Dataset:
    """Coupling-able ops defer *in place* — no bundle — and chain by handle.

    ``bind_dimensions`` only adds a field plus a coupling (schema ``extend``,
    one row per row, per-row-independent), so its deferred form records the
    ``BindDimensions`` coupling directly on the dataset and returns a handle to
    the produced ``clip`` field. ``collect()`` runs the coupling.
    """

    dim = IndexDimension(name="t")
    ds = pf.make_from_dataframe(
        pd.DataFrame({"start": [0, 5], "stop": [4, 9]}, index=["a", "b"]),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.DimensionField(name="start", dimension=dim),
                pf.DimensionField(name="stop", dimension=dim),
            )
        ),
    )
    ctx = ds.context()

    clip = pf.bind_dimensions(
        slice_field="clip",
        bindings={"t": (ctx.field("start"), ctx.field("stop"))},
    )

    assert isinstance(clip, pf.FieldHandle)
    # Same level: a coupling on the dataset itself, no BundleField carrier.
    assert all(not isinstance(field, pf.BundleField) for field in ctx.dataset.schema)
    return clip.collect()


def main() -> None:
    scores, labels = _scores(), _labels()

    print("=== eager vs deferred: the same op, two operand types ===")
    eager_one = pf.where(scores, lambda df: df["score"] >= 20)
    deferred_one = pf.where(
        pf.bundle(scores).field("cell_0"), lambda df: df["score"] >= 20, out="kept"
    )
    print(f"  where(dataset)            -> {type(eager_one).__name__}")
    print(f"  where(handle, out='kept') -> {type(deferred_one).__name__}")
    pd.testing.assert_frame_equal(eager_one.table, deferred_one.collect().table)
    print("  collect() of the deferred form matches the eager one\n")

    print("=== a deferred bundle pipeline: join -> merge -> where -> drop ===")
    eager = eager_pipeline(scores, labels)
    deferred = deferred_pipeline(scores, labels)
    print("eager result:")
    print(eager.table.to_string())
    print("\ndeferred result (one collect() at the end):")
    print(deferred.table.to_string())
    assert eager.schema.names() == deferred.schema.names()
    pd.testing.assert_frame_equal(eager.table, deferred.table)
    print("\n  deferred chain collect() matches the eager pipeline\n")

    print("=== same-level deferral: bind_dimensions records a coupling in place ===")
    clipped = same_level_deferral()
    print("collected 'clip' slices:")
    for item_id in clipped.table.index:
        print(f"  {item_id}: {clipped.table.at[item_id, 'clip'].dims}")

    print("\nAll duality checks passed.")


if __name__ == "__main__":
    main()
