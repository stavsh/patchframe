"""Adtech, the patchframe-shaped pivot: assemble attribution training data.

Where the roll-up (``adtech_analysis.py``) is pandas-shaped, this is the workload
patchframe is *for*: a streamable dataset of variable-length per-identity
**journeys** — each a time-ordered sequence of creative exposures with
heterogeneous per-step features — labelled with the user's conversion outcome,
ready to feed a sequence model (an attention attribution transformer) via a
DataLoader. The transformer itself is downstream (``patchframe_examples``, torch);
this core module is the dependency-light *data assembly*, which is the argument
for patchframe in this use case: pandas cannot hold a dataset *of sequences* of
heterogeneous features and stream it.

Pipeline (reusing the seven sources + the attribution foundation):

1. **Resolve identity** — bridge impressions to ``advertiser_user_id`` through the
   (confidence-gated) ``identity_map``. Cross-device unification falls out: a
   person's devices map to one advertiser id, so their impressions land in one
   journey. Lossy (unmapped / null-device impressions drop) — the realistic
   identity story now shaping the *training set*.
2. **Journeys = sequence fibers** — ``partition`` by ``advertiser_user_id`` over a
   minted ``users`` domain; each fiber is that identity's exposure sub-dataset.
3. **Per-step features** — each step carries the creative's visual feature
   (recovered from the asset) + viewability; the sequence is assembled (ordered
   by time) per journey.
4. **Label** — the user's conversion outcome (converted / revenue), aggregated
   per identity over the same ``users`` domain and attached by identity alignment.
5. **Stream** — ``journeys.rows()`` yields ``(sequence_features, label)`` samples;
   ``collate_journeys`` pads variable lengths. A torch DataLoader plugs directly
   in (zero torch in core).

In-memory v1 (measure the payoff; out-of-core is the marked forcing function).
Marked gaps surfaced: **no ``sort`` operator** so sequence order is done at
sample-assembly (ordered fibers / ``sort`` is the gap); **out-of-core journey
streaming** needs the lazy ``BundleField`` cell (this is plausibly the workload
that gates it); **``map_fields`` opaque returns** (the per-step assembly fn) carry
the honesty concern; per-creative feature lookup is an ``assign``+map
(field-expression gap).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import patchframe as pf
from examples.adtech import (
    ADVERTISER_USER_ID,
    CREATIVE_ID,
    DANGLING_CREATIVE,
    DEVICE_ID,
    CONFIDENCE,
    REVENUE,
    TIMESTAMP,
    VIEWABLE,
    AdtechSources,
    make_adtech_sources,
)
from examples.adtech_analysis import (
    DEFAULT_MIN_CONFIDENCE,
    KEEP_LEFT,
    VISUAL_FEATURE,
    _count,
    attribute,
    attribution_accuracy,
    creative_visuals,
)

SEQUENCE = "sequence"  # the BundleField fiber: a journey's exposure sub-dataset
SEQUENCE_FEATURES = "sequence_features"  # the assembled (T, F) per-step feature array
SEQUENCE_LENGTH = "sequence_length"
CONVERTED = "converted"
JOURNEY_REVENUE = "journey_revenue"
IMPRESSION_ID = "impression_id"


def _assemble_sequence(fiber: pf.Dataset) -> np.ndarray:
    """Order a journey's steps by time and stack per-step features into (T, F).

    MARK (ordered fibers / ``sort`` gap): there is no ``sort`` operator and sorting
    via ``.table`` would be a transform-smell, so the time order is applied here,
    at sample assembly. MARK (note #3): an opaque ``map_fields`` return — the
    declared output field cannot vouch for this array's shape/contents.
    """

    steps = fiber.table.sort_values(TIMESTAMP)
    return np.stack(
        [
            steps[VISUAL_FEATURE].to_numpy(dtype=float),
            steps[VIEWABLE].astype("float").to_numpy(),
        ],
        axis=-1,
    )


def _converted(fiber: pf.Dataset) -> bool:
    return len(fiber.table) > 0


def _journey_revenue(fiber: pf.Dataset) -> float:
    column = fiber.table[REVENUE]
    return float(column.sum()) if len(column) else 0.0


def _resolve_identity(sources: AdtechSources, *, min_confidence: float) -> pf.Dataset:
    """Bridge impressions to ``advertiser_user_id`` and attach the per-step visual.

    The identity bridge (attribution hop 1) unifies a person's devices into one
    journey. The creative visual feature is looked up per creative and attached to
    each impression (MARK: dimension lookup / field-expression — an ``assign``+map).
    """

    creative_visual = creative_visuals(sources).table[VISUAL_FEATURE]
    impressions = pf.where(
        sources.impression_log,
        lambda t: t[DEVICE_ID].notna() & (t[CREATIVE_ID] != DANGLING_CREATIVE),
    )
    impressions = impressions.assign(
        **{VISUAL_FEATURE: impressions.table[CREATIVE_ID].map(creative_visual).astype("Float64")}
    )
    bridge = pf.where(sources.identity_map, lambda t: t[CONFIDENCE] >= min_confidence)
    bridged = pf.merge(
        impressions, bridge, pf.join(impressions, bridge, on=DEVICE_ID), collision=KEEP_LEFT
    )
    return pf.set_index(bridged, "left_index", index_name=IMPRESSION_ID)


def _user_domain(bridged: pf.Dataset) -> pf.Dataset:
    """Mint the resolved-identity universe (the users reached). MARK: a derived
    dimension (a real CRM/user source would supply this)."""

    ids = pd.unique(bridged.table[ADVERTISER_USER_ID].dropna())
    table = pd.DataFrame(index=pd.Index(ids, name=ADVERTISER_USER_ID))
    return pf.make_from_dataframe(table, pf.Schema(fields=(pf.IndexField(name=ADVERTISER_USER_ID),)))


def make_journeys(
    sources: AdtechSources, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> pf.Dataset:
    """Assemble the journeys dataset: one row per identity, a sequence + a label."""

    bridged = _resolve_identity(sources, min_confidence=min_confidence)
    users = _user_domain(bridged)

    # Journeys = exposure sequence fibers over the users domain.
    bridged = pf.link(bridged, users, ADVERTISER_USER_ID)
    journeys = pf.partition(bridged, ADVERTISER_USER_ID, domain=users, into=SEQUENCE)
    journeys = pf.map_fields(journeys, [SEQUENCE], _assemble_sequence, out=SEQUENCE_FEATURES)
    journeys = pf.map_fields(journeys, [SEQUENCE], _count, out=SEQUENCE_LENGTH)

    # Label = conversion outcome per identity, attached by identity alignment.
    # Only identities with a journey (>=1 reachable impression) can be a sample;
    # converters who saw no reachable ad fall outside the users domain.
    user_ids = set(users.table.index)
    conversions = pf.where(sources.conversions, lambda t: t[ADVERTISER_USER_ID].isin(user_ids))
    conversions = pf.link(conversions, users, ADVERTISER_USER_ID)
    outcomes = pf.partition(conversions, ADVERTISER_USER_ID, domain=users, into="conversions")
    outcomes = pf.map_fields(outcomes, ["conversions"], _converted, out=CONVERTED)
    outcomes = pf.map_fields(outcomes, ["conversions"], _journey_revenue, out=JOURNEY_REVENUE)

    return pf.concat_columns(
        pf.drop(journeys, [SEQUENCE]),
        pf.drop(outcomes, ["conversions"]),
    )


def collate_journeys(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length journey sequences into a batch (the DataLoader collate).

    Pure-numpy here (a torch collate would return tensors); the point is the
    protocol — ``DataLoader(journeys.rows(), collate_fn=collate_journeys)`` plugs
    in with zero torch in core.
    """

    lengths = [int(sample[SEQUENCE_LENGTH]) for sample in batch]
    max_len = max(lengths) if lengths else 0
    feature_dim = batch[0][SEQUENCE_FEATURES].shape[-1] if batch else 0
    padded = np.zeros((len(batch), max_len, feature_dim), dtype=float)
    for i, sample in enumerate(batch):
        sequence = sample[SEQUENCE_FEATURES]
        padded[i, : sequence.shape[0]] = sequence
    return {
        "sequences": padded,
        "lengths": np.array(lengths),
        "converted": np.array([bool(sample[CONVERTED]) for sample in batch]),
        "revenue": np.array([float(sample[JOURNEY_REVENUE]) for sample in batch]),
    }


def main() -> None:
    sources = make_adtech_sources()
    journeys = make_journeys(sources)

    print("=== journeys: a dataset of variable-length exposure sequences ===")
    lengths = journeys.table[SEQUENCE_LENGTH]
    converted = journeys.table[CONVERTED]
    print(f"  {len(journeys.table)} journeys (resolved identities); "
          f"sequence length min/mean/max = {int(lengths.min())}/{lengths.mean():.1f}/{int(lengths.max())}")
    print(f"  converted journeys: {int(converted.sum())} / {len(converted)}")

    print("\n=== one training sample (via rows(), the DataLoader protocol) ===")
    sample = journeys.rows()[int(np.argmax(journeys.table[SEQUENCE_LENGTH].to_numpy()))]
    features = np.asarray(sample[SEQUENCE_FEATURES])
    print(f"  sequence_features {features.shape} (T steps x [visual, viewable]); "
          f"converted={bool(sample[CONVERTED])} revenue={float(sample[JOURNEY_REVENUE]):.1f}")

    print("\n=== a padded batch (collate_journeys; torch-free) ===")
    rows = journeys.rows()
    batch = collate_journeys([rows[i] for i in range(min(8, len(rows)))])
    print(f"  sequences {batch['sequences'].shape} (batch x max_T x F); "
          f"lengths {batch['lengths'].tolist()}")

    # Label check: a journey is 'converted' iff its identity has a conversion.
    truth = set(sources.warehouse.conversions[ADVERTISER_USER_ID])
    got = set(journeys.table.index[journeys.table[CONVERTED].astype(bool)])
    reachable_converters = {u for u in truth if u in set(journeys.table.index)}
    assert got == reachable_converters, (len(got), len(reachable_converters))
    print(f"\n  labels verified: {len(got)} converted journeys match conversions-by-identity")

    print("\n=== payoff, measured honestly (in-memory v1) ===")
    total_steps = int(journeys.table[SEQUENCE_LENGTH].sum())
    print(f"  the data is the fibers: {total_steps} exposure-steps across "
          f"{len(journeys.table)} journeys, all resident in memory")
    print("  -> NO in-memory memory/streaming win: the journeys are fully materialized.")
    print("     the perf payoff is out-of-core (the lazy BundleField cell) - the marked")
    print("     structural gap this workload gates.")
    print("  what patchframe buys here is STRUCTURAL:")
    print("   + rows() is a DataLoader-ready map-style view for free (no custom Dataset class)")
    print("   + one substrate, two pivots: the same 7 sources + attribution feed both the")
    print("     tabular roll-up (adtech_analysis) and these sequence journeys")
    print("   + identity/lineage tracked through the composition (cross-device unification,")
    print("     domain-aligned labels) - pandas merges lose this")
    print("  a pandas+torch hand-roll does the in-memory assembly in comparable lines")
    print("  (merge + groupby.apply + a custom Dataset for padding); patchframe's delta is the")
    print("  free DataLoader, the composition/reuse, and the path to out-of-core - not in-memory perf")
    print("\n  honest read: the in-memory payoff is structural; the perf payoff needs the")
    print("  structural improvements the example surfaced (lazy fiber cell, sort/ordered")
    print("  fibers, AggSpec, composite keys) - which is the case for building them next.")

    print("\nadtech journeys (in-memory v1) checks passed.")


if __name__ == "__main__":
    main()
