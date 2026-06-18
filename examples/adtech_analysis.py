"""Adtech analysis: composing the seven raw sources into multi-grain answers.

Built on ``examples/adtech.py`` (the synthetic warehouse + sources). This module
is the *usage* layer — the reports advertisers normally receive pre-aggregated
(reach/frequency, sales/ROAS) are **composed in-framework** here, because revenue,
cost, identity, and audience each come from their own source.

This file currently lands the **attribution foundation** (the keystone the
sections sit on). The section lessons — roll-up + measure additivity, visual <->
ROAS, audience overfit, and the attribution noise/confidence sensitivity study —
build on top and land next.

Attribution is a *join*, not a logged fact (there is no single user id across the
lifecycle): an impression's ``device_id`` is bridged to a conversion's
``advertiser_user_id`` through the external ``identity_map``, within a lookback
window, then resolved to the last touch. The key idea is that the resulting
**correspondence plan** is the interchangeable/additive seam: a deterministic key
match (here), a fuzzy candidate-gen, or a consumed external map all produce the
same shape, and the downstream is unchanged. ``confidence`` rides as an ordinary
column (modelled, not forced through the join primitives).

Marked gaps surfaced building this (addressed in a following iteration):

- **Chaining correspondences through ``merge`` needs manual plan-column hygiene**
  (``set_index``/``drop`` between hops to stop ``left_index``/``right_index``
  colliding). ``dimension_join(..., candidates=...)`` is the intended cleaner
  chain; raw merge-chaining is verbose.
- **The lookback is an interval predicate** done here as a ``where`` over the
  pair table — it belongs inside ``match`` as ``overlap``/``within`` (needs the
  timestamps as a temporal-dimension slice; the fusion example's pattern).
- **Last-touch resolution is a reduce/argmax** over the correspondence — done in
  pandas here; the declared agg-spec / ``reduce`` vocabulary is deferred.
- **Per-pair ``confidence`` is carried as a column, not on the correspondence
  itself** — the payload seam (``dimension-join-execution.md`` §5), modelled not
  built.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import patchframe as pf
from examples.adtech import (
    ADVERTISER_USER_ID,
    CAMPAIGN_ID,
    CONFIDENCE,
    CONVERSION_ID,
    CREATIVE_ID,
    DANGLING_CREATIVE,
    DEVICE_ID,
    IMPRESSION_ID,
    LOOKBACK_DAYS,
    REVENUE,
    TIMESTAMP,
    AdtechSources,
    Warehouse,
    make_adtech_sources,
)

#: Most composition hops collide only on the join key (same value both sides) and
#: the accumulating plan columns; keep the left side for the former.
KEEP_LEFT = pf.ColumnCollisionStrategy(mode="keep", side="left")

IMPRESSION_TS = "impression_ts"
CONVERSION_TS = "conversion_ts"

#: Default identity-confidence gate. Dropping low-confidence/false links before
#: attribution is the one lever that sharply improves the recovered signal (the
#: payload seam paying off), so it defaults on.
DEFAULT_MIN_CONFIDENCE = 0.7


def attribution_candidates(
    sources: AdtechSources, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> pf.Dataset:
    """The candidate (impression, conversion) correspondence via the identity bridge.

    Two equality hops through the open seam: impression --``device_id``-->
    ``identity_map`` --``advertiser_user_id``--> conversion. The result is a
    correspondence plan (``left_index`` = impression, ``right_index`` =
    conversion) carrying each side's columns plus the bridge ``confidence`` — the
    same shape a fuzzy matcher would emit (interchangeable), and one that further
    candidate sources could be ``concat``-ed into (additive).

    Null-device impressions cannot bridge and are dropped first; the
    ``confidence`` gate removes low-confidence/false links before the join.
    """

    impressions = pf.where(sources.impression_log, lambda t: t[DEVICE_ID].notna())
    bridge = pf.where(sources.identity_map, lambda t: t[CONFIDENCE] >= min_confidence)
    conversions = pf.rename(sources.conversions, {TIMESTAMP: CONVERSION_TS})

    # Hop 1: impression -> advertiser id (device_id equals).
    bridged = pf.merge(
        impressions, bridge, pf.join(impressions, bridge, on=DEVICE_ID),
        collision=KEEP_LEFT,
    )
    # Plan-column hygiene between hops (the marked friction): promote the
    # impression identity back to the index and drop the spent right reference,
    # so the next merge's plan columns do not collide.
    bridged = pf.set_index(bridged, "left_index", index_name=IMPRESSION_ID)
    bridged = pf.drop(bridged, ["right_index"])
    bridged = pf.rename(bridged, {TIMESTAMP: IMPRESSION_TS})

    # Hop 2: bridged impression -> conversion (advertiser id equals).
    return pf.merge(
        bridged, conversions, pf.join(bridged, conversions, on=ADVERTISER_USER_ID),
        collision=KEEP_LEFT,
    )


def attribute(
    sources: AdtechSources,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    lookback_days: int = LOOKBACK_DAYS,
) -> pf.Dataset:
    """Resolve the candidate correspondence to one creative per conversion.

    Narrows the candidates to the lookback window and keeps the **last touch**
    (the most recent impression before the conversion). Returns the attributed
    conversions: ``conversion_id`` -> ``creative_id`` / ``campaign_id`` /
    ``revenue`` / bridge ``confidence``.

    The lookback ``where`` (an interval predicate) and the last-touch groupby (a
    reduce/argmax) are the marked gaps — done over the pair table here; the
    join-produced correspondence above is the real, reusable seam.
    """

    pairs = attribution_candidates(sources, min_confidence=min_confidence)
    table = pairs.table
    elapsed = (table[CONVERSION_TS] - table[IMPRESSION_TS]).dt.total_seconds()
    within = table[(elapsed >= 0) & (elapsed <= lookback_days * 24 * 3600)]
    last_touch = within.sort_values(IMPRESSION_TS).groupby("right_index").tail(1)

    attributed = (
        last_touch.rename(columns={"right_index": CONVERSION_ID})
        .loc[:, [CONVERSION_ID, CREATIVE_ID, CAMPAIGN_ID, REVENUE, CONFIDENCE]]
        .set_index(CONVERSION_ID)
    )
    attributed[CREATIVE_ID] = attributed[CREATIVE_ID].astype("string")
    attributed[CAMPAIGN_ID] = attributed[CAMPAIGN_ID].astype("string")
    attributed[REVENUE] = attributed[REVENUE].astype("Float64")
    attributed[CONFIDENCE] = attributed[CONFIDENCE].astype("Float64")
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=CONVERSION_ID),
            pf.ValueField(name=CREATIVE_ID, dtype=str),
            pf.ValueField(name=CAMPAIGN_ID, dtype=str),
            pf.ValueField(name=REVENUE, dtype=float),
            pf.ValueField(name=CONFIDENCE, dtype=float),
        )
    )
    return pf.make_from_dataframe(attributed, schema)


def attribution_accuracy(attributed: pf.Dataset, warehouse: Warehouse) -> dict[str, float]:
    """Compare attributed creatives against the latent causal truth (for tests/demo).

    ``coverage`` = fraction of true conversions that got attributed at all;
    ``accuracy`` = fraction of attributed conversions whose creative is correct.
    """

    truth = warehouse.true_attribution.set_index(CONVERSION_ID)[CREATIVE_ID]
    got = attributed.table[CREATIVE_ID]
    common = got.index.intersection(truth.index)
    accuracy = float((got.loc[common].astype(str) == truth.loc[common].astype(str)).mean())
    coverage = float(len(common) / len(truth)) if len(truth) else 0.0
    return {"coverage": coverage, "accuracy": accuracy, "attributed": float(len(got))}


def main() -> None:
    print("=== attribution = a lossy join through the identity bridge ===")
    print("(impression.device_id -> identity_map -> conversion.advertiser_user_id,")
    print(" within lookback, resolved to last touch; confidence-gated)\n")

    ideal = make_adtech_sources(attribution_noise=0.0)
    pairs = attribution_candidates(ideal)
    attributed = attribute(ideal)
    stats = attribution_accuracy(attributed, ideal.warehouse)
    print(f"idealized (recency) regime:")
    print(f"  candidate correspondence: {len(pairs.table)} (impression, conversion) pairs")
    print(f"  attributed {int(stats['attributed'])} conversions  "
          f"coverage {stats['coverage']:.0%}  creative-accuracy {stats['accuracy']:.0%}")
    assert stats["accuracy"] >= 0.85, stats

    realistic = make_adtech_sources(attribution_noise=1.0)
    noisy = attribution_accuracy(attribute(realistic), realistic.warehouse)
    print(f"\nrealistic (random-delay) regime:")
    print(f"  attributed {int(noisy['attributed'])} conversions  "
          f"creative-accuracy {noisy['accuracy']:.0%}")
    assert noisy["accuracy"] <= 0.55, noisy

    print("\nThe correspondence above is the interchangeable/additive seam: a")
    print("deterministic key match (here), a fuzzy candidate-gen, or a consumed")
    print("external map all produce the same shape; only the producer changes.")
    print("\nAttribution foundation checks passed.")


if __name__ == "__main__":
    main()
