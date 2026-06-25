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

import numpy as np
import pandas as pd

import patchframe as pf
from examples.adtech import (
    ADVERTISER_USER_ID,
    CAMPAIGN_ID,
    CONFIDENCE,
    CONVERSION_ID,
    CPM,
    CREATIVE_ID,
    DANGLING_CREATIVE,
    DEFAULT_SEED,
    DEVICE_ID,
    IMPRESSION_ID,
    LOOKBACK_DAYS,
    PLACEMENT_ID,
    REVENUE,
    SEGMENT_ID,
    TIMESTAMP,
    ASSET,
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

    # Null-device impressions cannot bridge; dangling-creative impressions cannot
    # be attributed to a known creative (no metadata) — both drop out here.
    impressions = pf.where(
        sources.impression_log,
        lambda t: t[DEVICE_ID].notna() & (t[CREATIVE_ID] != DANGLING_CREATIVE),
    )
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


#: The rebuilt grain ``attribute`` declares to the escape: one row per
#: conversion. The escape validates the resolved table against this (return
#: honesty) and mints a fresh ``conversion_id`` row identity.
ATTRIBUTED_SCHEMA = pf.Schema(
    fields=(
        pf.IndexField(name=CONVERSION_ID),
        pf.ValueField(name=CREATIVE_ID, dtype=str),
        pf.ValueField(name=CAMPAIGN_ID, dtype=str),
        # the resolved impression's device — the bridge to segment membership
        # (a conversion has no segment of its own; it inherits the exposure's).
        pf.ValueField(name=DEVICE_ID, dtype=str),
        pf.ValueField(name=REVENUE, dtype=float),
        pf.ValueField(name=CONFIDENCE, dtype=float),
    )
)

@pf.table_transform(schema=ATTRIBUTED_SCHEMA)
def resolve_last_touch(dataset: pf.Dataset, *, lookback_days: int) -> pd.DataFrame:
    """Narrow the candidate pairs to the lookback window, keep the last touch.

    A reusable ``Dataset -> Dataset`` transform: ``@table_transform`` binds the
    escape contract once (the named form of ``pf.pipe``). Supplying
    ``ATTRIBUTED_SCHEMA`` selects the rebuild (construct) shape — fresh
    ``conversion_id`` row identity, the return re-validated against the schema,
    and the seven source records carried forward (what ``make_from_dataframe``
    would have dropped) — so no explicit ``transitions=`` is needed. The body
    stays plain pandas over the input dataset's table.

    This is the part that is *not yet framework-expressible*: the lookback is an
    interval predicate (belongs in ``match`` as ``overlap``/``within``) and
    last-touch is an argmax over the correspondence (a ``reduce``/argmax — no
    ``ArgMax`` reducer exists yet, and it needs ordered fibers). The escape is
    the guard rail until those land; promote this to ``match(overlap)`` + an
    ``ArgMax`` reducer then.
    """

    table = dataset.table
    elapsed = (table[CONVERSION_TS] - table[IMPRESSION_TS]).dt.total_seconds()
    within = table[(elapsed >= 0) & (elapsed <= lookback_days * 24 * 3600)]
    last_touch = within.sort_values(IMPRESSION_TS).groupby("right_index").tail(1)
    attributed = (
        last_touch.rename(columns={"right_index": CONVERSION_ID})
        .loc[:, [CONVERSION_ID, CREATIVE_ID, CAMPAIGN_ID, DEVICE_ID, REVENUE, CONFIDENCE]]
        .set_index(CONVERSION_ID)
    )
    attributed[CREATIVE_ID] = attributed[CREATIVE_ID].astype("string")
    attributed[CAMPAIGN_ID] = attributed[CAMPAIGN_ID].astype("string")
    attributed[DEVICE_ID] = attributed[DEVICE_ID].astype("string")
    attributed[REVENUE] = attributed[REVENUE].astype("Float64")
    attributed[CONFIDENCE] = attributed[CONFIDENCE].astype("Float64")
    return attributed


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

    The lookback (an interval predicate) and last-touch (a reduce/argmax) are
    still the marked gaps, so the resolution is a ``.table`` transform — but a
    *sanctioned* one: ``resolve_last_touch`` is a ``@table_transform`` (the named
    form of the ``pf.pipe`` escape), not raw ``.table`` + ``make_from_dataframe``.
    It re-validates its result against ``ATTRIBUTED_SCHEMA``, mints the fresh
    ``conversion_id`` row identity, and **carries the source provenance forward**
    — the seven raw sources ``make_from_dataframe`` would have dropped. The
    join-produced correspondence above is still the real, reusable seam; the
    escape only guards the bit not yet expressible as an operator.
    """

    pairs = attribution_candidates(sources, min_confidence=min_confidence)
    return resolve_last_touch(pairs, lookback_days=lookback_days)


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


# --------------------------------------------------------------------------- #
# Section A — roll-up + measure additivity ("row = creative_id").
# --------------------------------------------------------------------------- #

# Derived measure columns (composed in-framework — not raw sources).
IMPRESSIONS = "impressions"
SPEND = "spend"
REACH = "reach"
CONVERSIONS = "conversions"
ROAS = "roas"
IMPRESSIONS_FIBER = "impressions_fiber"
CONVERSIONS_FIBER = "conversions_fiber"


# Section A's roll-up and Section C's per-cell counts both use the declared
# ``reduce`` operator (Sum/Count/Distinct) instead of opaque ``map_fields`` passes
# — the gap the example surfaced, filled. Section C's top-segment share is the
# residual non-standard aggregation (a per-fiber ``map_fields``).
def attach_spend(sources: AdtechSources) -> pf.Dataset:
    """Per-impression spend = rate_card CPM / 1000 (cost is its own source).

    MARK (.table / field-expression gap, pinned): a per-row derived column from a
    dimension lookup. Expressed as ``assign`` over values mapped from the declared
    ``rate_card`` source, because joining a value column to a dimension table's
    *index* is awkward today and field-expression algebra (UC1) is deferred. The
    cost still originates in the rate_card source, composed here — not baked in.
    """

    impressions = pf.where(sources.impression_log, lambda t: t[CREATIVE_ID] != DANGLING_CREATIVE)
    cpm = sources.rate_card.table[CPM]
    spend = (impressions.table[PLACEMENT_ID].map(cpm).astype("Float64") / 1000.0)
    return impressions.assign(**{SPEND: spend})


def creative_performance(sources: AdtechSources, attributed: pf.Dataset) -> pf.Dataset:
    """Re-grain base events + attributed revenue to the **creative** grain.

    The roll-up lesson and measure additivity in one place:

    - **additive** (sum across the finer grain): impressions, spend, revenue,
      conversions.
    - **non-additive** (distinct count, recomputed from the base log): ``reach``
      — distinct devices; it cannot be summed from a (creative, segment) grain.
    - **ratio, recomputed** (never averaged): ``ROAS = Σrevenue / Σspend`` — and
      cross-fact (revenue from attribution, spend from delivery).

    Delivery (from the impression log) and revenue (from the attributed
    conversions) are aggregated on the *same* creatives identity (``link`` +
    ``partition(domain=creatives)``), so they recombine by **identity alignment**
    — ``concat_columns`` needs no collision strategy. ``domain=creatives`` makes
    the grain total: a creative with no impressions/conversions gets an empty
    fiber (and 0 / NA), not a missing row.
    """

    impressions = pf.link(attach_spend(sources), sources.creatives, CREATIVE_ID)
    delivery = pf.reduce(
        impressions,
        CREATIVE_ID,
        domain=sources.creatives,
        aggs={
            IMPRESSIONS: pf.Count.on(),  # additive
            SPEND: pf.Sum.on(SPEND),  # additive
            REACH: pf.Distinct.on(DEVICE_ID),  # NON-additive: distinct devices
        },
    )
    attributed = pf.link(attributed, sources.creatives, CREATIVE_ID)
    outcomes = pf.reduce(
        attributed,
        CREATIVE_ID,
        domain=sources.creatives,
        aggs={REVENUE: pf.Sum.on(REVENUE), CONVERSIONS: pf.Count.on()},  # both additive
    )
    # Delivery and revenue share the creatives identity (domain=), so they
    # recombine by alignment. ROAS is a *ratio* — not a reduction; recompute it
    # from the re-summed components afterward (MARK: .table / field-expression).
    combined = pf.concat_columns(delivery, outcomes)
    roas = combined.table[REVENUE].astype("Float64") / combined.table[SPEND]
    return combined.assign(**{ROAS: roas})


# --------------------------------------------------------------------------- #
# Section B — visual <-> ROAS (the lazy-asset payoff; "which visuals win").
# --------------------------------------------------------------------------- #

VISUAL_FEATURE = "visual_feature"


def _visual_feature(asset: object) -> float:
    """Recover a creative's visual 'style' from its decoded thumbnail.

    The mean red channel (normalized) — a stand-in for a real creative-feature
    extractor. Receives the *decoded* array because the ``materialize`` coupling
    runs before this ``map_fields`` reduce (the materialize -> map ordering).
    """

    array = np.asarray(asset)
    return float(array[..., 0].mean()) / 255.0


def creative_visuals(sources: AdtechSources) -> pf.Dataset:
    """Decode creative assets **on demand** and recover the visual feature.

    The laziness payoff: Section A re-grained everything — impressions, spend,
    reach, revenue, ROAS — without decoding a single asset (laziness lives below
    the table, in the accessors). Only *here*, to ask "which visuals win", do we
    pay decode, and only for the creatives under analysis. ``materialize`` records
    the decode coupling (deferred); ``map_fields`` consumes the decoded arrays.
    """

    creatives = pf.materialize(sources.creatives, ASSET)
    return pf.map_fields(creatives, [ASSET], _visual_feature, out=VISUAL_FEATURE)


# --------------------------------------------------------------------------- #
# Section C — audience overfit / cross-grain ("row = creative x segment").
# --------------------------------------------------------------------------- #

SEGMENT_CELLS_FIBER = "segment_cells"
TOP_SEGMENT_SHARE = "top_segment_share"
PEAK_SEGMENT = "peak_segment"


def _sum_conversions(fiber: pf.Dataset) -> int:
    return int(fiber.table[CONVERSIONS].sum())


def _top_segment_share(fiber: pf.Dataset) -> float:
    # Pullback: the creative's total (the base) is broadcast down to its segment
    # cells (the fibers) to form each share; the max share is the concentration.
    cells = fiber.table[CONVERSIONS]
    total = float(cells.sum())
    return float(cells.max() / total) if total else 0.0


def _peak_segment(fiber: pf.Dataset) -> str:
    table = fiber.table
    return str(table.loc[table[CONVERSIONS].idxmax(), SEGMENT_ID])


def attach_segments(dataset: pf.Dataset, sources: AdtechSources) -> pf.Dataset:
    """Segment-attach by ``device_id`` -> ``segment_membership`` (a join).

    A device matches *several* audiences (multi-membership), so this fans out —
    the honest "which audiences saw this" — and a device in no audience drops out
    (unattributed). MARK: a categorical-equals join, done via ``join``+``merge``.
    """

    plan = pf.join(dataset, sources.segment_membership, on=DEVICE_ID)
    return pf.merge(dataset, sources.segment_membership, plan, collision=KEEP_LEFT)


def audience_overfit(sources: AdtechSources, attributed: pf.Dataset) -> pf.Dataset:
    """Flag creatives whose conversions concentrate in one demographic (overfit).

    The cross-grain pivot — row becomes **creative x segment**. Attributed
    conversions are segment-attached and re-grained to the composite grain — a
    real ``(creative_id, segment_id)`` ``MultiIndex`` via ``reduce(by=[...])``,
    no synthetic combined key — then each creative's **top-segment conversion
    share** (volume-gated downstream) is the overfit score: a creative that
    converts almost only in one segment scores high.

    The rollup is the composite "way out": ``reset_index`` decomposes the
    (creative x segment) grain back to columns, and ``partition(by=creative_id)``
    groups its segment cells into a per-creative fiber. The creative's total is
    then broadcast down to those cells to form each share (a base->fiber pullback,
    inside the ``map_fields``). Noise caveat (the robust-metric lesson): naive
    cross-segment ROAS *variance* also flags low-volume creatives on sample noise
    — conversion *concentration* + a volume gate is the signal that isolates the
    real overfit.
    """

    attached = attach_segments(attributed, sources)
    # The cross-grain: a real (creative x segment) composite index — conversions
    # per cell, no synthetic combined key.
    cells = pf.reduce(attached, [CREATIVE_ID, SEGMENT_ID], aggs={CONVERSIONS: pf.Count.on()})

    # Roll up by creative: decompose the composite back to columns, then group by
    # it — each creative's segment cells become its fiber.
    flat = pf.reset_index(cells)
    by_creative = pf.partition(flat, CREATIVE_ID, into=SEGMENT_CELLS_FIBER)
    by_creative = pf.map_fields(by_creative, [SEGMENT_CELLS_FIBER], _top_segment_share, out=TOP_SEGMENT_SHARE)
    by_creative = pf.map_fields(by_creative, [SEGMENT_CELLS_FIBER], _sum_conversions, out=CONVERSIONS)
    by_creative = pf.map_fields(by_creative, [SEGMENT_CELLS_FIBER], _peak_segment, out=PEAK_SEGMENT)
    return pf.drop(by_creative, [SEGMENT_CELLS_FIBER])


# --------------------------------------------------------------------------- #
# Section D — attribution noise / confidence sensitivity (the Monte-Carlo lesson).
# --------------------------------------------------------------------------- #


def attribution_sensitivity(
    *,
    seed: int = DEFAULT_SEED,
    noise_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> pd.DataFrame:
    """How attribution noise degrades the recovered creative-ROAS signal.

    Reuses Section A's roll-up at each ``attribution_noise`` setting. The lesson:
    noise does not merely bias the recovered signal — past a point it *destroys*
    it, and (via the seed Monte Carlo in ``main``) makes the estimate so uncertain
    its confidence interval spans zero. Confidence-gating the identity links
    (``min_confidence``) is the lever that claws accuracy back.
    """

    rows = []
    for noise in noise_levels:
        sources = make_adtech_sources(seed=seed, attribution_noise=noise)
        attributed = attribute(sources, min_confidence=min_confidence)
        accuracy = attribution_accuracy(attributed, sources.warehouse)["accuracy"]
        performance = creative_performance(sources, attributed)
        style = pd.Series(sources.warehouse.styles)
        keep = [c for c in style.index if c != sources.warehouse.overfit_creative]
        roas_indexed = _campaign_indexed(
            performance.table[ROAS], sources.warehouse.creatives[CAMPAIGN_ID]
        )
        rows.append(
            {
                "attribution_noise": noise,
                "creative_accuracy": accuracy,
                "style_roas_corr": _corr(style, roas_indexed, keep),
            }
        )
    return pd.DataFrame(rows)


def _campaign_indexed(values: pd.Series, campaign: pd.Series) -> pd.Series:
    """Index a per-creative measure within its campaign (control for campaign).

    Cross-campaign ROAS is confounded by the ~80x AOV spread; indexing each
    creative against its campaign mean is the control that exposes the visual
    signal. Analysis-only ``.table`` reads (correlation, not a transform).
    """

    aligned = campaign.reindex(values.index)
    return values / values.groupby(aligned).transform("mean")


def _corr(left: pd.Series, right: pd.Series, keys: list[str]) -> float:
    """Pearson correlation over a creative subset (analysis-only)."""

    x = left.reindex(keys).astype(float)
    y = right.reindex(keys).astype(float)
    mask = x.notna() & y.notna()
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    ideal = make_adtech_sources(attribution_noise=0.0)
    attributed = attribute(ideal)
    style = pd.Series(ideal.warehouse.styles)
    campaign = ideal.warehouse.creatives[CAMPAIGN_ID]
    no_overfit = [c for c in style.index if c != ideal.warehouse.overfit_creative]

    print("=== foundation: attribution is a lossy join, not a logged fact ===")
    stats = attribution_accuracy(attributed, ideal.warehouse)
    print(f"  idealized: attributed {int(stats['attributed'])} conversions "
          f"(coverage {stats['coverage']:.0%}, creative-accuracy {stats['accuracy']:.0%})")
    realistic = make_adtech_sources(attribution_noise=1.0)
    noisy = attribution_accuracy(attribute(realistic), realistic.warehouse)
    print(f"  realistic: creative-accuracy {noisy['accuracy']:.0%} (last-touch noise)")
    print("  the correspondence is the interchangeable/additive seam - deterministic")
    print("  match, fuzzy candidate-gen, or a consumed map all produce the same shape")
    assert stats["accuracy"] >= 0.85 and noisy["accuracy"] <= 0.55

    print("\n=== Section A: roll-up to the creative grain (measure additivity) ===")
    performance = creative_performance(ideal, attributed)
    print(performance.table[[IMPRESSIONS, SPEND, REACH, CONVERSIONS, REVENUE, ROAS]]
          .round(2).to_string())
    roas = performance.table[ROAS]
    roas_indexed = _campaign_indexed(roas, campaign)
    corr_raw = _corr(style, roas, no_overfit)
    corr_indexed = _corr(style, roas_indexed, no_overfit)
    print(f"\n  additive: impressions/spend/revenue/conversions (summed). "
          f"non-additive: reach (distinct devices). ratio: ROAS = sum(rev)/sum(spend).")
    print(f"  style vs ROAS: raw cross-campaign {corr_raw:+.2f} (confounded by ~80x AOV) "
          f"-> campaign-indexed {corr_indexed:+.2f} (the visual signal)")
    assert corr_indexed > 0.5

    print("\n=== Section B: visual <-> ROAS (assets decoded only now, on demand) ===")
    visuals = creative_visuals(ideal)
    visual = visuals.table[VISUAL_FEATURE]
    corr_visual = _corr(visual, roas_indexed, no_overfit)
    corr_recovers = _corr(visual, style, list(style.index))
    print(f"  recovered visual feature vs campaign-indexed ROAS: {corr_visual:+.2f}")
    print(f"  (the feature recovers the latent style: corr(visual, style) = "
          f"{corr_recovers:+.2f}; Section A never decoded an asset)")
    assert corr_visual > 0.5 and corr_recovers > 0.95

    print("\n=== Section C: audience overfit (row = creative x segment) ===")
    overfit = audience_overfit(ideal, attributed)
    judged = (
        overfit.table[overfit.table[CONVERSIONS] >= 8]
        .sort_values(TOP_SEGMENT_SHARE, ascending=False)
    )
    print(judged[[TOP_SEGMENT_SHARE, PEAK_SEGMENT, CONVERSIONS]].round(2).to_string())
    flagged = str(judged.index[0])
    print(f"  flagged (highest concentration): {flagged} -> peak segment "
          f"{judged[PEAK_SEGMENT].iloc[0]} (median share {judged[TOP_SEGMENT_SHARE].median():.2f})")
    print(f"  planted: {ideal.warehouse.overfit_creative} -> {ideal.warehouse.target_segment}")
    print("  (naive cross-segment ROAS *variance* would also flag low-volume "
          "creatives on noise; volume-gated concentration is the robust signal)")
    assert flagged == ideal.warehouse.overfit_creative
    assert str(judged[PEAK_SEGMENT].iloc[0]) == ideal.warehouse.target_segment

    print("\n=== Section D: attribution noise -> signal degradation (Monte Carlo) ===")
    sweep = attribution_sensitivity()
    print(sweep.round(2).to_string(index=False))
    assert sweep.iloc[0]["creative_accuracy"] > sweep.iloc[-1]["creative_accuracy"]
    assert sweep.iloc[0]["style_roas_corr"] > 0.5
    corrs = [
        attribution_sensitivity(seed=seed, noise_levels=(1.0,)).iloc[0]["style_roas_corr"]
        for seed in range(DEFAULT_SEED, DEFAULT_SEED + 8)
    ]
    lo, hi = np.percentile(corrs, [2.5, 97.5])
    print(f"  Monte Carlo (noise=1.0, 8 seeds): corr mean {np.mean(corrs):+.2f}, "
          f"95% CI [{lo:+.2f}, {hi:+.2f}] - the noisy estimate is uncertain (spans ~0)")
    assert float(np.mean(corrs)) < float(sweep.iloc[0]["style_roas_corr"])

    print("\nadtech analysis: foundation + Sections A-D checks passed.")


if __name__ == "__main__":
    main()
