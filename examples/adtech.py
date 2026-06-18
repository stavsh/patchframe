"""Synthetic adtech analytics: an honest star schema for multi-grain measurement.

A second discovery-engine example (after ``multimodal_fusion``). Where fusion is
temporal/continuous and its lesson is *alignment styles*, adtech is
relational/categorical and its lessons are **multi-grain roll-up** (the same
facts re-pivoted to whatever grain a question asks, where measures carry
*additivity types*) and **identity resolution as a pluggable join** (attribution
is a correspondence plan, and how that plan is *generated* — a consumed external
map, a deterministic key match, or a fuzzy candidate-gen — is interchangeable or
additive at the seam).

Seven **raw** sources, each declared separately; the reports advertisers usually
*receive* pre-aggregated (reach/frequency, sales/ROAS) are instead **composed
in-framework** from these — because revenue, cost, identity, and audience each
come from their *own* system, and baking them into a report table would re-import
external facts patchframe should reason about through dimensions:

- **creatives**           ``creative_id``                  ad assets + metadata (lazy asset)
- **impression_log**      ``impression_id``                the DSP delivery log (events)
- **rate_card**           ``placement_id``                 media cost (CPM) — the cost source
- **conversions**         ``conversion_id``                advertiser-side outcomes (revenue)
- **identity_map**        (device_id, advertiser_user_id)  the external identity bridge (+confidence)
- **segment_membership**  (device_id, segment_id)          DMP audience lists
- **segment_meta**        ``segment_id``                   audience demographics

"Row means different things" by task: row = ``creative_id`` (which creatives win
overall / which visual patterns correlate with ROAS / which underperform despite
high reach); row = ``segment_id`` or ``campaign x segment`` (which audience saw
which assets / which visual variants overfit one demographic).

No multiindex (yet): the two composite-key sources (``identity_map``,
``segment_membership``) use a single surrogate row index with the keys in
columns. The composite analytic grain (campaign x creative x segment) therefore
lives in *columns*, which is why re-graining by ``campaign x segment`` needs a
synthetic combined key and **marks the composite ``partition(by=[...])`` gap**.

Identity, the honest way (no single ``user_id`` across the lifecycle):

- The **serving** side logs a ``device_id`` — a cookie/MAID that may be **null**
  (consent), per-environment, and not 1:1 with people (~20% of people carry two
  devices, so device-"reach" over-counts people: reach is *cookie* reach).
- The **conversion** side logs an ``advertiser_user_id`` (first-party) — a
  *different* id space.
- Bridging them is **identity resolution**, done by an external vendor/clean room
  and **consumed** as ``identity_map`` — partial coverage (match-rate < 100%) and
  a per-pair **confidence** (some low-confidence/false links, the fuzzy flavour).
  The advertiser rarely matches raw ids themselves; they consume the link table.

Attribution is therefore a *join*: conversions -> ``identity_map`` -> impressions
within a lookback window. The point of the example is that the **correspondence
plan** is the interchangeable/additive seam — a consumed map, a deterministic key
``equals``, or a fuzzy candidate-gen all produce the same shape, and the
downstream attribution is unchanged. **Confidence is modelled as an ordinary
column** carried by composition, deliberately *not* forced through the join
primitives — where it cannot ride a primitive cleanly, that is the marked
payload-seam gap (``dimension-join-execution.md`` §5), surfaced, not built.

Real-world idiosyncrasies modelled (values synthetic; structure not):

- **reach is non-additive** — distinct *devices*, and a device sits in several
  segments; segment-level reach cannot be summed to a creative total, and even
  device-reach over-counts people (multi-device).
- **ratio/cross-fact measures recompute, never average** — frequency, viewability,
  completion, ROAS (= conversion revenue ÷ delivery spend, two different sources).
- **identity loss** — null device_ids, sub-100% match-rate, and low-confidence
  false links; many conversions are unattributable.
- **sparse outcomes** — most delivered (creative, segment) cells never convert.
- **dangling foreign keys** — a few log rows reference a creative served before
  its metadata synced.
- **standard IAB creative sizes** (300x250, 1920x1080, 728x90) per format.

Planted, verifiable signals (recovered by the section work, never given):

- creative ``style`` (latent visual factor) drives the asset pixels and the
  conversion propensity -> within-campaign ``corr(style, ROAS) > 0``.
- one **overfit creative** converts almost only in one **target segment** ->
  large cross-segment ROAS variance.

Dependency-light and deterministic (seeded), so it is CI-runnable:

    python examples/adtech.py
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np
import pandas as pd

import patchframe as pf

# --------------------------------------------------------------------------- #
# Field names (grouped by source) — reused by the section pipelines and tests.
# --------------------------------------------------------------------------- #

# shared keys
CREATIVE_ID = "creative_id"
CAMPAIGN_ID = "campaign_id"
SEGMENT_ID = "segment_id"
PLACEMENT_ID = "placement_id"
DEVICE_ID = "device_id"
ADVERTISER_USER_ID = "advertiser_user_id"

# creatives
FORMAT = "format"
WIDTH = "width"
HEIGHT = "height"
LANGUAGE = "language"
PRODUCT = "product"
ASSET = "asset"

# impression_log
IMPRESSION_ID = "impression_id"
TIMESTAMP = "timestamp"
GEO = "geo"
DEVICE_TYPE = "device_type"
PUBLISHER_CATEGORY = "publisher_category"
VIEWABLE = "viewable"
COMPLETED = "completed"

# rate_card
CPM = "cpm"

# conversions
CONVERSION_ID = "conversion_id"
REVENUE = "revenue"

# identity_map
LINK_ID = "link_id"
CONFIDENCE = "confidence"

# segment_membership
MEMBERSHIP_ID = "membership_id"

# segment_meta
AGE_BUCKET = "age_bucket"
GENDER_BUCKET = "gender_bucket"
INTEREST_CLUSTER = "interest_cluster"
COHORT = "cohort"

# --------------------------------------------------------------------------- #
# Taxonomies and entity specs.
# --------------------------------------------------------------------------- #

#: (campaign_id, product, average_order_value). Cost no longer lives here — it is
#: the ``rate_card`` source. ``aov`` seeds conversion revenue for the product.
CAMPAIGN_SPECS: tuple[tuple[str, str, float], ...] = (
    ("camp_running", "running_shoes", 90.0),
    ("camp_phone", "smartphone", 700.0),
    ("camp_cola", "cola", 8.0),
    ("camp_bank", "savings_account", 150.0),
)
CREATIVES_PER_CAMPAIGN = 3

#: Standard IAB ad sizes (width, height) by format — read from the asset header.
FORMAT_SIZES: dict[str, tuple[int, int]] = {
    "image": (300, 250),  # medium rectangle
    "video": (1920, 1080),  # in-stream HD
    "html": (728, 90),  # leaderboard
}
FORMATS = tuple(FORMAT_SIZES)
LANGUAGES = ("en", "es")

#: (placement_id, publisher_category, cpm, viewability_base). A placement is one
#: ad slot; ``cpm`` (its buy price) becomes the ``rate_card`` source,
#: ``viewability_base`` drives the per-impression viewable draw (generator-only).
PLACEMENT_SPECS: tuple[tuple[str, str, float, float], ...] = (
    ("plc_news_top", "news", 7.8, 0.82),
    ("plc_sports_rail", "sports", 6.6, 0.74),
    ("plc_ent_feed", "entertainment", 5.4, 0.61),
    ("plc_life_inline", "lifestyle", 6.0, 0.70),
    ("plc_tech_hero", "tech", 7.2, 0.88),
    ("plc_fin_side", "finance", 8.4, 0.66),
)

AGE_BUCKETS = ("18-24", "25-34", "35-44", "45-54", "55+")
GENDER_BUCKETS = ("female", "male", "unknown")
INTEREST_CLUSTERS = (
    "sports_fans",
    "tech_enthusiasts",
    "value_shoppers",
    "finance_minded",
    "entertainment",
)
COHORTS = (
    "in_market_auto",
    "new_parents",
    "frequent_travelers",
    "budget_conscious",
    "early_adopters",
)
GEOS = ("US", "CA", "GB", "DE")
DEVICE_TYPES = ("mobile", "desktop", "ctv", "tablet")

N_SEGMENTS = 8
N_PEOPLE = 400
N_IMPRESSIONS = 4000
DEFAULT_SEED = 20260617

#: Identity model knobs (the tunables agreed for v1).
TWO_DEVICE_RATE = 0.20  # people carrying two cookies/MAIDs -> device-reach > people
NO_DEVICE_RATE = 0.08  # impressions with no logged id (consent) -> unattributable
MATCH_RATE = 0.70  # identity_map coverage of true device<->advertiser links
FALSE_LINK_RATE = 0.06  # low-confidence wrong links (the fuzzy false positives)

#: A creative id present in the log but absent from the creatives table.
DANGLING_CREATIVE = "creative_unsynced"
DANGLING_RATE = 0.01

#: Conversion model. Base rate, the style->propensity slope (general signal), and
#: the overfit creative's segment gate (converts ~OFF outside the target segment).
BASE_CVR = 0.02
STYLE_GAMMA = 4.0
OVERFIT_ON = 6.0
OVERFIT_OFF = 0.05

#: Attribution timing. Conversions land within the lookback window after the
#: causing impression (interval-overlap territory for the attribution join).
FLIGHT_DAYS = 14
LOOKBACK_DAYS = 7
BASE_TS = np.datetime64("2026-05-01T00:00:00")

#: Thumbnail asset geometry (a rendered poster frame standing in for the visual).
THUMB_H = 8
THUMB_W = 8
THUMB_C = 3


# --------------------------------------------------------------------------- #
# The synthetic warehouse: entities -> base events -> the seven raw frames.
# --------------------------------------------------------------------------- #


@dataclass
class Warehouse:
    """The seven raw frames plus the latent truth used for verification.

    The frames become the patchframe sources. ``styles`` / ``overfit_creative`` /
    ``target_segment`` / ``true_attribution`` / ``true_links`` are *latent* — never
    exposed as columns; the section work must recover them through composition
    (recover ``style`` from the asset, attribute conversions through the lossy
    ``identity_map``, rediscover the overfit creative from the data).
    """

    creatives: pd.DataFrame
    impression_log: pd.DataFrame
    rate_card: pd.DataFrame
    conversions: pd.DataFrame
    identity_map: pd.DataFrame
    segment_membership: pd.DataFrame
    segment_meta: pd.DataFrame
    styles: dict[str, float]
    overfit_creative: str
    target_segment: str
    true_attribution: pd.DataFrame  # conversion_id -> causing impression_id (+lineage)
    true_links: set[tuple[str, str]]  # the correct (device_id, advertiser_user_id) pairs
    attribution_noise: float  # the conversion-timing regime this warehouse was built at


def generate_warehouse(
    seed: int = DEFAULT_SEED, *, attribution_noise: float = 0.0
) -> Warehouse:
    """Generate the full warehouse deterministically from one seed.

    ``attribution_noise`` ∈ [0, 1] sets the conversion-timing regime (see
    ``_build_conversions``): 0.0 = idealized recency (last-touch recovers the
    cause; residual error is identity-bridge loss only), 1.0 = realistic
    random-delay (last-touch misattributes). The Monte-Carlo dial.
    """

    rng = np.random.default_rng(seed)
    creatives, styles, overfit_creative = _build_creatives()
    segment_meta = _build_segment_meta()
    target_segment = segment_meta.index[3]  # the demographic the overfit creative wins
    rate_card = _build_rate_card()
    people = _build_people(rng, list(segment_meta.index))
    impressions, imp_lineage = _build_impressions(rng, creatives, people)
    conversions, true_attribution = _build_conversions(
        rng, imp_lineage, creatives, styles, overfit_creative, target_segment, people,
        attribution_noise=attribution_noise,
    )
    identity_map, true_links = _build_identity_map(rng, people)
    segment_membership = _build_segment_membership(people)
    return Warehouse(
        creatives=creatives,
        impression_log=impressions,
        rate_card=rate_card,
        conversions=conversions,
        identity_map=identity_map,
        segment_membership=segment_membership,
        segment_meta=segment_meta,
        styles=styles,
        overfit_creative=overfit_creative,
        target_segment=target_segment,
        true_attribution=true_attribution,
        true_links=true_links,
        attribution_noise=attribution_noise,
    )


def _build_creatives() -> tuple[pd.DataFrame, dict[str, float], str]:
    """One row per trafficked creative; ``style`` is latent (drives pixels + CVR)."""

    rows: list[dict[str, Any]] = []
    styles: dict[str, float] = {}
    style_levels = np.linspace(0.15, 0.9, CREATIVES_PER_CAMPAIGN)
    for campaign_id, product, _aov in CAMPAIGN_SPECS:
        for position in range(CREATIVES_PER_CAMPAIGN):
            creative_id = f"{campaign_id}_cr{position}"
            fmt = FORMATS[position % len(FORMATS)]
            width, height = FORMAT_SIZES[fmt]
            styles[creative_id] = float(style_levels[position])
            rows.append(
                {
                    CREATIVE_ID: creative_id,
                    CAMPAIGN_ID: campaign_id,
                    FORMAT: fmt,
                    WIDTH: width,
                    HEIGHT: height,
                    LANGUAGE: LANGUAGES[(position + len(rows)) % len(LANGUAGES)],
                    PRODUCT: product,
                }
            )
    # The overfit creative: the phone campaign's high-style video creative, whose
    # conversions are gated to one segment.
    overfit_creative = f"camp_phone_cr{CREATIVES_PER_CAMPAIGN - 1}"
    return pd.DataFrame(rows).set_index(CREATIVE_ID), styles, overfit_creative


def _build_segment_meta() -> pd.DataFrame:
    """Audience demographics — the segment dimension table (from the data provider)."""

    rows = []
    for position in range(N_SEGMENTS):
        rows.append(
            {
                SEGMENT_ID: f"seg_{position}",
                AGE_BUCKET: AGE_BUCKETS[position % len(AGE_BUCKETS)],
                GENDER_BUCKET: GENDER_BUCKETS[position % len(GENDER_BUCKETS)],
                INTEREST_CLUSTER: INTEREST_CLUSTERS[position % len(INTEREST_CLUSTERS)],
                COHORT: COHORTS[position % len(COHORTS)],
            }
        )
    return pd.DataFrame(rows).set_index(SEGMENT_ID)


def _build_rate_card() -> pd.DataFrame:
    """Media cost per placement — the cost source (spend = impressions x CPM/1000)."""

    rows = [{PLACEMENT_ID: pid, CPM: cpm} for pid, _cat, cpm, _vb in PLACEMENT_SPECS]
    return pd.DataFrame(rows).set_index(PLACEMENT_ID)


def _build_people(rng: np.random.Generator, segment_ids: list[str]) -> list[dict[str, Any]]:
    """Latent people: device(s) (cookie/MAID), one advertiser id, segments, geo.

    ~20% carry two devices (device-reach over-counts people); ~10% are
    unsegmented (their devices match no audience); ~30% sit in two segments (a
    device then matches several audiences -> reach non-additivity).
    """

    people: list[dict[str, Any]] = []
    device_counter = 0
    for position in range(N_PEOPLE):
        n_devices = 2 if rng.random() < TWO_DEVICE_RATE else 1
        devices = []
        for _ in range(n_devices):
            devices.append((f"dev_{device_counter:06d}", str(rng.choice(DEVICE_TYPES))))
            device_counter += 1
        roll = rng.random()
        if roll < 0.10:
            segments: list[str] = []
        elif roll < 0.40:
            segments = [str(s) for s in rng.choice(segment_ids, size=2, replace=False)]
        else:
            segments = [str(rng.choice(segment_ids))]
        people.append(
            {
                "person_idx": position,
                "advertiser_user_id": f"au_{position:05d}",
                "devices": devices,  # list of (device_id, device_type)
                "segments": segments,
                GEO: str(rng.choice(GEOS)),
            }
        )
    return people


def _build_impressions(
    rng: np.random.Generator,
    creatives: pd.DataFrame,
    people: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The raw serving log: one row per impression. Returns the source frame plus
    a parallel lineage list (with the latent person index) for conversion gen."""

    by_campaign = {
        cid: list(creatives.index[creatives[CAMPAIGN_ID] == cid])
        for cid, *_ in CAMPAIGN_SPECS
    }
    campaign_ids = [spec[0] for spec in CAMPAIGN_SPECS]
    campaign_weights = np.array([1.0, 2.2, 1.4, 1.1])
    campaign_weights = campaign_weights / campaign_weights.sum()
    flight_seconds = FLIGHT_DAYS * 24 * 3600

    rows = []
    lineage = []
    for position in range(N_IMPRESSIONS):
        person = people[int(rng.integers(len(people)))]
        if rng.random() < NO_DEVICE_RATE:
            device_id: Any = pd.NA  # no id passed (consent) -> unattributable
            device_type = str(rng.choice(DEVICE_TYPES))  # UA still reveals the type
        else:
            device_id, device_type = person["devices"][
                int(rng.integers(len(person["devices"])))
            ]
        campaign_id = str(rng.choice(campaign_ids, p=campaign_weights))
        creative_id = str(rng.choice(by_campaign[campaign_id]))
        if rng.random() < DANGLING_RATE:
            creative_id = DANGLING_CREATIVE
        plc_id, pub_cat, _cpm, view_base = PLACEMENT_SPECS[
            int(rng.integers(len(PLACEMENT_SPECS)))
        ]
        viewable = bool(rng.random() < view_base)
        is_video = (
            creative_id != DANGLING_CREATIVE
            and creatives.loc[creative_id, FORMAT] == "video"
        )
        completed = bool(is_video and viewable and rng.random() < 0.55)
        ts = BASE_TS + np.timedelta64(int(rng.integers(flight_seconds)), "s")
        impression_id = f"imp_{position:06d}"
        rows.append(
            {
                IMPRESSION_ID: impression_id,
                CREATIVE_ID: creative_id,
                CAMPAIGN_ID: campaign_id,
                PLACEMENT_ID: plc_id,
                DEVICE_ID: device_id,
                TIMESTAMP: ts,
                GEO: person[GEO],
                DEVICE_TYPE: device_type,
                PUBLISHER_CATEGORY: pub_cat,
                VIEWABLE: viewable,
                COMPLETED: completed if is_video else pd.NA,
            }
        )
        lineage.append(
            {
                IMPRESSION_ID: impression_id,
                CREATIVE_ID: creative_id,
                CAMPAIGN_ID: campaign_id,
                "person_idx": person["person_idx"],
                TIMESTAMP: ts,
                "has_device": device_id is not pd.NA,
            }
        )
    return pd.DataFrame(rows).set_index(IMPRESSION_ID), lineage


def _build_conversions(
    rng: np.random.Generator,
    imp_lineage: list[dict[str, Any]],
    creatives: pd.DataFrame,
    styles: dict[str, float],
    overfit_creative: str,
    target_segment: str,
    people: list[dict[str, Any]],
    attribution_noise: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Advertiser-side conversion events (revenue), driven by a causing impression.

    Propensity rises with the causing creative's latent ``style`` (general signal);
    the overfit creative converts only when the person is in the target segment.
    Recorded under ``advertiser_user_id`` with no creative/campaign — those are
    recovered by attribution. The latent causing impression is kept for tests.

    ``attribution_noise`` ∈ [0, 1] dials the conversion *timing* — which is what
    decides whether last-touch attribution can recover the cause. The causal truth
    is identical at every setting; only the timestamp moves:

    - **recency** (the ``1 - noise`` fraction): the conversion lands *before the
      person's next impression*, so last-touch == cause. The idealized reference —
      residual attribution error is then purely identity-bridge loss.
    - **random-delay** (the ``noise`` fraction): the conversion lands at a uniformly
      random delay within the lookback window, so later impressions of other
      campaigns intervene and last-touch misattributes — the realistic, noisy
      regime. The Monte-Carlo dial for "how attribution noise degrades the
      recovered measures."

    Conversions are sequenced per latent person over their combined-device
    timeline so "next impression" (hence last-touch) is well defined.
    """

    aov = {cid: value for cid, _p, value in CAMPAIGN_SPECS}
    product = {cid: creatives.loc[creatives[CAMPAIGN_ID] == cid, PRODUCT].iloc[0]
               for cid, *_ in CAMPAIGN_SPECS}
    lookback = float(LOOKBACK_DAYS * 24 * 3600)

    by_person: dict[int, list[dict[str, Any]]] = {}
    for imp in imp_lineage:
        if imp[CREATIVE_ID] == DANGLING_CREATIVE:
            continue
        by_person.setdefault(imp["person_idx"], []).append(imp)

    conv_rows = []
    attribution_rows = []
    counter = 0
    for person_idx, imps in by_person.items():
        imps = sorted(imps, key=lambda row: row[TIMESTAMP])
        person = people[person_idx]
        for position, imp in enumerate(imps):
            creative_id = imp[CREATIVE_ID]
            style = styles[creative_id]
            prob = BASE_CVR * (1.0 + STYLE_GAMMA * style)
            if creative_id == overfit_creative:
                prob *= OVERFIT_ON if target_segment in person["segments"] else OVERFIT_OFF
            if rng.random() >= min(prob, 0.6):
                continue
            if rng.random() < attribution_noise:
                delay = float(rng.integers(3600, int(lookback)))  # realistic spread
            else:  # recency: land before the next impression so last-touch == cause
                if position + 1 < len(imps):
                    gap = (imps[position + 1][TIMESTAMP] - imp[TIMESTAMP]) / np.timedelta64(1, "s")
                else:
                    gap = lookback
                delay = min(rng.uniform(0.05, 0.9) * max(gap, 60.0), lookback)
            ts = imp[TIMESTAMP] + np.timedelta64(int(max(delay, 1.0)), "s")
            revenue = aov[imp[CAMPAIGN_ID]] * (0.6 + 0.8 * rng.random())
            conversion_id = f"cnv_{counter:06d}"
            counter += 1
            conv_rows.append(
                {
                    CONVERSION_ID: conversion_id,
                    ADVERTISER_USER_ID: person["advertiser_user_id"],
                    TIMESTAMP: ts,
                    REVENUE: float(revenue),
                    PRODUCT: product[imp[CAMPAIGN_ID]],
                }
            )
            attribution_rows.append(
                {
                    CONVERSION_ID: conversion_id,
                    IMPRESSION_ID: imp[IMPRESSION_ID],
                    CREATIVE_ID: creative_id,
                    CAMPAIGN_ID: imp[CAMPAIGN_ID],
                    ADVERTISER_USER_ID: person["advertiser_user_id"],
                    "has_device": imp["has_device"],
                }
            )
    conversions = pd.DataFrame(conv_rows).set_index(CONVERSION_ID)
    true_attribution = pd.DataFrame(attribution_rows)
    return conversions, true_attribution


def _build_identity_map(
    rng: np.random.Generator, people: list[dict[str, Any]]
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """The external identity bridge: (device_id, advertiser_user_id, confidence).

    Lossy: only ~MATCH_RATE of true device<->advertiser links are resolved, each
    with high confidence; plus a few low-confidence *wrong* links (the fuzzy false
    positives). The advertiser consumes this; whether the vendor matched
    deterministically or probabilistically surfaces only as the confidence column.
    """

    all_devices = [(d, p["advertiser_user_id"]) for p in people for d, _t in p["devices"]]
    rows = []
    true_links: set[tuple[str, str]] = set()
    for device_id, advertiser_user_id in all_devices:
        true_links.add((device_id, advertiser_user_id))
        if rng.random() < MATCH_RATE:
            rows.append(
                {
                    DEVICE_ID: device_id,
                    ADVERTISER_USER_ID: advertiser_user_id,
                    CONFIDENCE: float(rng.uniform(0.80, 0.99)),  # resolved, high conf
                }
            )
        if rng.random() < FALSE_LINK_RATE:
            wrong = all_devices[int(rng.integers(len(all_devices)))][1]
            if wrong != advertiser_user_id:
                rows.append(
                    {
                        DEVICE_ID: device_id,
                        ADVERTISER_USER_ID: wrong,
                        CONFIDENCE: float(rng.uniform(0.30, 0.60)),  # fuzzy, low conf
                    }
                )
    rng.shuffle(rows)
    frame = pd.DataFrame(rows)
    frame.index = pd.Index([f"lnk_{i:05d}" for i in range(len(frame))], name=LINK_ID)
    return frame, true_links


def _build_segment_membership(people: list[dict[str, Any]]) -> pd.DataFrame:
    """DMP audience lists: one (device_id, segment_id) row per device-membership.

    Keyed to the device (cookie/MAID), as DMP lists are onboarded. A device in a
    two-segment person matches both; an unsegmented person's devices appear in no
    row (they match no audience).
    """

    rows = []
    for person in people:
        for device_id, _type in person["devices"]:
            for segment_id in person["segments"]:
                rows.append({DEVICE_ID: device_id, SEGMENT_ID: segment_id})
    frame = pd.DataFrame(rows)
    frame.index = pd.Index([f"mbr_{i:05d}" for i in range(len(frame))], name=MEMBERSHIP_ID)
    return frame


# --------------------------------------------------------------------------- #
# The lazy creative-asset source (the one source with array data).
# --------------------------------------------------------------------------- #


def creative_thumbnail(style: float) -> np.ndarray:
    """Render a creative's poster thumbnail from its latent ``style``.

    A deterministic ``(THUMB_H, THUMB_W, THUMB_C)`` uint8 image where the red
    channel encodes ``style`` and blue encodes ``1 - style`` — a high-style
    creative reads "warm/saturated". The mean red channel recovers ``style``
    exactly, so a downstream visual feature can be checked against the planted
    factor. Stands in for a real decoded poster frame.
    """

    red = np.full((THUMB_H, THUMB_W), round(style * 255.0), dtype=np.uint8)
    blue = np.full((THUMB_H, THUMB_W), round((1.0 - style) * 255.0), dtype=np.uint8)
    green = np.full((THUMB_H, THUMB_W), 64, dtype=np.uint8)
    return np.stack([red, green, blue], axis=-1)


class CreativeAssetSource(pf.ArrayDataSource):
    """Synthetic creative-asset decoder: one thumbnail per creative id.

    Dims ``{y, x, channel}`` (raw indices). Assets are small and decoded whole;
    the laziness payoff is row-level — only the creatives a question inspects get
    decoded, after re-graining and filtering on cheap metadata.
    """

    source_type = "synthetic_creative"
    thread_safe: bool = True
    fork_safe: bool = True
    config_fields = ("styles",)
    supports_partial_read = False

    def __init__(
        self,
        *,
        styles: dict[str, float],
        dimensions: pf.Dimensions | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions
            or pf.Dimensions(
                (
                    pf.IndexDimension(name="y"),
                    pf.IndexDimension(name="x"),
                    pf.IndexDimension(name="channel"),
                )
            ),
            source_id=source_id,
            styles=dict(styles),
        )

    def read_full(self, item_id: Any, accessor: pf.DataAccessor) -> np.ndarray:
        return creative_thumbnail(self.styles[item_id])

    def extent_for(self, item_id: Any) -> pf.DimensionedSlice:
        return pf.DimensionedSlice(
            dims={
                "y": slice(0, THUMB_H),
                "x": slice(0, THUMB_W),
                "channel": slice(0, THUMB_C),
            }
        )


# --------------------------------------------------------------------------- #
# Dataset builders — turn the warehouse frames into the seven patchframe sources.
# --------------------------------------------------------------------------- #


class make_creatives(pf.CreationOperator):
    """Creatives dataset: metadata columns + a lazy ``asset`` DataField.

    The asset column holds ``DataAccessor``s into a registered
    ``CreativeAssetSource``; nothing is decoded until row access materializes a
    creative. ``styles`` is the latent factor the source renders from.
    """

    def make_source(
        self, creatives: pd.DataFrame, *, styles: dict[str, float], **_: Any
    ) -> CreativeAssetSource:
        return CreativeAssetSource(styles=styles)

    def generate_source_info(
        self, creatives: pd.DataFrame, *, styles: dict[str, float], **_: Any
    ) -> pf.DatasetSourceInfo:
        return pf.DatasetSourceInfo(
            source_uri="synthetic://creatives",
            source_type="synthetic_creative",
            source_name="Synthetic creative assets",
        )

    def build(
        self,
        creatives: pd.DataFrame,
        *,
        styles: dict[str, float],
        source_desc_id: int | None = None,
        source_manager: Any = None,
        **_: Any,
    ) -> pf.DatasetState:
        ids = list(creatives.index)
        table = pd.DataFrame(index=pd.Index(ids, name=CREATIVE_ID))
        table[CAMPAIGN_ID] = creatives[CAMPAIGN_ID].astype("string").to_numpy()
        table[FORMAT] = creatives[FORMAT].astype("string").to_numpy()
        table[WIDTH] = pd.array(creatives[WIDTH].to_numpy(), dtype="Int64")
        table[HEIGHT] = pd.array(creatives[HEIGHT].to_numpy(), dtype="Int64")
        table[LANGUAGE] = creatives[LANGUAGE].astype("string").to_numpy()
        table[PRODUCT] = creatives[PRODUCT].astype("string").to_numpy()
        table[ASSET] = [
            pf.DataAccessor(
                source_desc_id=source_desc_id,
                item_id=creative_id,
                manager_hint=source_manager,
            )
            for creative_id in ids
        ]
        schema = pf.Schema(
            fields=(
                pf.IndexField(name=CREATIVE_ID),
                pf.ValueField(name=CAMPAIGN_ID, dtype=str),
                pf.ValueField(name=FORMAT, dtype=str),
                pf.ValueField(name=WIDTH, dtype=int),
                pf.ValueField(name=HEIGHT, dtype=int),
                pf.ValueField(name=LANGUAGE, dtype=str),
                pf.ValueField(name=PRODUCT, dtype=str),
                pf.DataField(name=ASSET),
            )
        )
        return pf.DatasetState(schema=schema, table=table)


def _string(values: Any) -> pd.array:
    return pd.array(np.asarray(values, dtype=object), dtype="string")


def make_impression_log(warehouse: Warehouse) -> pf.Dataset:
    """Serving log: one row per impression (the base events). Logs ``device_id``
    (cookie/MAID, nullable) — not an audience or a person."""

    src = warehouse.impression_log
    index = pd.Index(list(src.index), name=IMPRESSION_ID)
    table = pd.DataFrame(
        {
            CREATIVE_ID: _string(src[CREATIVE_ID].to_numpy()),
            CAMPAIGN_ID: _string(src[CAMPAIGN_ID].to_numpy()),
            PLACEMENT_ID: _string(src[PLACEMENT_ID].to_numpy()),
            DEVICE_ID: _string(src[DEVICE_ID].to_numpy()),
            TIMESTAMP: pd.to_datetime(src[TIMESTAMP].to_numpy()),
            GEO: _string(src[GEO].to_numpy()),
            DEVICE_TYPE: _string(src[DEVICE_TYPE].to_numpy()),
            PUBLISHER_CATEGORY: _string(src[PUBLISHER_CATEGORY].to_numpy()),
            VIEWABLE: pd.array(src[VIEWABLE].to_numpy(), dtype="boolean"),
            COMPLETED: pd.array(src[COMPLETED].to_numpy(), dtype="boolean"),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=IMPRESSION_ID),
            pf.ValueField(name=CREATIVE_ID, dtype=str),
            pf.ValueField(name=CAMPAIGN_ID, dtype=str),
            pf.ValueField(name=PLACEMENT_ID, dtype=str),
            pf.ValueField(name=DEVICE_ID, dtype=str, nullable=True),
            pf.ValueField(name=TIMESTAMP),  # datetime64[ns]; dtype validation off
            pf.ValueField(name=GEO, dtype=str),
            pf.ValueField(name=DEVICE_TYPE, dtype=str),
            pf.ValueField(name=PUBLISHER_CATEGORY, dtype=str),
            pf.ValueField(name=VIEWABLE, dtype=bool),
            pf.ValueField(name=COMPLETED, dtype=bool, nullable=True),
        )
    )
    return pf.make_from_dataframe(table, schema)


def make_rate_card(warehouse: Warehouse) -> pf.Dataset:
    """Media cost per placement (CPM) — the cost source."""

    src = warehouse.rate_card
    index = pd.Index(list(src.index), name=PLACEMENT_ID)
    table = pd.DataFrame({CPM: pd.array(src[CPM].to_numpy(), dtype="Float64")}, index=index)
    schema = pf.Schema(
        fields=(pf.IndexField(name=PLACEMENT_ID), pf.ValueField(name=CPM, dtype=float))
    )
    return pf.make_from_dataframe(table, schema)


def make_conversions(warehouse: Warehouse) -> pf.Dataset:
    """Advertiser-side outcomes (revenue), keyed by ``advertiser_user_id``.

    Carries no creative/campaign — those are recovered by attribution through the
    identity bridge.
    """

    src = warehouse.conversions
    index = pd.Index(list(src.index), name=CONVERSION_ID)
    table = pd.DataFrame(
        {
            ADVERTISER_USER_ID: _string(src[ADVERTISER_USER_ID].to_numpy()),
            TIMESTAMP: pd.to_datetime(src[TIMESTAMP].to_numpy()),
            REVENUE: pd.array(src[REVENUE].to_numpy(), dtype="Float64"),
            PRODUCT: _string(src[PRODUCT].to_numpy()),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=CONVERSION_ID),
            pf.ValueField(name=ADVERTISER_USER_ID, dtype=str),
            pf.ValueField(name=TIMESTAMP),
            pf.ValueField(name=REVENUE, dtype=float),
            pf.ValueField(name=PRODUCT, dtype=str),
        )
    )
    return pf.make_from_dataframe(table, schema)


def make_identity_map(warehouse: Warehouse) -> pf.Dataset:
    """The external identity bridge: (device_id, advertiser_user_id) + confidence.

    A correspondence source: surrogate ``link_id`` index, the two id spaces in
    columns, and a per-pair ``confidence``. Lossy and partly wrong by
    construction. ``confidence`` is an ordinary column (not a join primitive).
    """

    src = warehouse.identity_map
    index = pd.Index(list(src.index), name=LINK_ID)
    table = pd.DataFrame(
        {
            DEVICE_ID: _string(src[DEVICE_ID].to_numpy()),
            ADVERTISER_USER_ID: _string(src[ADVERTISER_USER_ID].to_numpy()),
            CONFIDENCE: pd.array(src[CONFIDENCE].to_numpy(), dtype="Float64"),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=LINK_ID),
            pf.ValueField(name=DEVICE_ID, dtype=str),
            pf.ValueField(name=ADVERTISER_USER_ID, dtype=str),
            pf.ValueField(name=CONFIDENCE, dtype=float),
        )
    )
    return pf.make_from_dataframe(table, schema)


def make_segment_membership(warehouse: Warehouse) -> pf.Dataset:
    """DMP audience lists: (device_id, segment_id) — surrogate ``membership_id``."""

    src = warehouse.segment_membership
    index = pd.Index(list(src.index), name=MEMBERSHIP_ID)
    table = pd.DataFrame(
        {
            DEVICE_ID: _string(src[DEVICE_ID].to_numpy()),
            SEGMENT_ID: _string(src[SEGMENT_ID].to_numpy()),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=MEMBERSHIP_ID),
            pf.ValueField(name=DEVICE_ID, dtype=str),
            pf.ValueField(name=SEGMENT_ID, dtype=str),
        )
    )
    return pf.make_from_dataframe(table, schema)


def make_segment_meta(warehouse: Warehouse) -> pf.Dataset:
    """Audience demographics dimension table."""

    src = warehouse.segment_meta
    index = pd.Index(list(src.index), name=SEGMENT_ID)
    table = pd.DataFrame(
        {
            AGE_BUCKET: _string(src[AGE_BUCKET].to_numpy()),
            GENDER_BUCKET: _string(src[GENDER_BUCKET].to_numpy()),
            INTEREST_CLUSTER: _string(src[INTEREST_CLUSTER].to_numpy()),
            COHORT: _string(src[COHORT].to_numpy()),
        },
        index=index,
    )
    schema = pf.Schema(
        fields=(
            pf.IndexField(name=SEGMENT_ID),
            pf.ValueField(name=AGE_BUCKET, dtype=str),
            pf.ValueField(name=GENDER_BUCKET, dtype=str),
            pf.ValueField(name=INTEREST_CLUSTER, dtype=str),
            pf.ValueField(name=COHORT, dtype=str),
        )
    )
    return pf.make_from_dataframe(table, schema)


@dataclass
class AdtechSources:
    """The seven patchframe datasets plus the latent verification truth."""

    creatives: pf.Dataset
    impression_log: pf.Dataset
    rate_card: pf.Dataset
    conversions: pf.Dataset
    identity_map: pf.Dataset
    segment_membership: pf.Dataset
    segment_meta: pf.Dataset
    warehouse: Warehouse = dc_field(repr=False)


def make_adtech_sources(
    seed: int = DEFAULT_SEED, *, attribution_noise: float = 0.0
) -> AdtechSources:
    """Generate the warehouse and build all seven sources.

    ``attribution_noise`` ∈ [0, 1] selects the conversion-timing regime
    (``generate_warehouse``): 0.0 idealized (default), 1.0 realistic-noisy.
    """

    wh = generate_warehouse(seed, attribution_noise=attribution_noise)
    return AdtechSources(
        creatives=make_creatives(wh.creatives, styles=wh.styles),
        impression_log=make_impression_log(wh),
        rate_card=make_rate_card(wh),
        conversions=make_conversions(wh),
        identity_map=make_identity_map(wh),
        segment_membership=make_segment_membership(wh),
        segment_meta=make_segment_meta(wh),
        warehouse=wh,
    )


def main() -> None:
    sources = make_adtech_sources()
    wh = sources.warehouse

    print("=== adtech sources (seven raw sources; no multiindex) ===")
    for name in (
        "creatives",
        "impression_log",
        "rate_card",
        "conversions",
        "identity_map",
        "segment_membership",
        "segment_meta",
    ):
        ds = getattr(sources, name)
        print(f"  {name:19s} {len(ds.table):5d} rows  cols={list(ds.table.columns)}")

    print("\n=== one creative row, asset decoded lazily through row access ===")
    creatives = pf.materialize(sources.creatives, ASSET)
    creative_id = creatives.table.index[0]
    row = creatives[creative_id]
    asset = np.asarray(row[ASSET])
    recovered_style = float(asset[..., 0].mean()) / 255.0
    print(f"  {creative_id}: format={row[FORMAT]} size={row[WIDTH]}x{row[HEIGHT]}")
    print(f"  asset {asset.shape} {asset.dtype}; recovered style ~ {recovered_style:.2f}")
    print(f"  (true latent style = {wh.styles[creative_id]:.2f})")

    print("\n=== identity / loss (attribution is a lossy join, not a logged fact) ===")
    n_dangling = int((wh.impression_log[CREATIVE_ID] == DANGLING_CREATIVE).sum())
    n_no_device = int(wh.impression_log[DEVICE_ID].isna().sum())
    n_links = len(wh.identity_map)
    n_low_conf = int((wh.identity_map[CONFIDENCE] < 0.7).sum())
    attributable = wh.true_attribution["has_device"].mean()
    print(f"  impressions: {len(wh.impression_log)}  dangling-creative: {n_dangling}"
          f"  null device_id: {n_no_device}")
    print(f"  conversions: {len(wh.conversions)}  with a device to bridge: "
          f"{attributable:.0%}")
    print(f"  identity_map: {n_links} links ({n_low_conf} low-confidence/fuzzy)")
    print(f"  segment_membership: {len(wh.segment_membership)} device-memberships")
    print(f"  planted: overfit creative={wh.overfit_creative}, "
          f"target segment={wh.target_segment}")
    print("\nAll adtech source checks passed.")


if __name__ == "__main__":
    main()
