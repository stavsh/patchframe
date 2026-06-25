# Adtech Example: Findings and Structural Work

Status: findings log, agreed 2026-06-18. Records what the adtech example surfaced
as a forcing function — the example's honest verdict and the framework gaps it
forced — and the agreed work order for addressing them. The example lives in
`examples/adtech.py` (sources), `examples/adtech_analysis.py` (the tabular
roll-up), and `examples/adtech_journeys.py` (the sequence pivot). Nothing here is
built yet; this is the bridge from "examples as the discovery engine" to the
structural work they justify.

Cross-references:

- `partition-aggregate.md` §5 — the `reduce`/AggSpec vocabulary this sharpens.
- `lazy-and-bundle.md` §3, §7 — the lazy `BundleField` cell / out-of-core substrate.
- `design-constraints.md` §9 — the `.table` / authoring surface.
- `join-dimensions-identity.md`, `dimension-join-execution.md` — the join/identity
  model the example reuses (and §10, UC1/UC2 — field expressions and tensor payload).
- `schema-specs.md` — plan typing, adjacent to the honesty concern.

## 1. The example

A synthetic adtech measurement stack, deliberately *not* a single table:

- **Seven raw sources** (`adtech.py`), each its own system, no multiindex
  (surrogate keys): `creatives` (+ a lazy thumbnail asset), `impression_log`
  (events, `device_id`), `rate_card` (cost), `conversions` (revenue,
  `advertiser_user_id`), `identity_map` (the lossy device↔advertiser bridge, with
  per-pair `confidence`), `segment_membership`, `segment_meta`.
- **Identity is honest**: no single user id across the lifecycle; serving logs a
  device/cookie (nullable), conversions log a first-party id, and resolution is a
  consumed, lossy `identity_map`. Reach is *device* reach (multi-device
  over-counts people).
- **Two pivots on the same substrate** ("row means different things"):
  - **Tabular roll-up** (`adtech_analysis.py`): attribution foundation
    (`device_id`→`identity_map`→`advertiser_user_id` join, lookback, last-touch)
    → creative-grain roll-up with measure additivity (additive sums, non-additive
    reach, recomputed ROAS) → visual↔ROAS (lazy asset) → audience overfit
    (creative×segment) → an attribution-noise Monte-Carlo sensitivity study.
  - **Sequence journeys** (`adtech_journeys.py`): `partition` by resolved identity
    into per-identity time-ordered exposure **sequence fibers**, with attributed
    conversion labels, streamed as `(sequence, label)` samples via `rows()` for a
    DataLoader.

## 2. The verdict (honest — this is what makes it an argument)

- **The roll-up is pandas-shaped.** `groupby().agg()` + `merge` does Sections A–C
  natively, faster, with *less* ceremony; the patchframe version is *more*
  ceremony (the `.table` escapes, the N-pass `map_fields`, the join choreography).
  It was a strong **gap-surfacing** exercise and the foundation is reusable, but
  it does not sell patchframe.
- **The journeys pivot is patchframe-shaped.** A dataset *of variable-length
  sequences* of heterogeneous features, streamed to a DataLoader, with
  cross-device identity unification and identity-aligned labels falling out of the
  composition — none of which pandas holds cleanly.
- **The in-memory payoff is structural, not perf** (measured): the data is the
  fibers, all resident, so there is no in-memory memory/streaming win. What
  patchframe buys in v1 is `rows()`-as-DataLoader-for-free, one-substrate-two-pivots,
  and lineage through the composition. **The perf payoff is out-of-core** (lazy
  fiber cell). The argument for patchframe is therefore *use-case-specific*
  (multi-source identity resolution → sequence assembly → streaming training data)
  and credible precisely because it does not overclaim the in-memory case.

## 3. Forcing-function findings (the gaps)

Each: what it is, where it surfaced, the direction, and the tier (tactical /
protocol / foundational).

### 3.1 `reduce` / AggSpec vocabulary — *protocol; least well-defined; start here*

Roll-up aggregations (sum spend/revenue/impressions/conversions, distinct-count
reach, the ROAS ratio) and journey labels are **N separate
`map_fields(fn, out=...)` passes**, each an opaque callable. The declared
aggregation vocabulary (`partition-aggregate.md` §5) — *semantics + a vectorized
strategy + a contract-tested honest output type* — collapses the passes, gives the
engine a fast path, and makes outputs honest (subsuming 3.3 for declared aggs).
The example gives it the consumer §5 said was missing: prior consumers only
*collected*; the roll-up does real numeric reduction (sums, ratios, distinct
counts). **Least well-defined of the set → first to design.**

### 3.2 Constrained `.table` escape — *protocol; least well-defined; start here*

`attribute()`, the ROAS/spend/composite-key derivations, and the journey lookups
drop to `.table` + pandas because the operation is not framework-expressible.
`.table` as a *transform* surface bypasses the operator contracts (identity,
couplings, transitions) the package exists to maintain — distinct from sanctioned
`.table` use (inspection; a fiber-reduce *inside* a `map_fields` fn). The
direction: name that boundary, and provide a *constrained* escape — a `pipe`/
`apply` taking a `table → table` fn but **re-validating schema/identity
afterward** — versus growing the operator vocabulary. The right mix is itself the
open question. **Least well-defined of the set → first to design.**

### 3.3 `map_fields` return-honesty — *tactical (partial of 3.1)*

An opaque `map_fields` fn returns whatever it likes, so the declared output field
is not honest (`out="x"` is typed but unchecked). Minimum fix: validate the return
against the declared output dtype and fail loud. The fuller answer is 3.1.

### 3.4 `sort` / ordered fibers — *tactical*

No `sort` operator exists, so the journey sequence order is applied at
sample-assembly (`_assemble_sequence` sorts the fiber on exit) — and sorting via
`.table` would be the 3.2 smell. Direction: a `sort` operator, and/or order as a
first-class *fiber* property (ordered fibers).

### 3.5 Composite-key `partition` → multiindex + nullable index — *foundational*

The (campaign × segment) and (creative × segment) grains need composite keys; v1
mints a synthetic combined key (`creative|segment`). Real support is a composite
*row identity*, colliding with today's "one `IndexField`, unique, non-null = row
identity" invariant: it needs a multiindex (or a first-class composite-key field)
*and* nullable index labels (the unattributed row), and reopens "null keys error
in v1" for `partition`. Touches `IndexIdentity`/`IndexField` — **design-note first.**

### 3.6 Lazy `BundleField` cell / out-of-core fiber streaming — *foundational*

The journeys hold all fibers in memory; the real perf payoff (streaming millions
of journeys whose features cannot all be resident) needs the lazy fiber cell /
streaming `partition` — the workload-gated executor (`lazy-and-bundle.md` §7).
**The journey example is plausibly the workload that gates it.** Design-note first;
do not build the executor blind.

### 3.7 Output-assignment sugar — *tactical*

Repeated `map_fields(..., out="x")` is the clunky part of `creative_performance`.
Auto-named output + field-return assignment (`ds.new_field("x").loc[:] = handle`,
`ctx["x"] = handle`) over the landed assignment conventions. Fork to settle: does
assigning a *handle* defer (record a coupling named by the LHS) or compute
(assign-assigns-values)? Lean: defer.

### 3.8 Field-expression / lightweight dimension lookup — *medium (UC1)*

Derived columns (per-impression `spend` = cpm/1000; `ROAS` = rev/spend; the
composite key; the per-creative feature lookup) are done via `assign` over
`.table`-read values, because joining a value column to a dimension table's
*index* is awkward and field-expression algebra is deferred (`dimension-join-
execution.md` §10, UC1). A `pl.col`-style symbolic operand layer is the general
form.

## 4. The returning arrow (v2 north star; not built)

The linear pipeline (load attribution candidates → train) is one-way: torch owns
the tensors after the `rows()` handoff, so **tensor-payload integrity is not
required here**. It becomes load-bearing only under a *feedback loop*: candidates
→ train → *model attributions* → re-evaluate per-`user_id` ROAS/frequency **inside
patchframe** → compare to total ROAS as **another (consistency / conservation)
supervision signal**. There the model's outputs (tensors) re-enter patchframe's
aggregation operators (the round-trip), which is where tensor integrity and the
payload seam matter (UC2 territory). The sequence pivot also reframes the project:
heuristic last-touch labels vs a transformer that *learns* attribution, with the
attribution-noise Monte Carlo becoming the *supervision-quality* story.

## 5. Work order

Three tiers (§3). The agreed prioritization (user, 2026-06-18):

- **Prefer the difficult/high-impact paths over the easy tactical wins**, because
  they affect more of the code.
- Among all, **`reduce`/AggSpec (3.1) and the constrained `.table` escape (3.2)
  are the least well-defined** — even though they need *fewer* foundational
  changes than the multiindex (3.5) or the executor (3.6). Defining them is the
  priority, so **start with them.**
- The two foundational items (3.5, 3.6) get **design notes first** (they touch the
  identity invariants and the executor — not coded blind).
- The tactical items (3.3, 3.4, 3.7) follow; 3.3 is subsumed by 3.1 for declared
  aggregations.

First concrete step: **define `reduce`/AggSpec and the constrained `.table`
escape** (design, since they are the least well-defined), then implement. This is
the session's first core work, so the 553-test suite becomes the live guard, and
the examples earn their regression tests once they stabilize.

## Open questions

- AggSpec shape — does it rhyme with `WindowSpec`/`FiberSpec`/`MatchPredicate`
  (the fourth per-X protocol)? How do declared aggs and the opaque-callable escape
  coexist (3.1 vs 3.3)?
- The `.table` boundary — the exact constrained-escape mechanism (`pipe`/`apply`
  re-validating) vs read-only inspection vs growing operators (3.2).
- Whether 3.1 and 3.2 are related: a declared agg is the safe path *because* it
  does not need the `.table` escape — defining them together may clarify both.
- Sequencing of the foundational design notes (3.5, 3.6) relative to the protocol
  work.
