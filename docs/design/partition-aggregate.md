# Partition and Aggregation

Status: design rationale, agreed 2026-06-10. Records the resolution of the
**partition/aggregate fork** (the group-by gap surfaced by the multimodal
fusion example) before implementation.

Cross-references:

- `lazy-and-bundle.md` §3 — the Bundle, `partition` as the flat→bundle
  morphism, the identity round-trip law (amended here, §4); §4 — the three-way
  routing that classifies partition's lazy form.
- `join-dimensions-identity.md` §2 — the predicate protocol whose
  semantics-plus-vectorized-strategy shape §5 mirrors; §6/§7.5 — the
  matched-set consumer this unblocks.
- `examples/multimodal_fusion.py` — `attach_transcript_segments`, the first
  consumer.

## Purpose

Two consumers hit the same wall: per-clip transcript aggregation (segment rows
→ a per-clip collection) and matched-set aggregation (a join's correspondence
pairs → sets-per-window). The fork asked: does patchframe gain a *partition*
primitive, an *aggregate* operator, or both? This document records the answer:

- **Partition needs no new representation.** A grouping relation is already
  expressible in the plan language: a `ForeignIndexField` column on the member
  dataset, or a correspondence pair-plan from `join`. Correspondences are
  relations, not functions — the non-unique case is first-class.
- **Aggregate needs no new computation.** Per-group computation is
  `map_fields` over a dataset-valued cell; reduction specs with engine
  kernels are a later, benchmark-gated vocabulary (§5).
- **The missing piece is one operator: `partition`** — the collapse-direction
  *reading* of a relation, dual to `explode`'s expansion-direction reading.
  A correspondence thereby has two consumers: `merge` reads it as pair rows
  (wide), `partition` reads it as group rows (fibers).

## 1. Output: the tall bundle, eagerly

`partition(ds, by=...)` returns an **ordinary `Dataset`** playing the tall
bundle role (`lazy-and-bundle.md` §3): one row per group, the group key as the
base index, and a single `BundleField` column whose cells are the member
sub-datasets (fibers).

- **Fibers are unmodified row-subsets of `ds`**: same schema (including the
  `by` column — required by the round-trip law), same couplings, same sources.
  Lazy `DataField`s inside a fiber keep their accessors and materialize
  through them as usual.
- **Eager-at-the-table ≠ decode.** Laziness in patchframe lives *below* the
  table, in accessors and couplings. The eager partition pays a metadata-level
  gather (same cost class as `explode`), never data materialization. The
  precedent extends: eager wide-record first (`ops/bundle.py`), eager
  tall-collection second, same cell primitive.
- **Round-trip law**: `flatten(partition(ds, by=k))` returns `ds`'s rows and
  row identity (modulo order). Fibers all carry `ds`'s identity, so
  `concat_rows`' `row_stack` coalescing preserves it.

## 2. No function operand

`partition` takes no aggregation function — ever. The vocabulary separates
structure from computation: `partition` is the fibering; aggregation is the
adjoint (`pushforward`/`reduce`), and its v1 *is* `map_fields` over the fiber
column. A fiber cell is a per-row value in an object column; `MapCoupling`
already does read-cells → fn → write-cell.

This keeps the timing of user code under the operand-dispatch law, in one
place: `map_fields(groups, [fiber_col], fn, out=...)` computes now (Dataset
arm); `map_fields(groups.fields([fiber_col]), fn, out=...)` records and defers
(handle arm). Lazy aggregation over an eager grouping — partition itself never
needs a lazy arm for the deferral users actually want. Pandas fuses
`groupby(...).agg(fn)` because pandas has no other deferral mechanism;
patchframe does.

## 3. Lazy partition = blocking node (deferred with the executor)

Partition is not per-row-independent — a group's cell needs every input row
with that key; it is the shuffle. It therefore routes as a **blocking node**
(`lazy-and-bundle.md` §4) alongside `join`, `merge`, `sort`, and `set_index`:
deferrable in principle (single-fiber barrier), eager-only as the
pre-substrate stand-in, exactly as those operators behave today. "Lazy
groupby" is not a question this fork answers; it arrives with the lazy
executor, workload-gated. The lazy *cell* (`DatasetAccessor` instead of
`Dataset`) is the orthogonal future axis `fields.py` already marks.

## 4. `by=` identity dispatch (amends `lazy-and-bundle.md` §3)

`lazy-and-bundle.md` §3 says "the base gets a fresh key-namespace identity" —
written with a plain value key in mind. Both waiting consumers group by a
*reference into an existing namespace*, so `by=` dispatches on field type,
the same split as `join`'s `on=` (`join-dimensions-identity.md` §3):

- **`ForeignIndexField` key → identity scope.** The base index inherits the
  referenced `IndexIdentity`. An optional `domain=` operand (the target
  dataset) makes the base **total** over the target index: groups appear in
  domain order, and a zero-member group gets an *empty fiber* (zero rows,
  member schema preserved — strictly better than a sentinel `()`).
  Foreign labels are validated ⊆ the domain index. Without `domain=`, the
  base covers observed keys in first-appearance order, identity still
  inherited (subset semantics). Inherited identity is what makes the attach
  step pure composition: `concat_columns(clips, keep(groups, [...]))`
  unifies by identity alignment, no collision strategy.
- **Plain value key (`ValueField`/`DimensionField`) → categorical.** Fresh
  mint, observed keys, first-appearance order — as `lazy-and-bundle.md`
  states. `domain=` is rejected here: there is no identity to validate the
  domain against. (When `link` lands, upgrading the key column is the way to
  opt into the identity arm.)

This makes the identity transition depend on the *field type* of `by=` —
schema-determined, not call-argument-determined, so it does not trip the
flag-dependent-contract rule. It is the same move as identity-aware
`concat_columns` (dispatch on schema-carried identity).

**Null keys error in v1.** A null foreign label is a dangling reference; a
null categorical key has no principled group. Revisit if a workload wants a
null group.

## 5. Engine group-by: strategies, never semantics

The user's concern (2026-06-10): the bundle path uses none of pandas'
group-by machinery, which is more efficient in some circumstances. The
resolution splits what the engine offers into two pieces, each landing in an
already-sanctioned place. The general law — the same relation the predicate
protocol establishes between pairwise semantics and `searchsorted`/hash-join
strategies — is that **the engine supplies strategies; patchframe defines
semantics.**

1. **Partition's gather lowers to engine group-by internally.**
   `df.groupby(by, sort=False, dropna=False).indices` is the vectorized
   key→positions map; fibers are positional takes. Implementation detail,
   invisible to the contract. The operator owns the policies the engine would
   otherwise quietly decide: `sort=False` (first-appearance determinism),
   `dropna=False` (null-key policy is ours: error), `observed` for
   categoricals, and totality over `domain=` — which pandas cannot express
   (it cannot emit a group it never saw).
2. **Fused agg kernels live in a future declared-spec vocabulary.**
   `groupby().mean()` is fast because the reduction is a fused kernel that
   never materializes groups — available only for *declared* aggregations
   (no engine vectorizes an opaque Python callable; `.apply(fn)` pays the
   Python path in pandas too). The future `reduce(..., aggs={"mu": mean("x")})`
   mirrors the predicate protocol: each spec supplies **semantics** (its
   fiber-wise definition — what contract tests check) and a **vectorized
   strategy** (the engine kernel). This is also where the
   `engine="pandas"|"polars"` knob from the performance notes plugs in.

One spec object serves three execution levels:

| level | form | kernel | status |
|---|---|---|---|
| eager fast path | `reduce(ds, by=..., aggs=...)` on the flat table | one `groupby().agg()`; fibers never exist | benchmark-gated |
| bulk consume | reduce coupling on an existing partition | per-fiber kernel calls (K, not N) | with reduce specs |
| fusing executor | deferred partition + reduce read from the graph | flat kernel; fibers elided | workload-gated (the heavy build; same sentence as "`merge(join(...))` fuses") |

**Eligibility is declaration, never detection.** A declared spec rides the
fast lane; an opaque callable takes the fiber path — always correct, never
inspected. The coupling engine dispatches on the coupling's declared bulk
strategy (a slot that already exists: couplings declare row and bulk compute
behavior); it does not pattern-match graphs or bytecode.

**Why not build `reduce` now:** both waiting consumers *collect* — the one
aggregation no engine has a kernel for (`.agg(tuple)` is Python-object
assembly everywhere). The observed `groupby` usage pattern that motivates the
fast path is real but unrepresented in current workloads (user, 2026-06-10:
group-by followed by `.apply(list)` is the common case in their experience —
recorded as an assumption to test, not a universal). So: `partition` joins the
benchmark suite; the benchmark, not the elegance of the fused path, triggers
building reduce specs.

What is deliberately *not* exposed: the engine's `GroupBy` object — a lazy,
un-inspectable engine artifact. The tall bundle is its inspectable
counterpart: an ordinary dataset you can filter, sample, and attach columns
to before aggregating.

## 6. The consumers, restated

Per-clip transcript aggregation (replaces the pandas group-by inside
`attach_transcript_segments`):

```python
transcript = link(transcript, clips, "clip_id")        # ValueField → ForeignIndexField
groups = partition(transcript, by="clip_id", domain=clips, into="segments")
clips = concat_columns(clips, keep(groups, ["segments"]))   # identity alignment
```

with the fuse function receiving a fiber dataset (or a `map_fields` step
projecting it to tuples). Matched-set aggregation composes the same way once
the unified join lands: `join` → pair plan → `explode` (gather matched member
rows; output inherits plan identity and carries the window reference) →
`partition` by the window `ForeignIndexField`.

## 7. Staging

1. **Eager `partition`** with the `by=` dispatch of §4; `map_fields` over a
   `BundleField` column confirmed by test; `flatten∘partition` round-trip
   test pinning the identity law. The minimal Stage-2 opening of
   `lazy-and-bundle.md` §7 — the two consumers are the trigger — without
   `over_fibers`, `pullback`, `chunk`, `section`, lazy cells, or the executor.
   **Landed 2026-06-11.**
2. **`link`** (`join-dimensions-identity.md` §4) — partition's identity arm
   wants typed keys; the fusion example is the first consumer of both.
   **Landed 2026-06-11** as `link(ds, target, field)` with the deferred
   bundle arm (per the handles-always-lazy ruling in
   `lazy-duality-plan.md`).
3. **Partition benchmark** (groups-count sweep; fiber-construction cost
   isolated) next to `window_expansion_plan`.
4. **Benchmark-gated:** `reduce` + declared agg specs + engine strategies.
5. **Workload-gated:** blocking-node deferral, lazy cells, fusion (§3, §5).

`partition` should not foreclose the `FiberSpec` protocol
(`lazy-and-bundle.md` §3: partition/chunk over pluggable fiberings). The v1
key-column form is the first, degenerate fibering; the `by=` surface stays
compatible with a spec-shaped generalization.

## 8. Anti-patterns

- An aggregation-function operand on `partition` (re-fuses structure and
  computation; moves user-code timing outside the operand-dispatch law).
- Engine semantics leaking into the contract: silent NaN-key dropping,
  sort-by-default, engine-specific group ordering.
- Detecting vectorizable patterns in opaque callables (bytecode inspection,
  graph peephole rules) — eligibility is declaration only.
- Exposing a `GroupBy`-like lazy engine artifact as the public output.
- Building reduce specs or the fused path before a benchmark/workload demands
  them.

## 9. Open questions

- **Null-key policy** — v1 errors; a workload may want a null group or
  skip-with-warning.
- **Composite keys** (`by=["a", "b"]`) — v1 is single-key; composite keys
  interact with the identity arm (no single referenced namespace) and likely
  arrive with `FiberSpec`.
- **Reduce spec protocol shape** — design alongside `WindowSpec`/`FiberSpec`/
  match predicates so the per-X protocols rhyme (this would be the fourth).
- **Fiber-column conventions** — default name for `into=`; whether a
  member-count convenience column earns its place on the base.
- **Resolved (2026-06-11) — name-valued field arguments are `ParamInput`.**
  The tension this bullet originally recorded (should `by=` be a `FieldInput`
  that tolerates names?) was settled by the operand-dispatch ruling in
  `lazy-duality-plan.md`: handles always select the lazy arm, `FieldInput`
  strictly means handle-accepting, and field-naming arguments on bundle-arm
  ops (`by=`, `set_index.field`, `link.field`) are honestly `ParamInput` —
  replay data resolved against the (possibly deferred) operand at run time.
  No binder change needed.
