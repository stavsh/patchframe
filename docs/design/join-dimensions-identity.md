# Join, Dimensions, and Identity

Status: design rationale, agreed 2026-06-10. Not a public API spec. Records the
model before implementation so the join redesign and its prerequisites can be
staged without re-litigating the ontology.

Cross-references:

- `field-identity.md` — `IndexIdentity`, `ForeignIndexField`.
- `lazy-and-bundle.md` §4 — capability declarations; `join` as a blocking node.
- `design-constraints.md` — plans as ordinary datasets; extensible dimensions.
- CLAUDE.md "Extensible Dimensions" — the `WindowSpec` protocol this mirrors.

## Purpose

Settles three questions with one model:

- What is a join, structurally? (**A translator from dimensional comparison to
  index correspondence.**)
- How do equality-key joins, index joins, and interval joins relate? (**They
  are predicates over dimensions or compositions of identity references — and
  the two are different things.**)
- How do joining capabilities extend (geometry, sparse, tolerant time)?
  (**Through per-dimension predicate vocabularies, never through new join
  operators.**)

The forcing workload is the multimodal fusion example: matching transcript
caption cues to time windows is an interval join with tolerance, scoped to a
shared clip namespace — currently hand-rolled inside the fuse function and
explicitly marked as the place this design must land.

## 1. The one-way flow

The spine of the model (user, 2026-06-10):

```
dimensional compare   →   correspondence          →   ForeignIndex   →   Plan
(predicates over          (the join's output: a       (typed row         (an ordinary
 natural-unit values)      relation between two        references)        dataset)
                           indexes; maybe non-unique)
```

Two vocabularies with a one-directional translation between them:

- **Dimensions are the comparison vocabulary.** Natural-unit value spaces
  (`time` seconds, `y`/`x` pixels, categorical label spaces). Values in a
  dimension are compared by *predicates* — equality, interval overlap, spatial
  intersection.
- **Index / ForeignIndex is the result vocabulary.** Row identity namespaces
  (`IndexIdentity`) and typed references into them (`ForeignIndexField`).
  Identity is never predicate-compared; labels are only *matched* within a
  namespace, justified by shared `IndexIdentity`.
- **Join is the translator.** It consumes dimensional comparisons and produces
  an index correspondence — a relation, possibly non-unique — materialized as a
  plan dataset whose mapping columns are `ForeignIndexField`s.

Nothing translates backward: a `ForeignIndexField` never becomes an operand of
a dimensional predicate. (See §8 for the rejected design that would have
allowed this.)

### Design rules

1. **Join translates dimensions to ForeignIndex.** All dimensional reasoning
   happens at plan *creation*; correspondences are the only thing plans say
   about rows.
2. **The plan language is ForeignIndex plus dimensional payload.** Row
   correspondence is expressed exclusively through `ForeignIndexField`s.
   Plans may additionally carry per-row *dimensional payload* for the applying
   operator — `window_expansion_plan`'s slice column is the exemplar — but the
   payload is consumed by binding operators (`slice_data`), never by the
   gather. Do not read this document as "plans contain only ForeignIndex."
3. **Plan application is pure identity mechanics.** `merge` and `explode`
   gather by foreign labels and validate namespaces; they never re-do
   dimensional reasoning. (Already true in the code; now a stated invariant.)
4. **Identity is never predicate-compared.** There is no "equals" predicate
   over index labels across namespaces. If two label sets are comparable as
   *values*, that comparison happens in a categorical dimension (§3), and its
   output is the first correspondence between those rows.
5. **Correspondences are relations, not functions.** Non-unique mappings are
   first-class; `explode` repeating source rows by repeated foreign labels is
   the feature, and join cardinality stays `unknown`.
6. **Two plan kinds, by cardinality** (clarified 2026-06-15). A *single-FK*
   plan is **one-to-many**: one side is the plan's own index, each row maps to
   one foreign (`window_expansion_plan`: window → one clip; `explode` consumes
   this). **Many-to-many** needs a *throwaway index and two FKs* — the
   correspondence pair table (`match`/`join` output; `implode`/`candidates=`
   consume it). A single FK cannot carry many-to-many. The compact *factored*
   alternative is two single-FK plans through a shared block identity, which
   is **not** one plan dataset (different lengths) — see
   `dimension-join-execution.md` §2.

## 2. One join model: per-dimension predicates

A join specification is a conjunction of terms, each one of:

- **Identity scope** — "both sides reference the same row of X": each side
  contributes a `ForeignIndexField` (or its own index where identities align,
  §5) into the *same* `IndexIdentity`. Matching is label agreement. This term
  is **validated**: if the two references target different namespaces, the
  join errors instead of silently producing garbage pairs. (Today's
  `DimensionJoin(on="source_index")` scoping is this term, currently
  implemented as unvalidated same-name equality.)
- **Dimensional predicate** — "the sides' values in dimension D satisfy P":
  each side contributes a field bound to the same dimension (`DimensionField`
  values, or a `DimensionedSliceField` selector for that dimension), and P is
  drawn from the dimension type's predicate vocabulary.

The existing strategies dissolve into this model:

| today | becomes |
|---|---|
| `FieldEqualityJoin(on="city")` | `equals` predicate on a categorical dimension |
| `DimensionJoin(dimensions=("y","x"))` | `overlap` predicates on interval dimensions |
| `DimensionJoin(on=key, ...)` | identity scope **or** categorical `equals` (§3) |
| `IndexJoin` | identity alignment **or** categorical `equals` over labels (§5) |

Joining capabilities extend by adding dimensions and their predicates — never
by adding join operators.

### The predicate protocol

Predicates live **with the dimension type**, because the dimension owns
natural-unit semantics — the third instance of the per-dimension protocol
pattern (`WindowSpec` for expansion, `FiberSpec` for fibering, this for
matching):

- `CategoricalDimension`: `equals`, `isin`.
- Interval dimensions (`TemporalDimension`, `IndexDimension` ranges):
  `overlap(pad=0)`, `within`, `contains`, `nearest(tol)`. `overlap(pad=...)`
  is the caption-slack case generalized — tolerance is predicate vocabulary,
  not user code.
- Geometry extension dimensions: `intersects`, `dwithin(distance)`, etc. —
  shipped by the extension with its `GeometryDimension`.

A predicate must supply two things:

1. **Semantics**: the pairwise truth test (the spec; used by contract tests).
2. **A vectorized pairing strategy**: how to produce matching pairs in bulk —
   `equals` partitions (hash join), interval predicates sweep within
   partitions (sort + `searchsorted` / interval index), spatial predicates
   bring their own index (R-tree). The current `_dimension_join_table` is an
   O(n·m) `iterrows` loop per scope — acceptable for plan-sized inputs,
   disqualifying for the unified join.

Execution order is a fixed heuristic, not a query optimizer: identity scopes
and `equals` predicates partition first; interval/spatial predicates pair
within partitions; anything without a bulk strategy filters candidate pairs
last. (The execution model — the blocks→pairs IR, stage classes, null
semantics, the two-tier commensurability rule, and the dimension/source
resolvability handshake — is settled in `dimension-join-execution.md`,
2026-06-12.)

## 3. The `on=` split

Equality keys conflate two semantically different terms; the unified model
separates them, and the split adds safety:

- The key columns **already reference another dataset's index**
  (`ForeignIndexField`, carrying `index_identity`): this is an **identity
  scope**. No dimension, no predicate, and namespace agreement is checked via
  the carried identities.
- The key columns hold **values in a shared categorical space** (`city`,
  `audio_id` before any dataset relation exists): this is a **categorical
  `equals` predicate**, a genuine dimensional compare whose output mints the
  first correspondence between those rows.

The user-facing wrapper dispatches on field type: `ForeignIndexField` →
identity scope (validated); `DimensionField` over a categorical → `equals`;
plain `ValueField` → promote to an ad-hoc shared `CategoricalDimension`
(deterministic: frozen-dataclass equality means independently minted
`CategoricalDimension(name="city")` compare equal, so both sides agree without
coordination). Same call surface as today's `join(left, right, on="key")`,
strictly more meaning.

## 4. The `link` operator (missing entry point)

A column that *is* a reference but is *typed* as a value cannot participate in
identity scoping. The fusion example mistypes exactly this way: the
transcript's `clip_id` is a reference into the clips namespace, declared as a
`ValueField` because `make_transcript` runs before composition and identity is
minted at creation boundaries — the target identity does not exist yet at
creation time.

The missing piece is an explicit **`link`** operator:

```python
transcript = link(transcript, clips, "clip_id")
```

validates labels ⊆ target index (configurable: error / allow-dangling), and
upgrades the column to a `ForeignIndexField` carrying the target's
`IndexIdentity`. It is the dual of `set_index` (which downgrades a primary
index to an `IndexColumnField` that keeps its old identity), and the formal
entry point into the plan language: the moment "this column references that
dataset" becomes typed rather than conventional. Linking is inherently
post-composition; that is consistent with the identity model, not a wart.

## 5. `IndexJoin`, reclassified

"Join on the index" is not one operation:

- **Same `IndexIdentity`** on both sides (preserved/aligned namespaces):
  identity *alignment* — not a comparison at all; rows correspond by
  definition and the correspondence is checkable.
- **Different namespaces** whose labels happen to encode the same real-world
  key: an honest comparison of label *values* — a categorical `equals` over
  the label space, with the output minting the correspondence.

The fusion example's video+audio composition is the second case (two
independently minted identities whose labels agree as values), and the
`concat_columns` IndexField-collision friction it hit is this distinction
surfacing without vocabulary.

**Landed 2026-06-10:** `concat_columns` now applies this classification —
same-name `IndexField`s sharing one `IndexIdentity` unify silently (alignment,
no collision strategy); distinct namespaces still require an explicit
strategy. First payoff: **plan-column carry by alignment**. `explode` output
inherits the plan's index identity, so plan-only columns (a fresh slice field,
the `source_index` mapping) attach by composition —

```python
windows = explode(clips, plan)
windows = concat_columns(windows, keep(plan, ["plan_id", "source_index", "window"]))
```

— instead of a schema-changing flag on `explode`. A `keep_source_index=` flag
was proposed and rejected (user, 2026-06-10) because it makes the operator's
schema transition depend on a call argument, breaking mechanically testable
contracts. The general form of that problem — operators whose shape
legitimately varies by call form — is the **explicit-overloads** direction
(one `TransitionPlan` per overload), deferred per `lazy-duality-plan.md`.

## 6. The fusion-example target, restated

With the model in place, the marked v1 hacks in
`examples/multimodal_fusion.py` become:

```python
transcript = link(transcript, clips, "clip_id")
matches = join(
    windows_plan, transcript,
    scope=...,                       # identity scope on the clips namespace
    predicates={"time": overlap(pad=CAPTION_SLACK_SECONDS)},
)
```

producing an ordinary correspondence plan (window_index, segment_index as
`ForeignIndexField`s) — inspectable and filterable before anything
materializes. Attaching matched segments *per window* is then a plan-consuming
aggregation, which is the **partition/aggregate fork** (per-clip transcript
grouping hit the same wall). That fork gates the consumer experience of this
design, not the design itself.

## 7. Staging

1. **Design note (this document).** Pin the model.
2. **Predicate protocol + vectorized strategies** on `CategoricalDimension`
   (`equals`) and interval dimensions (`overlap(pad=)`); generalize
   `DimensionJoin` to consume per-dimension predicates; reimplement the `on=`
   scope as the validated identity-scope term.
3. **`link`** operator (small; first consumer is the fusion example).
   **Landed 2026-06-11** as `link(ds, target, field)`; see
   `partition-aggregate.md`.
4. **Wrapper sugar**: `join(left, right, on=...)` dispatching per §3;
   `FieldEqualityJoin`/`IndexJoin` become thin shims, deprecated on the usual
   alias path.
5. **Matched-set aggregation** (segments-per-window): the partition/aggregate
   fork is now resolved — `partition` by the window `ForeignIndexField` over
   the explode-gathered pairs (see `partition-aggregate.md` §6). What remains
   blocked here is only the join itself (items 2–4).

Interval predicates require composed slice columns; users compose via
`compose_slice` first (the transcript span case dogfoods it). Keep that
requirement rather than accepting loose start/stop column pairs in the join.

## 8. Rejected: index-as-dimension

The bridge "an `IndexIdentity` induces a categorical dimension" (proposed
2026-06-10, retracted same day) is **rejected**. It would let identity leak
into the comparison vocabulary — making it legal to predicate-match
identities — collapsing the two-vocabulary structure that gives this model its
invariants (§1 rules 3–4). The needs it tried to serve are met inside the
model: index-equality joins are either identity alignment or categorical
label-value comparison (§5), and foreign-key scoping is the validated identity
scope (§3).

Anti-patterns, for the record:

- A predicate whose operands are index labels across namespaces.
- A `ForeignIndexField` accepted where a `DimensionField` is expected (or vice
  versa) without an explicit `link`/classification step.
- New join *operators* for new matching semantics — extension is per-dimension
  predicates only.
- Plan-applying operators (`merge`, `explode`) reading dimensional payload to
  decide row correspondence.

## 9. Open questions

- **Predicate protocol naming and shape** (`MatchSpec`? methods vs registry) —
  design alongside the `WindowSpec`/`FiberSpec` consolidation so the three
  per-dimension protocols rhyme.
- **Resolved — aggregation of matched sets.** The partition/aggregate fork is
  decided in `partition-aggregate.md`: `partition` (the tall bundle) reads a
  correspondence as group rows, `merge` reads it as pair rows; aggregation is
  `map_fields` over fiber cells.
- **Asymmetric predicate forms** — `within`/`contains` are directional;
  decide whether direction is part of the predicate or the term declaration.
- **Rational-rate dimensions** (NTSC finding) — independent of this model but
  adjacent: predicate math must not assume integer rates.
- **Flag-dependent transition contracts** — operators whose output shape
  varies by call form want explicit overloads, each with its own
  `TransitionPlan` (see §5's rejected `explode` flag). Design with the
  overload direction in `lazy-duality-plan.md`, not piecemeal.
