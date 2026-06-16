# Dimension Join: Execution Model

Status: design rationale, agreed 2026-06-12. Layered over
`join-dimensions-identity.md` (which stays the ontology: the one-way flow,
the `on=` split, predicates-live-with-dimensions, the pair-plan output). This
document settles *how the join runs*: the intermediate representation,
predicate staging, null semantics, commensurability, and the
dimension/source resolvability handshake the commensurability ruling makes
necessary.

Cross-references:

- `join-dimensions-identity.md` §2 — the predicate protocol this executes.
- `lazy-and-bundle.md` §4 — join as a blocking node; the fiber-product future.
- `partition-aggregate.md` — the downstream consumer of the pair plan
  (join → explode → partition: matched sets per row).
- CLAUDE.md "Sparse point data" — sources that accept natural-form selectors.

## 1. Requirements

Two stated requirements (user, 2026-06-12), plus what they imply:

1. **Extensibility** — users add dimension types and how joins apply between
   them, never new join operators (the ontology's rule).
2. **Efficiency** — vectorized in the rows direction; cheap predicates
   (categorical equals) must shrink the search space for expensive ones
   (interval, spatial).
3. **Strategy == semantics, mechanically checked** — every fast path is
   verified against the pairwise truth test (`assert_predicate_contract`,
   the sibling of `assert_source_contract`). The slow path is the spec; an
   extension may ship semantics-only and add the strategy later.
4. **Determinism** — same inputs → same pair plan, same row order (canonical
   left-position-major ordering; hash-grouping internals must not leak).
5. **A payload seam** — predicates may contribute per-pair columns (overlap
   seconds, distance, projected coordinates, depth); retrofitting later
   means re-running matches.
6. **Directionality** — semantic direction (`within`/`contains`, which side
   projects into which space) is part of the term declaration; probe/build
   side-swapping is the strategy's private business.
7. **Scale posture** — join is a blocking node; current workloads are
   plan-sized. Strategies consume column arrays and emit position pairs
   (never `Dataset`s), so the same strategies run unchanged per-fiber when
   the executor lifts the join, and the table engine stays swappable.

## 2. The intermediate representation: blocks, then pairs, narrowing only

The pipeline carries a *correspondence-in-progress* with exactly two
representations:

- **Blocks** — a co-partitioning of both sides (pairs implicit). Cheap and
  composable: the identity scope and every `equals` predicate refine one
  multi-key grouping. The pipeline stays in block-land as long as possible.
- **Pairs** — explicit position arrays. The *first* non-equality predicate
  converts blocks → pairs (its sweep runs vectorized within each block);
  every subsequent predicate is a vectorized filter over the pair table.

Worked example (windows × transcript segments; terms: clip scope +
`overlap(pad=0)` on time):

```
left:  w0 (clip=a, [0,2))   w1 (a, [1,3))   w2 (b, [0,2))
right: s0 (clip=a, [0.4,1.1))   s1 (a, [2.6,2.9))   s2 (b, [0.5,0.9))
       s3 (a, slice with no time selector)

stage 1 (blocks):  {a: ([w0,w1], [s0,s1,s3]),  b: ([w2], [s2])}
                   — 7 implicit candidates, not the 3×4 cross product
stage 2 (pairs):   overlap within each block (sort + searchsorted):
                   (w0,s0) (w1,s0) (w0,s3) (w1,s3) (w2,s2)
stage 3 (filter):  any further predicate masks the pair table
output:            pair plan — left/right ForeignIndexFields (+ payload)
```

**Broadcast sets**: rows unconstrained in a dimension (s3 — a whole-clip
annotation) live in a per-side, per-dimension broadcast set unioned into
every block's sweep — cheap when rare; if everything broadcasts the
predicate simply is not selective.

Once the IR is fixed, "vectorized in the rows direction" is a property of
the representation, not of each predicate's cleverness.

### Block representation — and why it is not a single plan dataset

(Clarified 2026-06-15.) The two representations differ in *kind*, not just
size, because of a cardinality fact about plans:

- **A single-FK plan is one-to-many.** A plan row carries its own index plus
  one `ForeignIndexField` — so one side of the relation *is* the plan's own
  index, each row is a function value (row → one foreign), and many rows may
  share a foreign. `window_expansion_plan` (index `plan_id` = windows, FK
  `source_index` = clip) is this: window → one clip, one-to-many. A single FK
  *cannot* carry many-to-many — the column holds one value per row and the
  other side is pinned to the row identity.
- **Many-to-many needs a throwaway index and two FKs.** The **expanded** form
  is the two-FK pair table (index `pair_id` meaningless; `left_index` +
  `right_index` both FKs) — one row per pair, the same left free to pair with
  many rights across rows. This *is* a single plan dataset (one length =
  #pairs). It is what `match`/`join`/`dimension_join` produce and `implode`/
  `candidates=` consume.

The **factored** form expresses the same many-to-many as **two single-FK
plans sharing a block identity** — `left→block` and `right→block`, each
one-to-many into a minted block namespace; correspond ⟺ same block; expand =
merge on block. Worked (windows × segments scoped by clip):

```
blocks (id B)      left→block            right→block
block_0 | clip a   w0 | block_0          s0 | block_0
block_1 | clip b   w1 | block_0          s1 | block_0
                   w2 | block_1          s2 | block_1
                   w3 | block_0
```

For an equality partitioner the block *is* the distinct key value, so
`left→block` is just the existing key column (`source_index`) and the explicit
block dataset is redundant — the natural form is **implicit**: the operands'
key columns + "correspond ⟺ equal key," the hash-join deferred. Blocks are
always key-value groups (factored ⟺ factorizable ⟺ a union of products,
produced only by equality/scope partitioners; interval/spatial pairers force
expansion).

**The unresolved point (user, 2026-06-15):** the factored form is **not
representable as one plan dataset** — its three constituents (`blocks`,
`left→block`, `right→block`) have *different lengths* (|B|, |L|, |R|), so they
cannot share one index. It is inherently a **multi-dataset constellation**
sharing a block identity: either a wide-record **bundle** of the three cells
(the substrate fits — heterogeneous datasets in one carrier), or kept
**implicit** on the operands (no materialization; block ≅ key). So the
expanded↔factored choice is also a single-plan ↔ bundle/implicit choice, not
just enumerate-or-not. How the factored correspondence is represented as a
first-class object — bundle vs implicit, and whether a minted block dataset
is materialized for addressable blocks — is **open**, to settle when the
factored optimization lands (deferred; v1 is all-expanded). The pair-of-
single-FK-plans framing also closes the loop with the plan-cardinality fact:
many-to-many cannot live in one single-FK plan, but two of them through a
shared block reconstitute it.

## 3. Stage classes: capability declaration, never a cost model

Execution order is a fixed heuristic (the ontology's commitment). Foreign
predicates slot in by declaring their **stage class**:

- **partitioner** — refines blocks (identity scope, `equals`, `isin`).
- **block-pairer** — produces pairs within a block, possibly via its own
  structure (sorted endpoints, interval index, R-tree, point spatial index).
- **pair-filter** — pairwise truth test only.

Ordering is by class (partitioners → one block-pairer → filters), ties by
declaration order. No cost estimation: a predicate declares what it *can*
do, never how fast it is. A semantics-only predicate is automatically a
pair-filter — correct, slow, upgradeable. If several block-pairers appear,
the first declared converts; the rest demote to filters.

## 4. Null semantics: absent constraint broadcasts; absent value never matches

Two different nulls, resolved by conventions already in the package:

- **Absent constraint** — a slice is present but does not mention the joined
  dimension: the slicing convention says full extent, so the row
  **broadcasts** (matches everything within its block). The
  whole-clip-annotation case, and the same defaulting Phase 5 validated.
- **Absent value** — a null scalar in a `DimensionField`, a fully-null slice
  cell, or a null scope reference: missing data, **no match**. Precedents:
  `window_expansion_plan` skips null slice rows; a null foreign label is a
  dangling reference.

Stated per term, not discovered. Configurability waits for a workload.

## 5. Commensurability: two tiers

**Generic dimension equality is rejected** as the term-validity rule: it
would canonize the fusion example's `TemporalDimension(rate=1)` clock hack
and make honest rates unjoinable (video at 30000/1001 vs the transcript's
seconds). The root: `TemporalDimension` conflates the **axis** (the quantity
— time, in seconds, within some frame: what makes values commensurable) with
the **sampling** (`sample_rate`: how one source discretizes the axis — a
storage property, consumed by `to_index`, already per-source in the resolve
path). A transcript cue's true dimensional content is an interval in
seconds — an axis-valued fact with no sampling at all.

### Tier 1 — shared-axis terms (the common case)

Term validity is a **dimension-type-owned judgment**:
`Dimension.comparable_with(other)`.

- Base default: **same concrete type and same name**.
- `TemporalDimension`: the default; **rate is excluded** (sampling, not axis).
- Extension types fold in the physical parameters they actually carry — a
  geometry dimension's CRS agreement is the canonical case; resolution never
  participates.

Alternatives rejected (user fork, 2026-06-12):

- *Name-only equality* — repeats the "unvalidated same-name equality"
  pattern the ontology criticizes, and discards type information
  (`IndexDimension("x")` vs `TemporalDimension("x")` must fail structurally).
- *A separate physical-meaning tag* ("time", "space", "pixels") — duplicates
  the type system: the kind **is** the dimension's class; kind-specific
  physics are that type's own fields. A tag is a second name to coordinate
  and a drift hazard.

This preserves the no-coordination property (independently minted dimensions
describing one axis agree without shared objects, per the `on=` split's
deterministic-equality trick) and has a maturation path: when the
consolidation track introduces a first-class **axis object** (axis/sampling
split — sources bind axis+rate; value columns bind the axis alone), the
`comparable_with` default consults it; the protocol seam does not move.

### Tier 2 — bridged terms (declared transforms)

Sharing an axis and being commensurable are not the same thing. The LiDAR ↔
camera join (user challenge, 2026-06-12) is the canonical counterexample:
LiDAR points in (x, y, z) meters and calibrated pixel coordinates (u, v)
share no axis, yet the join is exactly sensor fusion. A **bridged term**
carries an explicit **transform into a comparison space** — e.g.
`projects_into(camera_model, tol=...)`: map the point side through K·[R|t]
into the image side's pixel space, then apply an ordinary spatial predicate
there (containment / ε-distance, plus frustum validity).

Properties of bridges:

- **Declared, never inferred** — the design's honesty rule. The bridge is
  term data.
- **Often per-row**: a moving camera has per-frame extrinsics, so transform
  parameters come from *columns* (the term shape is therefore multi-column
  per side + parameter columns, asymmetric by declaration — requirement 6
  is load-bearing here, not a footnote).
- **Subsumes the datum problem**: epoch-vs-recording-start offsets and CRS
  reprojection are trivial/standard bridges. A *scope* asserts same-frame; a
  *bridge* converts frames; both are declarations.
- **The precompute gradation bounds v1**: a *constant* transform precomputes
  into shared-axis columns via `map_fields` and joins with a tier-1 term —
  pure composition, no bridge machinery. Bridges become necessary exactly
  when the transform is pair-dependent (per-row poses: project(point,
  pose-of-this-image-row) is a function of the *pair*, so neither side can
  precompute alone). v1 builds tier 1 only; the term protocol's shape must
  anticipate tier 2 so `comparable_with` is a gate, not a dead end.

Worked LiDAR example under the full model: scope on scene (identity),
`overlap` on timestamp (tier 1), `projects_into(calibration_columns, tol)`
(tier 2, block-pairer over a per-block point index), payload = projected
(u, v) + depth — then the standard pipeline: pair plan → `explode` →
`partition` → points-per-image.

### Supporting decisions

- **`sample_rate` becomes optional**: `None` = a continuous, unsampled axis.
  The fusion clock becomes honest (`TemporalDimension(name="time",
  sample_rate=None)`); `to_index` requires a rate and errors on a continuous
  axis. Rides the standing rational-rate open item.
- **Values compare in the comparison space's value units** — seconds,
  pixels, labels. "Natural units" ≠ SI units: pixels are the value space of
  an `IndexDimension`; nothing needs avoiding where value space and storage
  space coincide. What never enters the join is *a source's* backend
  resolution (`Dimensions.resolve`/`to_index`) — the NTSC hazard is
  per-source conversion, and it stays at materialization.
- **One judgment, both boundaries**: the same comparability rule should
  eventually guard slice resolution (today `DimensionedSlice` matches by
  name alone); that requires slices to carry the axis key — consolidation
  mechanics, noted, not a join blocker.

## 6. The resolvability handshake (dimension ↔ source)

Optional rate splits dimensions into slice-resolvable and not — and some
non-resolvable dimensions are still valid source inputs (a point-cloud
source takes a bounding volume; a database source takes seconds and looks
up internally). The `supports_partial_read` precedent applies: explicit
declaration, validated at the boundary, never exception-discovered.

- **Selector forms.** A selector exists in **index form** (backend
  positions; requires sampling) or **natural form** (axis-unit values). A
  dimension *produces* the forms it can (`rate=None` → natural only) — a
  derived, mechanically checkable property, not a user knob.
- **Sources declare accepted forms per dimension.** `ArrayDataSource`
  requires index form (today's implicit contract, made explicit);
  sparse/database sources accept natural form and resolve internally.
- **Validation at source definition.** Producible ∩ accepted ≠ ∅ per
  dimension, checked when dimensions bind to the source, with an actionable
  error ("'time' is a continuous axis; ArrayDataSource needs
  index-resolvable dimensions — give it a sample_rate or use a source that
  accepts natural-unit selectors"). One user-facing rule: *comparability is
  the axis's business; resolvability is the sampling's business; sources
  that do their own lookup say so; you find out at definition time, not at
  first read.*
- **`Dimensions.resolve` becomes form-targeted**: index-accepting sources
  get indices; natural-accepting sources get the selector passed through in
  axis units (what `ResolvedSlice`'s name-keeping design was groping toward
  — folds into the consolidation TODO). The full-read-plus-slice fallback
  exists only on the index path.
- **`assert_source_contract`** exercises a claimed natural-form acceptance
  with a natural selector, as partial-read equivalence is exercised today.

The join never participates in this handshake: it compares, never resolves —
a fully non-resolvable axis is a first-class join citizen and invalid for an
`ArrayDataSource`, orthogonally.

## 7. Smaller pinned points

- **Conjunction only**: terms AND; one pair appears once; disjunction is out
  of scope and said so.
- **Self-joins** are legitimate (windows × windows); whether `(i, i)` pairs
  are included is predicate-dependent; the identity scope degenerates to
  block-per-row there and is usually not what you want.
- **Empty results** are ordinary empty plans.
- **Term resolution**: a dimension name resolves per side to a composed
  `DimensionedSliceField` selector or a `DimensionField` value by dimension
  binding (via `comparable_with`); ambiguity errors into explicitness.
  Interval predicates require composed slice columns (`compose_slice` first
  — kept deliberately; the transcript span dogfoods it). Bridged terms name
  their operand and parameter columns explicitly.

## 8. Staging

1. `comparable_with` + optional `sample_rate` (small; makes the fusion clock
   honest; the rational-rate fix rides along or follows).
2. The IR + stage classes with the two core predicates: categorical `equals`
   (partitioner) and interval `overlap(pad=)` (block-pairer), plus the
   pair-filter fallback path; `assert_predicate_contract`.
3. Generalize `DimensionJoin` to consume terms; the `on=` wrapper per the
   ontology's §3; the fusion example's `combine_sample` interval filter
   replaced by join → explode → partition.
4. The resolvability handshake (substrate; lands independently of 2–3).
5. Later: bridged terms (tier 2 — first consumer is a real multi-frame
   workload, e.g. LiDAR+camera in `patchframe_examples`), payload columns
   beyond v1, geometry/sparse predicates (extensions), per-fiber lift
   (executor), engine swap (benchmark-gated).

## 9. Open questions

- Predicate/term object naming and shape (`MatchSpec`?) — design so the
  per-dimension protocols rhyme (`WindowSpec`, `FiberSpec`, this), and so
  the term shape covers tier 2 (multi-column operands, parameter columns,
  declared direction) without building it.
- Bridge representation: transform objects (picklable, CallSpec-like?) and
  their contract tests.
- Axis object and datum metadata — consolidation track.
- Slice-boundary comparability (slices carrying the axis key) —
  consolidation track.
- Payload column conventions (names, dtypes, opt-in surface).
- Whether `isin`/multi-key scopes need dedicated strategies in v1.

## 10. Forward use cases (north stars; not built)

Explored 2026-06-14. Both sit on already-designed seams (plan-as-
correspondence, payload columns, bridged transforms), which is the
reassurance the model anticipated them. Neither is built; they constrain what
the seams must stay compatible with.

### UC1 — neighborhood / offset matching (dimension algebra)

Patch matching across frames wants candidates at a structured offset set
(±1 on both pixel axes = a 3×3 neighborhood). Two framings, both supported:

- **As correspondence generators** — an offset/`NeighborhoodSpec` (sibling of
  `WindowSpec`) emits a candidate correspondence (left=A-patch, right=B-patch-
  at-offset, offset as payload) that feeds `dimension_join` via `candidates=`.
  The plan-is-the-open-seam realization with a concrete consumer.
- **As field-expression algebra** (the user's framing): `dimension_join(...,
  left_on=ds.field("height") + 1, ...)` then `concat` the ±1 shifts. This is a
  *general symbolic operand layer* over fields (`pl.col`-style), not join-
  specific — `assign`/`where`/`map_fields` would consume the same expressions.
  **Achievable today** via `assign` (pre-compute the shifted column) + `concat`,
  so field expressions are ergonomic sugar over an existing capability, not a
  gap. Deferred as its own deliberate track because it is framework-wide and
  has two tensions: (a) it collides with the *handles-always-select-the-lazy-
  arm* law — a `FieldExpression` must be a distinct non-handle eager operand
  spec, **or** a deliberate lazy computation (sugar over `map_fields`); that
  fork must be decided across every operator, not just join; (b) interval/
  dimension algebra has non-obvious semantics (`field("span")+1`: shift both
  bounds? grow? unit-aware? — consolidation territory). `dimension_join`'s
  `_resolve_operand` already dispatches on operand type, so a `FieldExpression`
  branch is additive — keep it compatible; do not build mid-join.

### UC2 — soft matches for differentiable training

Correspondences as *soft* (weighted) matches, trained end-to-end with a
learnable transform (homography/camera matrix) on a bridged term. The
boundary that makes this implementable: **patchframe does discrete candidate
generation + tensor-payload carrying; torch does the differentiable scoring.**
Gradients flow through a torch scoring function (`transform → projected
coords → soft weights → loss`) that *consumes* the discrete candidate plan as
its sparsity pattern — never *through* patchframe's numpy/pandas matching
(selection is argmax/threshold, non-differentiable wherever it lives). The
transform is a model parameter (used detached for proposal, attached for
scoring); the soft weights are the **payload seam**'s first real consumer.
The honest core requirement is **not** differentiable matching (that belongs
in torch) but **tensor-payload integrity**: a payload column holding torch
tensors must survive every operator without numpy coercion/detachment — which
patchframe's pickle-friendliness and dtype-normalization assumptions do *not*
currently guarantee. That audit + a contractually pass-through typed payload
is the real work item. Resist the tighter framing (autograd *through* the
join) — it forces a differentiable graph through a discrete materialized
substrate, the elegance that fights the framework.
