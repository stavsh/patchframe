# Lazy Execution and the Bundle Substrate

Status: internal design rationale. Not a public API spec. May be revised as
concrete workloads land.

Cross-references:

- `aspect-transition.md` — transition vocabulary; cardinality and per-row
  independence as adjacent operator capabilities.
- `field-identity.md` — `FieldIdentity`; `BundleField` carries one like any field.
- `design-constraints.md` §1 (identity shape-generic), §2 (transition is the
  contract), §3 (operator↔coupling duality), §4 (Bundle substrate), §9
  (authoring surface).

## Revision note

This revision supersedes the prior version. Substantive changes:

- Adds the **operand-type dispatch axis** as the spine: eager on `Dataset`,
  deferred on handle/bundle operands. The lazy/greedy duality and the
  operator/coupling duality are two faces of this one axis.
- Splits **"lazy"** into *deferred-application* (light) and *streaming/fusion*
  (heavy), and records the project posture: eager is the bulk, lazy is a
  first-class workflow, not the default.
- Replaces "chunk-global operators force materialization" with the
  **blocking / single-fiber barrier** model: deferrability is universal;
  streamability is what per-row-independence gates.
- Adds the **three-way routing** (`same-level coupling` / `streaming lift` /
  `blocking barrier`) *derived* from two declared capabilities.
- Adds the **fiber-bundle characterization** (verticality, pullback/pushforward,
  factorization granularity) and the honesty caveats around it.
- Records the **staging discipline**: the authoring ergonomics and the lazy
  *scaffolding* ship now on plain datasets; the lazy *executor* stays
  workload-gated.
- Resolves the old "bundle dispatch convention" open question (operand carries
  scope; off-schema access errors, no auto-lift).

## Purpose

Captures the design line connecting four problems patchframe will face as it
scales:

- Couplings as a real computation graph (not a parallel mini-framework).
- Streaming execution of cardinality-changing operators when intermediates do
  not fit in memory.
- Planners (especially `window_expansion_plan`) with compact closed-form
  descriptions but billion-row materializations.
- A predictable authoring surface where it is always clear which operands an
  operator accepts and whether a call runs now or later.

These are facets of one substrate (the **Bundle**) reached through one decision
(**operand-type dispatch**). This document records the design so the next
conversation can resume from a stable point.

## 1. The operand-type dispatch axis

The single decision everything else hangs from: **the kind of value you pass an
operator selects eager vs deferred execution, and it is the same axis that
distinguishes the two authoring surfaces.**

- A **`Dataset`** operand → **eager**. The operator runs now and returns a
  `Dataset`.
- A **handle** operand (`FieldHandle`, a field selection, or a bundle handle) →
  **deferred**. The operator records its work and returns a handle.

This is the lazy/greedy duality of `design-constraints.md` §4: the choice lives
at the call site, dispatched on input type — no `.lazy()` switch, no
`where_lazy`. It is also the seam that makes the authoring surface predictable:

```
Dataset      --op-->  Dataset       (eager arm)
FieldHandle  --op-->  FieldHandle    (lazy / authoring arm)
```

The arms are a closed, typed algebra. **The type you are holding tells you which
surface you are on.** Two bridges cross between them:

- `ds.field(...)` / `ds.fields([...])` — entry bridge (`Dataset` → handle).
- `handle.consume()` (candidate rename: `collect`) — exit bridge (handle →
  `Dataset`); the terminal that materializes pending work.

`first-class ≠ default`. Eager is the **operand-default**: a plain
`where(ds, pred)` runs now and the common case never pays a deferral tax. Lazy
is what you *opt into* by passing a handle. The dispatch axis is what *enforces*
that guarantee rather than leaving it to convention.

### Handles are producers, not executors

Field handles carry field-type-specific affordances — `slice_field.windows(...)`,
`dim.overlaps(other)`, future ndarray-style introspection — but they **produce**
specs / operands / couplings; **operators execute**. Three categories:

- **Produce** (`.windows`, `.overlaps`, `.bind_slice`): operands an operator
  consumes. The core ergonomic win.
- **Introspect** (`.shape`, `.dtype`): read-only; gated on the columnar
  batch-extent API (`design-constraints.md` §6); must not trigger silent
  per-row materialization.
- **Execute**: stays in operators. Never a handle method.

Affordances should live on the **`Field` type** (pure, data-free spec
factories); the handle is a generic binder that surfaces them with a dataset
attached. This avoids a parallel handle hierarchy; a registry is the fallback
for decorating field types you do not own. The one carve-out to "handles do not
execute" is the **nullary terminal** (`consume`/`collect`) — it surfaces no
operand and dispatches to the `consume` operator. Transform-methods on handles
(`.where`, `.join`) are *not* allowed; they would rebuild the competing facade
the dispatch axis exists to prevent.

### Field-intrinsic vs dataset-intrinsic

Handles fit where the operation's semantic subject is a field or field relation
(`bind_*`, join-on-fields, dimension windowing). They do not help where the
subject is the whole dataset (`concat` stacking, a bare row filter). The axis is
*field-intrinsic vs dataset-intrinsic*, which cuts across operator families —
some composition ops are field-intrinsic (`join`), some are not (`concat`). A
multi-field composition operand (`ds.fields([...])`) is just a list of
`FieldHandle`s sharing a dataset; name it a *selection* (not `FieldRef`, which is
the persisted coupling reference, and not `*Array*`, which collides with the
multidimensional data layer).

## 2. Two meanings of "lazy"

"Lazy" splits into two builds with very different costs. Keeping them distinct is
what makes the cost legible.

- **Deferred-application (light).** An operator given a handle returns an
  *unconsumed* artifact — a bundle carrying its inputs and a pending op — and the
  terminal runs it. No streaming, no fusion, no chunking; just "not applied yet."
  Buildable without a workload. Mostly dodges the §4 anti-patterns because the
  carrier is a `Dataset` with a `BundleField` (a schema property), not a parallel
  type or a fusing executor.
- **Streaming / fusion (heavy).** The real prize: `merge(join(...))` fuses, the
  intermediate plan never materializes, billion-row joins stream. This needs an
  actual lazy executor — and its policy questions (chunk size, identity
  determinism under filters, barrier handling) are exactly the ones that *need a
  workload* to settle. **Workload-gated.**

Costs the light version does not escape:

- It **defers** materialization; it does not **avoid** it. A deferred artifact is
  a closure over its inputs (graph leaves), so it **pins its inputs** until
  consumed. A downstream lazy `drop` shapes the *output* schema but does not
  relieve the pin — it is the liveness *signal* a fusing executor would read to
  release intermediates mid-run, which is the heavy build again.
- It **taxes the common case**. At N=1 a deferred op returns a wrapper you then
  consume; more steps for a deferral whose payoff (batch, fusion) is zero at N=1.
  Lazy-by-default would subsidize the batch case at the simple case's expense —
  hence eager stays the operand-default.

**Posture.** Not lazy-first: the bulk of the surface and usage is eager. But the
lazy workflow is a first-class supported path, not an afterthought. Concretely
that means the *scaffolding* (capability declarations, the dispatch law, a
unified terminal) lands now; the *executor* waits for a workload.

## 3. The Bundle

**Bundle is not a new dataset type.** It is a role a `Dataset` plays when its
schema contains a `BundleField`. The unification is **at the cell, not the
container**: a `BundleField` cell is a dataset-valued slot — *eager* (holds a
`Dataset`) or *lazy* (holds a `DatasetAccessor` into a `DatasetSource`) — exactly
mirroring how a `DataField` cell is array-or-`DataAccessor`. The bundle property
is a schema property; no parallel class hierarchy.

Two schemas built from `BundleField` cells, sharing the cell primitive but
**not** the container contract:

- **Tall / collection** (the substrate): many base rows, one `BundleField`
  column; each cell is a fiber. `total = concat(fibers)` typechecks because the
  fibers are a homogeneous family. This is where base/fiber/total identity,
  `over_fibers`, pullback/pushforward live.
- **Wide / record** (authoring): one base row, several `BundleField` columns
  (`{left, right}`), heterogeneous. Uses *none* of the substrate machinery —
  `over_fibers` is vacuous at one base row — it is a struct of datasets.

Do not reuse the substrate *role* for the record shape, or you import
identity/lift machinery that does not fit. Share the cell; separate the
container semantics.

**Operators stay bundle-agnostic.** A handle resolves against the bundle to a
plain `Dataset`; the operator runs as it always does; a thin authoring layer
threads the result into the next bundle. The bundle is a *carrier handles
resolve against*, never a thing operators transform — keep it out of operator
transition contracts.

### QFT mapping (structural, not decorative)

- Base B: a `Dataset` of chunk/working-set rows.
- Fiber F_b: the sub-`Dataset` attached to row b.
- Total E: the flattened concatenation of fibers.
- Section: one row per fiber (sampling, summarizing).
- Connection: couplings that move structure between base and fiber.

### Bundle ↔ flat morphisms

A complete set, every operation paired with an inverse or adjoint:

- **Flat → bundle.** `bundle(a, b, …)` combines distinct datasets into a record;
  `partition(ds, by=…)` splits one dataset into key-fibers; `chunk(ds, size=…)`
  is its row-block sibling (the streaming fibering). `partition`/`chunk` should
  be built on a pluggable **`FiberSpec`** — the sibling of the `WindowSpec`
  protocol — so geometry/sparse/time fiberings drop in without touching the
  operator.
- **Bundle → flat.** `flatten(bundle)` = the total space (`concat_rows` of
  fibers, base broadcast down), inverse of `partition`/`chunk`. `extract(bundle,
  "col")` pulls one member/fiber out, inverse of `bundle(…)`. `section(bundle,
  …)` takes one row per fiber.
- **Base ↔ fiber (the connection).** `pullback` broadcasts a base field into
  every fiber row; `pushforward`/`reduce` aggregates a fiber into a base field
  (adjoint). `over_fibers(op)` is the explicit per-fiber lift that the
  handle-call sugar desugars to.

`partition` ↔ `flatten` must round-trip identity: per §1, `flatten(partition(ds,
by=k))` returns `ds`'s row identity (modulo order). `partition` is
deterministic, the base gets a fresh key-namespace identity, the fibers keep
`ds`'s identity, and `ForeignIndexField` (directionally neutral) links
base → fiber.

## 4. Operator capabilities and the three-way routing

Two declared capabilities, adjacent to the `TransitionPlan` (`aspect-transition.md`):

- **`cardinality`**: `preserve` / `filter` / `expand` / `unknown`.
- **per-row-independence**: does output row *i* depend only on input row *i*.

These are the **two natural invariants of a fiber-bundle morphism** (see §5):
per-row-independence = *is the operator fiberwise*; cardinality = *what it does
to the fiber* (`preserve` = iso, `filter` = sub-fiber, `expand` = blow-up). They
are not ad hoc — they are the minimal classification of the operator as a bundle
morphism, and the routing below is read directly off them.

### Why cardinality is not redundant under bundling

Streamability tracks **per-row-independence, not cardinality** —
`window_expansion_plan` is `expand` yet fully streamable; `join` is un-streamable
because it is not per-row-independent, not because of its cardinality. So for the
*can-I-stream-it* question, cardinality is not the gate.

Cardinality earns its place serving a *different* consumer: **reasoning about a
result before computing it**, which is exactly what lazy introduces.

- **Size.** `len(lazy)` is known iff every step is `preserve`; one `filter` makes
  it `Optional[int]`, an `expand` makes it grow-by-unknown-factor. Buffer sizing,
  progress, balanced per-fiber sampling need this without materializing.
- **Identity.** `preserve` → same labels, `filter` → subset, `expand` → **mint**
  new labels. Identity propagation is reasoned in cardinality terms (§1
  determinism).
- **Scheduling.** It is part of how the executor knows a node is a barrier.

Bundling does not subsume this; it *stratifies* it — a fiber-lifted op is
`preserve` at the base (N fibers → N fibers) but carries its intrinsic cardinality
*inside* the fiber, and the fiber-internal cardinality is what predicts
total-space size and identity. In pure-eager you can nearly ignore cardinality
(run and look); deferring is what makes it operational.

### The three-way routing (derived)

From the two capabilities:

- `preserve` + per-row-independent → **same-level coupling**. Records a coupling
  on the current dataset; no fiber needed. (`bind_*`, `add_column`, `rename`,
  `drop`/`keep`.)
- `(filter|expand)` + per-row-independent → **streaming lift**. Chunk-local; runs
  per-fiber; streamable. (`where`, `explode`, `window_expansion_plan`,
  `concat_rows`.)
- ¬ per-row-independent → **blocking / single-fiber barrier**. (`join`, `merge`,
  `sort`, `dedupe`, `set_index`'s uniqueness.)

**Deferrability is universal; streamability is what is gated.** A blocking op is
still a lazy node: represent it as a coupling over a *single-fiber* bundle (the
whole input as one fiber; for a binary op, a one-row, two-`BundleField` bundle —
the wide record). It is a **materialization barrier** — it realizes its whole
input to run, and it collapses any upstream streaming at that node — but it is
deferrable and composable, not an eager carve-out. The old "force materialize" is
only the *pre-substrate stand-in*: until the bundle exists, a blocking op on lazy
input force-consumes; once it exists, it upgrades to a blocking node.

Three forces push an operator off `same-level`, and `set_index` is the proof you
need *both* declarations: it is `preserve` yet still leaves, because it trips
non-per-row-independence (uniqueness is global) **and** identity minting. Those —
plus cardinality change — are the three independent disqualifiers.

### Input type checks the routing; it does not derive it

What an operator *reads* (its handle operands) is orthogonal to what it *does* to
rows/identity. `explode(plan.field("slice"))`, `where(ds.field("score"), …)`, and
`set_index(ds.field("path"))` all take a single plain field handle yet land in
lift / lift / blocking. So input shape cannot derive cardinality.

But it is a sound **one-directional consistency check**: a bundle/selection
operand ⇒ definitely a collection op (lift). The contract suite can cross-check
declaration against operand shape (a `same-level` op must not accept a bundle
handle; a bundle-accepting op must not claim `preserve`+independent) — stronger
than either alone.

`consume` is the exception whose category is **dynamic**: its
per-row-independence is inherited from whatever couplings it runs, so it is
resolved at call time, not a fixed ClassVar. Its *deferred* form is fiber-scoped
(plain input → eager terminal now; bundlefield input → deferred per-fiber).

### Inventory

`cardinality` is partly declared already (transition Phase 6);
**per-row-independence is net-new**. Target:

| operator | cardinality | per-row-indep | routing |
|---|---|---|---|
| `where` | filter | yes | streaming lift |
| `rename` | preserve | yes | same-level |
| `drop` / `keep` | preserve | yes | same-level |
| `add_column` / `assign` | preserve | yes | same-level (data chunk-aligned) |
| `bind_slice` / `bind_materialize` | preserve | yes | same-level (declaration only) |
| `bind_dimensions` | preserve | yes | same-level |
| `window_expansion_plan` | expand | yes | streaming lift |
| `explode` | expand | yes | streaming lift |
| `concat_rows` | expand | yes | streaming lift (stream A then B) |
| `consume` | preserve | inherited | dynamic (from its couplings) |
| `concat_columns` | preserve | if pre-aligned | lift / blocking (alignment) |
| `set_index` | preserve | no (global uniqueness) | blocking |
| `join` | unknown | no | blocking |
| `merge` | unknown | no | blocking |
| `make_from_dataframe` | construct | — | n/a (creation) |
| `make_plan` | construct | — | n/a (creation) |
| `concat` (dispatcher) | — | — | inherits target |

## 5. The fiber-bundle characterization

The streamable/blocking split is the *precise* statement, not a metaphor, and it
does load-bearing work.

- **Streamable ⟺ fiberwise** (a fiber-preserving morphism over the identity on
  the base; "vertical" in the smooth cognate). The operator acts *within* each
  fiber and commutes with the projection — it respects the local product B × F.
  per-row-independence *is* this.
- **Blocking ⟺ not factorizable**: the operator needs the total space E *qua* E.

Pullback and pushforward are the two directions, and they are not symmetric:

- **Pullback** (base → fiber broadcast) is fiberwise — the *easy* direction, the
  exemplar of the factorizable regime. It is a special case; the full streamable
  class is *all* vertical morphisms (a `where` filtering within a fiber on fiber
  data is vertical but not a pullback).
- **Pushforward** (fiber → base reduction) is the canonical *witness of
  non-verticality*: it integrates over the whole fiber. A blocking op requires a
  pushforward, or a base reorganization driven by total-space content (`sort`,
  `set_index`), or a correlation across two bundles (`join` — a fiber product).

The payoff that makes this more than relabeling: recast per-row-independence as
**factorization granularity** — the finest base-chunking under which the operator
still acts fiberwise. Streamable ops admit the *finest* factorization (per-row
fibers); blocking ops admit only the *coarsest* (one fiber = the whole total
space). This **derives** the single-fiber barrier (it is the only admissible
factorization), and it **bounds chunk size**: an operator sets its own minimum
chunk granularity; the executor may chunk coarser, never finer.

Honesty caveats (this is the leash):

- **Discrete setting.** Fibers are sub-datasets; "integration over the fiber" is
  aggregation; the bundle is a map of sets, not a smooth manifold. Structural
  analogy, not differential geometry: pushforward = `reduce`, pullback =
  `broadcast`.
- **Predict, do not relabel.** Keep a borrowing only when it *constrains* (bounds
  chunk size, forces the barrier), not when it merely describes after the fact.
- **Suggest, never decide.** The analogy may generate and compress; a design
  choice must answer to the data/workload. A clean algebra no workload exercises
  is a beautiful cage — this is the elegance trap and the "shatter in your hands"
  posture guarding against it.

## 6. Coupling / operator / lazy unification

A coupling is structurally a **deferred operator application** (§3). So the lazy
arm and the coupling model are the same thing: a handle operand turns an operator
call into a coupling-*producer* instead of an executor.

§3 caps same-dataset couplings at the qualifying subset (single-dataset,
cardinality-preserving, add/fill) — `concat`, `merge`, `join`, `explode`, `where`
do not qualify there. The **bundle fiber-lift** is what relocates the wall: lift
any op per-fiber and, at the *base* level, it is "read cell(s), write cell, per
row" — cardinality-preserving, field-filling — so it qualifies as a base coupling.
The fiber cell encapsulates the fiber-internal cardinality change. §3 is honored,
not violated; it just applies at the level the work is recorded.

Note the deferral mechanisms are not one thing:

- **Coupling** — the field-fill subset; consumed by `consume`.
- **Plan/apply** — `join` eagerly *constructs* a plan dataset (a
  `ForeignIndexField` correspondence); `merge`/`explode` apply it. `join` is not
  "lazy" — it is eager-construct-a-plan, and the plan is what is inspectable/
  filterable before materialization. Plans live *inside* fibers; couplings are
  the base-level lift around them. The models nest.
- **Blocking node** — the single-fiber barrier above.

A unified terminal should resolve *any* pending deferred form — couplings, plans,
blocking nodes — so "deferred → consume" is one honest rule rather than three
special cases. `handle.consume()` runs the couplings whose end node is that field
(today's `consume(ds, column)` = `couplings_for_column`, including upstream),
giving targeted/partial materialization for free. It returns a `Dataset` — the
deliberate exception to "the lazy arm returns handles," because it is the *exit*
— and should be idempotent on an already-materialized field rather than raise.

## 7. Staged commitment and scope discipline

The substrate is not built before a workload requires it, and not foreclosed by
near-term work. The practical split:

**Ship now (cheap, eager-default, plain datasets):**

- `OperatorSignature` — the input contract mirroring `TransitionPlan`: declares
  which operands each operator accepts and enforces no-mixing, so the surface is
  predictable. Interpret the signature in a shared `normalize_call`; do not
  code-gen per-operator. Give it a `custom` escape hatch like `transitions`.
- Producer-handles with field-type affordances; field selections for composition
  readability.
- The **dispatch law** (operand type → eager/deferred) as an enforced rule, not a
  convention — the seam the executor later slots into.
- The **unified terminal** (`consume`/`collect`) resolving any deferred form.
- **Capability declarations** on every operator (`cardinality` narrowed from
  `unknown`; per-row-independence added) plus contract-test stubs. This is the
  difference between lazy-as-afterthought and lazy-as-first-class.

**Stage 1 — lazy operator dispatch under a flat facade.** Add the lazy arm to
chunk-local (streaming-lift) operators. Lazy state is structurally a `Dataset`
with a single chunk-accessor column. Covers streaming `explode`, billion-row
window plans, GPU-batched couplings.

**Stage 2 — open the Bundle API.** Triggered by a real workload needing per-chunk
metadata, fiber↔base couplings, or hierarchical structure. Promotes `BundleField`
to user-facing, adds `over_fibers`, base-vs-fiber dispatch, the blocking-node
representation. No refactor of Stage 1 — the substrate was already bundle-shaped.

The discipline: **scaffold bundle-shaped now, build the executor when a workload
shapes its policies.** The ergonomics and the lazy *contracts* do not wait on the
executor; the *executor* does not get built blind.

## 8. Substrate decisions that touch this

### Transition ontology

Lazy dispatch consumes two capabilities beyond the identity aspects: cardinality
and per-row-independence (§4). Both mechanically checkable; both feed the contract
suite and the future `over_fibers` lift. Keep their names generic (not
`chunk_local`) so one declaration serves both stages.

### Identity rules

`IndexIdentity` and `ForeignIndexField` stay shape-generic. A Bundle carries
three live identity namespaces (base, per-fiber, total); never bake in "one
dataset, one identity." `ForeignIndexField` stays directionally neutral —
inside-bundle base → fiber references use the same primitive. Identity propagation
must be deterministic: a lazy dataset materialized twice yields the same labels,
which constrains predicates and any future stochastic operators to declare
determinism.

### DataAccessor / DatasetSource genericity

`DatasetAccessor` is `DataAccessor`'s sibling: same shape, materializes to a
`Dataset` instead of an array. For this to land without a refactor: no operator
hardcodes that data-column materialization returns ndarray; the `DataSource`
contract stays output-type-generic; `SourceDescriptor`, `SourceManager`, and
`assert_source_contract` extend to a `DatasetSource` without shimming. Audit as
touched, not proactively.

### Pickle-friendliness

`DataAccessor`, `DatasetAccessor`, `DimensionedSlice`, coupling declarations,
operation specs, and `SourceDescriptor` stay pickle-friendly — the precondition
for Dask, multiprocessing, and persistence. Live runtime state (handles, caches,
locks, GPU handles, `SourceManager` references) stays out of serialized state.

## 9. Anti-patterns

- **A `LazyDataset` parallel class.** Forecloses chunk-as-row by giving "chunked"
  its own hierarchy instead of a schema property. A `LazyDataset` name, if
  exposed, is a facade over the bundle-shaped substrate.
- **Plan-specific lazy machinery.** Bolting laziness onto plan datasets rebuilds
  a worse bundle substrate inside one operator family. Use the chunk-accessor /
  `DatasetSource` pattern, or solve user-side until the substrate is ready.
- **Building the executor for elegance or authoring, before a workload.** The
  executor's policies (chunk size, fusion boundaries, identity determinism) need
  a workload to settle; building them to satisfy a uniform model guesses wrong.
- **Lazy-by-default.** Taxes the common case to subsidize batch/streaming. Eager
  stays the operand-default; lazy is opt-in via the handle/bundle surface.
- **Transform-methods on handles** (`handle.where(...)`). Rebuilds the competing
  facade. Only the nullary terminal is allowed as a handle method.
- **Letting the analogy decide.** Bundle/QFT structure may suggest and describe,
  never justify a choice the workload does not.

## 10. Open questions

- **Chunk size policy.** Lower-bounded by each operator's factorization
  granularity (§5). Open: who picks *within* the allowed range — likely the
  source-most lazy node seeds it, downstream inherits, users override per node.
  Resist auto-tuning early.
- **Cardinality after filters.** Filtered lazy datasets have unknown `len`;
  expose as `Optional[int]`; downstream (sampling, sizing, progress) handles the
  absent case.
- **`consume` vs `collect` naming.** The terminal-as-method probably wants a name
  distinct from the coupling-running operator. Deferred.
- **`FiberSpec` protocol.** Design alongside `WindowSpec` before geometry/sparse
  fiberings land.
- **Resolved — dispatch convention.** A bundle *is* a `Dataset`; accessing a
  field not in its schema **errors**, never auto-lifts. Scope is carried by the
  operand (`op(bundle, …)` → base; `op(bundle.field("fiber_col"), …)` →
  per-fiber), so there is no global base-vs-fiber default to choose.
- **Resolved — chunk-global handling.** Not "force materialization" as a fixed
  rule; **blocking node** (deferrable single-fiber barrier), with force-consume
  as the pre-substrate stand-in.

## Why this design line matters

Couplings-as-graph, streaming execution, large planners, and a predictable
authoring surface look unrelated. Without a unifying substrate each grows its own
machinery — couplings a parallel engine, streaming baked into operators, planners
a `lazy=True` flag, authoring a second facade — and all four must be reconciled
later at higher cost. Recognizing them as facets of one substrate (chunks as
rows, operators dispatching on operand type, accessors as the lazy primitive,
handles as the deferred-arm operand) is the cheaper path. The operand-type
dispatch is the operational mechanism; the Bundle is the structural truth
underneath it.
