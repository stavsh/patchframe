# Lazy Evaluation and the Bundle Substrate

Status: internal design rationale. Not a public API spec. May be revised as
concrete workloads land.

## Purpose

Captures the design line that connects three problems patchframe will need to
solve as it scales:

- Couplings as a real computation graph (not a parallel mini-framework).
- Streaming execution of cardinality-changing operators when intermediates do
  not fit in memory.
- Planners (especially `window_expansion_plan`) whose generators have compact
  closed-form descriptions but billion-row materializations.

The argument is that these three are facets of one missing abstraction. The
abstraction has a small substrate (the **Bundle**) and a thin near-term facade
(lazy operator dispatch). This document records the design so the next
conversation that touches any of these can resume from a stable point rather
than re-deriving it.

## Core idea: lazy/greedy duality at the operator level

Every operator that is **per-row independent** and **cardinality-preserving**
can run in two modes against the same logical value:

- **Eager**: input is a materialized `Dataset`, output is a materialized
  `Dataset`, computed now.
- **Lazy**: input is a chunk-emitting form of the same dataset, output is a
  chunk-emitting form, materialized only at a terminal operation
  (`materialize`, `write`, row access).

The duality lives at the **call site**. Whether `where(x, pred)` runs eagerly
or lazily is determined entirely by what kind of value `x` is. There is no
`where_lazy`, no `.lazy()` mode switch, no separate combinator API. Operators
gain a single capability declaration; dispatch follows from it.

This collapses several initially separate proposals (planner combinators,
streaming explode, GPU-batched couplings, deferred field transforms) into one
mechanism.

## The Bundle generalization

A chunk-emitting form of a dataset is structurally a **dataset whose rows are
datasets**. Each row carries a tiny lazy pointer (`DatasetAccessor`) into a
`DatasetSource` that materializes that row's sub-dataset on demand. This is
the existing `DataAccessor` / `DataSource` pattern, lifted one level: instead
of pointing to array data, the accessor points to a fiber Dataset.

QFT mapping (genuine, not decorative):

- Base space B: a Dataset of chunk-metadata rows.
- Fiber F_b: the sub-Dataset attached to row b.
- Total space E: the flattened concatenation of all fibers.
- Sections: picking one row per fiber (sampling, summarizing).
- Connection: couplings that move structure between base and fiber.

**Bundle is not a new dataset type.** It is a role a `Dataset` plays when its
schema contains a `BundleField` whose values are `DatasetAccessor`s. Same way
the current model has `DataField` whose values are `DataAccessor`s — the field
type makes the dataset data-bearing; a bundle field would make it
chunk-bearing. No parallel class hierarchy; the bundle property is a schema
property.

What Bundle exposure (beyond chunked streaming) buys:

- Per-chunk metadata as first-class base-row fields.
- Base → fiber broadcast (pullback): a label on the base propagates to every
  row of the fiber on materialization.
- Fiber → base reduction (pushforward): a coupling computes a per-chunk
  aggregate as a base-row field.
- Cached intermediate chunks (write specific chunks to disk; the bundle stores
  accessors that now point to disk).
- Hierarchical workloads as first-class structure: patient cohorts,
  multi-scale simulations, video/frame hierarchies.

## Staged commitment plan

The Bundle direction should not be built before a concrete workload requires
it. It should also not be foreclosed by near-term decisions.

**Stage 1 — lazy operator dispatch under a flat facade.** Add the lazy arm to
chunk-local operators. The lazy state is structurally a `Dataset` with a
single chunk-accessor column and no other base fields. User-facing name and
ergonomics are flat (`LazyDataset` as a type alias or thin facade; operators
are the same operators they already are). Covers the workloads on the current
horizon: streaming explode, billion-row window plans, GPU-batched couplings.

**Stage 2 — open the Bundle API.** Triggered by a real workload that needs
per-chunk metadata, fiber-to-base couplings, or hierarchical structure.
Promotes `BundleField` to a user-facing field type, adds `over_fibers(op)`
lift, exposes base-vs-fiber operator dispatch. No refactor of Stage 1; the
substrate was already bundle-shaped.

This way the conceptual cost of two-level thinking is deferred until a
workload actually justifies it, but the substrate stays compatible.

## Substrate decisions that touch this

Three places where near-term work has load-bearing implications:

### Transition ontology

`docs/design/aspect-transition-ontology.md` keeps table transitions deliberately
coarse and treats cardinality as an adjacent operator capability. Lazy dispatch
should consume two properties beyond the existing identity aspects:

- **Cardinality contract**: `preserve` / `filter` / `expand` / `unknown`.
  Determines whether an operator can run lazily without coordination.
- **Per-row independence**: whether the operator's output for row i depends
  only on row i's inputs. Defines chunk-local / fiber-local applicability.

Both properties are mechanically checkable and are the actual definition of
"chunk-local." Declared once, lazy dispatch and the future `over_fibers` lift
both consume them. Names should be generic (not `chunk_local`) so the same
declaration serves both stages.

### Identity rules

`IndexIdentity` and `ForeignIndexField` are the right primitives and should
stay shape-generic. A Bundle carries three live identity namespaces (base,
per-fiber, total); the framework must not bake in "one dataset, one identity."
Concretely:

- Do not introduce a `Dataset.identity` shortcut that assumes singularity.
- Keep `ForeignIndexField` directionally neutral — points from one namespace
  to another, no assumption that the target is "outside" the dataset.

### DataAccessor and source genericity

`DataAccessor` is patchframe's existing lazy-pointer primitive. `DatasetAccessor`
is its sibling: same shape, materializes to a `Dataset` instead of an array.
For this to work without a refactor:

- No operator should hardcode that data-column materialization returns ndarray.
- `DataSource` contract should remain output-type-generic.
- `SourceDescriptor`, `SourceManager`, and `assert_source_contract` should
  extend to a `DatasetSource` without shimming.

Audit as touched, not proactively. Fix accidental array assumptions when found.

## Anti-patterns

Two near-term moves would quietly close the door:

- **A `LazyDataset` class introduced as a parallel type.** Forecloses the
  chunk-as-row generalization by giving "chunked" its own class instead of
  letting it be a schema property. If a `LazyDataset` name is exposed, it
  should be a facade or alias over the bundle-shaped substrate, not a separate
  class hierarchy.
- **Plan-specific lazy machinery.** If a streaming workload (e.g., DAS-style
  windowing) arrives before Bundle, the temptation will be to bolt
  planner-specific laziness onto plan datasets. This rebuilds a worse version
  of the bundle substrate inside one operator family. Either use the
  chunk-accessor / `DatasetSource` pattern, or solve the workload user-side
  with explicit loops until the substrate is ready. The middle path is the
  trap.

## Open questions

To be resolved when Stage 1 or Stage 2 starts:

- **Chunk size policy.** Who picks the chunk size for lazy datasets, and how
  does it propagate through composed lazy operators. Probably: the source-most
  lazy node seeds it, downstream inherits, users override per node. Resist
  auto-tuning early — wrong defaults are worse than asking.
- **Cardinality after filters.** Filtered lazy datasets have unknown `len`;
  the framework should expose this as `Optional[int]` and downstream code
  (sampling, progress, sizing) should handle the absent case.
- **Chunk-global operators.** `merge`, `sort`, `dedupe` cannot be chunk-local.
  Near-term answer: force materialization on lazy input with a clear error.
  Long-term: separate shuffle / repartition story, much larger in scope.
- **Identity determinism under filters.** A lazy dataset's row identity must
  be stable across re-materializations. Requires deterministic predicates,
  which should be declared rather than assumed.
- **Bundle dispatch convention.** When (and only when) Stage 2 starts: do
  bundle-aware operators default to base, with `over_fibers(op)` lifting? Or
  do they require explicit scope? Default-to-base reads more naturally for the
  flat-facade migration but needs prototyping against a real workload.

## Why this design line matters

The three problems it solves (couplings as graph, streaming execution, large
planners) currently look unrelated. Without a unifying substrate, each would
grow its own machinery:

- Couplings would gain a parallel computation-graph engine.
- Streaming would get baked into specific operators (`explode`, `materialize`).
- Planners would acquire a `lazy=True` mode that breaks Dataset invariants.

All three would then need to be reconciled later, at higher cost. Recognizing
them as facets of one substrate — chunks as rows, operators dispatching on
materialization mode, accessors as the lazy primitive — is the cheaper path.
The lazy/greedy duality is the operational mechanism; the Bundle is the
structural truth underneath it.
