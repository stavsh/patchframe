# Design Constraints for Future Work

Status: internal design rationale. Not a public API spec. These are the rails
that current implementations must respect to keep future directions viable.
Some are already established invariants; others are scaffolding decisions for
work that is not yet built. Violating one of these does not necessarily break
patchframe today, but it forecloses or significantly raises the cost of a
known future direction.

Cross-references:

- `aspect-transition.md` — transition vocabulary and operator
  contract testing.
- `lazy-and-bundle.md` — lazy/greedy duality, Bundle substrate, staged
  commitment plan.

## How to use this doc

When implementing or refactoring in patchframe, find the relevant sections
below and check that the change respects the listed constraints. Each
constraint records what future direction it preserves so the cost of
violating it is explicit.

## 1. Identity and references

- **`IndexIdentity` must stay shape-generic.** Do not introduce a
  `Dataset.identity` shortcut, helper, or convention that assumes a dataset
  carries exactly one identity namespace. Bundles will need base, per-fiber,
  and total identity to coexist.
- **`ForeignIndexField` must remain directionally neutral.** It points from
  one namespace to another. Do not bake in an assumption that the target is
  "outside" the dataset. Inside-dataset references (e.g., base → fiber within
  a Bundle) must work with the same primitive.
- **Index identity must be semantic, serializable, and stable across
  save/load.** Index identities may be minted at creation boundaries and
  propagated by operators. They do not need to be content-derived, but they
  must round-trip through persistence so datasets saved in one session
  reconnect correctly in another. Python object-identity-based comparisons
  will break IO.
- **Identity propagation across operators must be deterministic.** A lazy
  dataset materialized twice must produce the same identity labels. This
  constrains operator predicates and any future random/stochastic operators
  to declare and respect determinism.
- **Source identity should be stable across sessions.** `source_id` should be
  derived from a source-owned canonical descriptor, not object-id based, so the
  same logical source loaded twice can compare equal. The exact identity basis
  is source-specific: path, URI, query, version, checksum, credential scope, or
  generated configuration may all matter.

## 2. Operators and capabilities

- **Operators must declare structural effects through `TransitionPlan`.** No
  silent schema, identity, coupling, or source effects. The transition
  declaration is the contract that operator tests, cache invalidation, and
  dispatch consume.
- **Phase out `AspectTransition("derive")`** in favor of explicit, mechanically
  checkable modes. See `aspect-transition.md`.
- **Operators must declare cardinality contract**
  (`preserve` / `filter` / `expand` / `unknown`) and **per-row independence**
  as adjacent capabilities. These define chunk-locality for lazy dispatch and
  fiber-locality for future Bundle lifts. Names should stay generic, not
  `chunk_local`, so the declaration serves both stages.
- **Flag-dependent contracts must be representable.** Some operators
  (`add_column`, `bind_dimensions`, `consume`, `concat_columns`, future
  `assign_at`, future `explode(keep_source_index=True)`) have contracts that
  depend on inputs or flags. Class-level static declarations must remain
  conservative; a resolved-transition hook can refine after parameters are
  normalized. Do not promise precision the declaration cannot deliver.
- **Operator APIs should stay independent of the table execution engine
  where practical.** Pandas is the default in-memory backend. Alternate
  engines (`engine="polars" | "auto"`) may be added per-operator only after
  benchmark validation and must preserve schema policies, nullable semantics,
  collision behavior, coupling refs, and the unique-index invariant.

## 3. Couplings and execution

- **Couplings must collapse into the operator/execution model**, not grow
  into a parallel framework. A coupling is structurally a deferred operator
  application: typed inputs, typed outputs, declared capabilities,
  evaluable eagerly or chunk-wise.
- **The operator↔coupling duality is restricted to the qualifying subset.**
  Only operators that are single-dataset, cardinality-preserving, and whose
  schema effect is exactly "add or fill declared output fields" qualify as
  coupling-able. Do not try to promote `concat`, `merge`, `join`, `explode`,
  or `where` into couplings — they belong to plan/apply or eager execution.
- **A "declared but unmaterialized" schema concept is required.** Deferred
  operators must declare their output fields in the schema before values
  exist. This generalizes the existing `DataAccessor` pattern (values that
  haven't been read yet).
- **A batched execution concept must exist between row-level and
  full-dataframe.** Couplings and operators must be able to advertise
  vectorization / batch / GPU capability; the executor decides chunk size.
  Row-level vs full-table is not the right axis — partition strategy is.

## 4. Lazy evaluation and the Bundle substrate

See `lazy-and-bundle.md` for full rationale. The hard constraints from that
direction:

- **No parallel `LazyDataset` class hierarchy.** Lazy state is structurally a
  `Dataset` with a single chunk-accessor field. The user-facing `LazyDataset`
  name, if exposed, must be a facade or alias over the bundle-shaped
  substrate.
- **No plan-specific lazy mode.** Plan datasets stay as ordinary datasets.
  Streaming over plans uses the chunk-accessor / `DatasetSource` pattern, not
  a `lazy=True` flag on plan datasets.
- **The lazy/greedy choice lives at the call site**, dispatched on input
  type. No separate `where_lazy` / `.lazy()` mode switches.
- **Chunk-local operators get the lazy arm for free**; chunk-global operators
  (`merge`, `sort`, `dedupe`) force materialization on lazy input with a
  clear error. A repartition/shuffle story is deferred until needed.
- **Bundle nesting is capped at one level** initially. Recursive bundles are
  mathematically valid but a foot-gun; revisit if a real workload requires
  nesting.

## 5. Sources, IO, and persistence

- **`DataAccessor` and `DataSource` output type must stay generic.** Do not
  hardcode "data column materialization returns ndarray." A `DatasetAccessor`
  / `DatasetSource` sibling must be addable without refactoring the existing
  source machinery.
- **`SourceDescriptor` must be sufficient for worker-process reopen.** Live
  file handles, caches, locks, GPU handles, and `SourceManager` references
  must not leak into descriptors or serialized dataset state.
- **`assert_source_contract` is the source-author testing primitive.** Source
  capability flags (`runtime`, `reopenable`, `portable`) must remain
  meaningful and mechanically checkable. New source kinds extend this
  contract test, not bypass it.
- **Partial-read support must be explicit.** `supports_partial_read` flag,
  not exception-discovered on hot paths.
- **Separate "source we read from" from "storage we own"** in any future IO
  / `SourceIOAdapter` design. A raster on a remote bucket and a Zarr store
  we wrote outputs to are different categories; conflating them under one
  adapter forces every adapter to handle every case.
- **Sources must declare a typed category** (read-only file, mutable store,
  append-only stream, queryable store, generated/derived, label source, model
  output, etc.) so UIs, operators, and provenance tools can reason about
  source roles without source-specific code.
- **Source capability integration needs a user-authorable path.** Users and
  extensions should be able to add source capabilities such as sync, preview,
  write-back, re-download, or validation without patching framework internals.
  This likely wants a registry or protocol, but the exact mechanism is still a
  design guideline rather than a fixed constraint.
- **Do not foreclose graph provenance.** Source sets may remain operationally
  flat for now, but composition APIs should not make it impossible to preserve
  lineage structure later, such as which sources were joined, merged, or
  derived together for downstream UI and audit use.
- **`SourceManager` should not be required for serialization.** Workers must
  reopen sources from descriptors alone; the manager is a process-local
  optimization, not part of the dataset's portable identity.

## 6. Dimensions and slicing extensions

- **Core stays non-geometric.** Geometry, CRS, shapely, GeoPandas, spatial
  indexes belong to extension packages.
- **A `WindowSpec` / `WindowExpansion` protocol must replace the current
  `AxisWindow`-only path before geometry or sparse planners land.**
  Core `window_expansion_plan` should normalize extents, dispatch to
  per-dimension window specs, and assemble the common plan contract.
  Extensions provide their own window specs without reimplementing the
  planning operator.
- **`SparseDimensions` and `GeometryDimensions` must be addable as
  extensions** without core changes. Sparse slice semantics (predicate /
  bounding-volume) and geometry slice semantics (polygon fragments, CRS, tile
  geometry, overlap scores) must fit through the same protocol as dense
  interval slices.
- **Sparse source capability flags must be addable.** Whether a source
  supports bounding-volume partial reads, has a persistent spatial index,
  performs scan-based partial reads, or can estimate occupancy must be
  declarable without core knowing about geometry.
- **Do not assume one logical windowed dimension maps to two start/stop
  selector columns.** Geometry windows can produce richer shapes.
- **Dimensional slicing object consolidation is pending.**
  `Dimension`, `Dimensions`, `DimensionIndex`, `DimensionedSlice`,
  `ResolvedSlice`, `DimensionedSliceArray` should be treated as
  consolidation-eligible. `ResolvedSlice` is explicitly provisional.

## 7. Worker / Dask compatibility (pickle-friendliness)

- **`DataAccessor`, `DimensionedSlice`, coupling declarations, operation
  specs, and `SourceDescriptor` must remain pickle-friendly.** This is the
  precondition for Dask, multiprocessing, and persistence — violations are
  costly to find later.
- **Live runtime state stays out of serialized state.** File handles,
  caches, locks, GPU handles, `SourceManager` references, threadpools.
- **Dask is an explicit, extension-owned execution layer.** Core operators
  must not hide Dask execution behind normal in-memory calls. The preferred
  shape is `patchframe_dask.map_field(...)` returning a Dask collection that
  the user calls `.compute()` on.
- **Dask partitions are row blocks or source-native chunks, not per-row
  tasks.** Per-row Dask graphs explode scheduler overhead.
- **Dask outputs must retain dataset index labels** so computed results can
  be joined back deterministically.
- **The lazy/greedy substrate and Dask are not the same thing.** Lazy
  dispatch covers in-process chunked execution. Dask covers distributed
  execution. They share the chunk-as-row pattern but Dask remains an
  extension-owned escape hatch, not core.

## 8. Engine independence and table backends

- **Pandas remains the default in-memory backend.** It supports nullable
  columns, object columns, extension arrays, and the current schema
  validation model directly.
- **Alternate engines (`engine="polars"`, `engine="auto"`) start as
  benchmark-only.** Promotion to core operators requires measured wins on
  realistic workloads plus correctness equality against pandas.
- **Polars must treat the pandas index as a reserved identity column** —
  materialize it on conversion, restore it on return, validate uniqueness.
  Patchframe row identity is the DataFrame index; Polars has no equivalent.
- **`DimensionedSliceArray` must not be routed through Polars** unless it
  can be preserved or rebuilt without scalar object materialization.
- **Object-column operators (`DataAccessor` columns, slice arrays) likely
  see no Polars benefit.** Do not promote `engine="polars"` paths for these
  without measurement.

## 9. Ergonomics and authoring surface

- **Three-tier complexity budget is a hard guideline.** Source-and-creation
  layer pays high complexity once; dataset-definition layer pays moderate
  complexity once; conventional usage layer stays close to dataframe /
  dataloader feel. Operators or APIs that push source internals or coupling
  ceremony into conventional usage are misplaced.
- **`ArrayDataSource`-style base classes own boilerplate.** New source
  kinds for common cases (datasets-as-sources, sparse, geometry-backed)
  must provide a similar high-level base so user-facing source authors
  implement only source-specific IO.
- **Future `FieldHandle` / `DatasetRef` layer is the planned ergonomic
  upgrade.** Immutable symbolic field references carrying name, type, and
  owning index identity, resolving inside explicit dataset/operator
  contexts. **No global pointer manager** of live datasets or fields; the
  resolution must remain explicit and context-bound.
- **Plan datasets must remain inspectable, filterable, sampleable, and
  concatenable.** This is the value of plan/apply separation. Lazy
  evaluation must preserve this through the chunk-accessor pattern, not by
  making plans opaque.
- **Common dataset usage must feel close to pandas dataframe / pytorch
  dataloader workflows.** Filtering, joins, concat, merge, sampling, row
  access, training integration are the conventional-usage path. Strictness
  is fine; ceremony is not.
- **Domain-specific complexity starts in `examples/`.** Promote to
  `patchframe/` only when a pattern appears in at least two unrelated use
  cases. When in doubt, keep it in `examples/`.
- **`Parameter` is for instance-level behavioral config, not per-call
  data.** Good: `copy=True`, `validate=False`, a database session. Bad:
  rename mappings, filter predicates, dataset-specific values.

## 10. Async access

- **Async row access is separate from table composition.** Core row access
  stays synchronous. An extension or alternate access API may expose
  async/concurrent row reads for IO-heavy sources.
- **Async targets IO-bound materialization and inspection only.** Not
  concat, join, merge, or metadata-only `consume` paths.

## 11. Extension boundary

- **Core is intentionally non-geometric and modality-agnostic.** Geometry,
  sparse points, Dask, Polars, async IO, GPU execution are all
  extension-owned.
- **Extensions extend through declared protocols**, not by monkey-patching
  or by reimplementing core planning/composition. The `WindowSpec` protocol,
  source capability registry, field policy registry, and (future) operator
  capability declarations are the documented extension points.
- **Extension authors must be able to verify their implementations against
  patchframe's structural invariants** without hand-writing every contract
  test. The aspect-transition ontology and `assert_source_contract` exist
  for this; future operator capability declarations should extend the
  pattern.

## 12. Metadata is advisory

- **`DatasetState.metadata` is advisory only.** Semantic correctness must
  come from schema, table structure, and operator validation. Metadata may
  provide convenience defaults or optimization hints; it must never be
  required for correctness.
- **Structural information that affects operator behavior must live in
  schema fields**, not in metadata keys. The migration from
  `metadata["patchframe.plan"]` semantics to `ForeignIndexField` is the
  canonical example.

## Constraints not yet captured

This list is incomplete; expect additions as future work surfaces them. New
constraints should be added when:

- A design decision is made that closes off a previously-viable direction.
- A pattern from one extension or operator generalizes to a framework-wide
  invariant.
- A near-term implementation discovers that a prior assumption was load-
  bearing in a way that wasn't documented.

When in doubt, write it here rather than rely on a future conversation to
re-derive it.
