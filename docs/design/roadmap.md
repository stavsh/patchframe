# Roadmap / Work Items

Status: the index of **known-but-unbuilt** work, so references stop getting lost
(the `SourceIOAdapter` design was a one-line constraint mention nobody could find
again — this doc is the fix). Each item is *captured and framed* with pointers,
**not designed** here: workload-gated items get their design at the forcing
function, per the staging discipline (`lazy-and-bundle.md` §7). When an item grows
its own design note, link it from here and shrink the entry to a pointer.

Tiers: **v1** = build next; design exists or is near. **v2** = future /
workload-gated; needs a measured workload (or a design note) before building.

Cross-references:

- `design-constraints.md` — the invariants any of this must respect (the rails).
- `lazy-and-bundle.md` — the bundle substrate, the executor split (light vs heavy).
- `adtech-findings.md` §3 — example-surfaced gaps that overlap (esp. §3.6).
- `operator-authoring.md`, CLAUDE.md "Source Authoring" — the *built* authoring
  surfaces these extend.

## v1 — IO and storage — designed: `storage-machinery.md`

`patchframe/io/` is empty; `patchframe/storage/` has `array_store.py` only
(`storage/__init__.py` exports nothing yet). The **design note is
`storage-machinery.md`** (the note the `SourceIOAdapter` gate required); it settles
the whole family and, deliberately, the honest capability spectrum. In brief:

- **`save` / `load` / `append`** — one user-facing operator each over a container of
  three field-routed sections (MetadataStore spine / ArrayStore / DatasetStore) plus
  a **manifest** (the transfer plan + status/error, write-ahead).
- **Storage is field-owned** — `to_storage`/`from_storage` (the 4th field capability
  after the column trio / row-exit / composition), encoding to a closed storable
  vocabulary (native + bytes + reference-struct) that makes parquet viable for every
  field and keeps extensions (geometry→WKB) out of core. The metadata/array/dataset
  split is the *outcome* of a field's encode, not an `isinstance`.
- **A transfer is a plan `Dataset`**; the `WriteCoupling` is the dual of `Materialize`
  (`DataAccessor.write`/`DatasetAccessor.write`, the latter a recursive `save` →
  `BundleField`'s plan is a tree). Store owns the block grid (`blocks()`), source
  keeps its existing read contract — **no `AxisWindow`, no upward `ops` dependency**;
  `SourceIOAdapter` is reserved for special cases (sparse/opaque/native-chunk, and
  mode mechanics like LMDB resize).
- **Execution** is an executor/`execution_context` over independent units (sequential
  default; parallel/async/Dask later), gated by `thread_safe`/`fork_safe` + store
  ordering. **Robustness** rides the manifest (isolation = per-row try/except; retry
  = idempotent re-run; resume = filter-not-done; errors = queryable columns).
- **`portable` is the reference-vs-own gate**; **`open(mode)`** (create/read/update/
  append) is a declared capability. **Stated limits (§11):** mid-array resume only
  on chunk-native stores; resume/append only on ≥`update`/`append` stores; append-only
  stores aren't idempotent; non-portable payloads must be owned. Not universal —
  documented. Must still respect pickle-friendliness (`design-constraints.md` §7) and
  the swappable table backend (§8).

## v1→v2 boundary — lazy Dataset access (the `BundleField` cell)

- **`DatasetAccessor` / `DatasetSource` — `dataset-accessor.md`; Phase 1 + `offload` BUILT 2026-06-30.** The
  lazy dataset-valued `BundleField` cell (the `DataAccessor`/`DataSource` sibling,
  one level up). The note settles: the cell-union (no separate lazy field type),
  the passive-pointer create→slice→materialize accessor (operators still
  transform the bundle lazily, materialize-before-call), the cardinality law
  (laziness is inherited or expansion-forced, never introduced by a
  row-preserving reshape),
  `offload = save(store=…)` as the realize primitive, the `IndexDimension`-only v1
  cut, the mandatory `read_partial(index)` reader contract, and the cardinality
  length-map / estimator deciding pad-vs-rechunk-vs-fusion. **v1** = the cell
  primitive + the §10 closed-form streaming pattern (lazy `window_expansion_plan`
  → eager index-chunk → lazy per-chunk), which streams out-of-core window
  expansion **without** a fusing executor; the executor (computed-recipe source,
  rechunk buffer, fusion) is **v2**. Gating workload reframed from the journeys
  grouping (regime-1) to out-of-core window expansion of large patches
  (`lazy-and-bundle.md` §2). `fields.py`'s `BundleField` still marks the eager-only
  state as future work.

## v1 — Extension surfaces (consolidation)

Mapped in **`../extensions.md`** (the surfaces that exist + the gaps). The work
itself: decide the uniform shape (subclass vs registry), make the per-X protocols
rhyme (`WindowSpec`/`FiberSpec`/`MatchPredicate`/reducing ops = semantics +
strategy + contract test), build the missing protocols (`WindowSpec` to replace
the `AxisWindow`-only path, `FiberSpec`, deferred `register_predicate`), and close
the verification gaps (`assert_operator_contract`, field/dimension contract tests).
Geometry/sparse/Dask are the driving consumers (`design-constraints.md` §6, §11).

## v2 — Executor (workload-gated; the Dask extension)

- **The streaming / fusion executor** (`lazy-and-bundle.md` §2, the *heavy* build):
  `merge(join(...))` fuses, intermediates never materialize, large joins stream.
  Workload-gated — its policies (chunk size, fusion boundaries, identity
  determinism under filters, barrier handling) need a real workload to settle;
  building them blind guesses wrong (`lazy-and-bundle.md` §9 anti-pattern). The
  **coupling engine becomes the optimizer** here: it owns fusing operations (the
  reduce `bulk_kernel` slot reserves the surface — N reductions over one partition
  → one `groupby().agg()`), which is deliberately *not* built into operators today.
- **Dask = the extension-owned distributed layer** (`design-constraints.md` §7):
  `patchframe_dask.map_field(...)` returning a Dask collection the user
  `.compute()`s — never hidden behind a core call. Partitions are row blocks /
  source-native chunks, not per-row tasks; outputs retain dataset index labels for
  deterministic join-back. Shares the chunk-as-row pattern with the in-process
  executor but stays an escape hatch, not core.

## v2 — Execution context (internal GPU support)

- **A future `execution_context`** (`design-constraints.md` §9), separate from
  `DatasetContext`: the cursor `DatasetContext` carries *dataset ownership*; the
  `execution_context` carries *executor/device selection* (GPU), not ownership.
  Couplings and operators advertise batch / vectorization / GPU capability
  (`design-constraints.md` §3 — "partition strategy is the axis, not row-vs-table");
  the execution context decides chunk size + device. Distinct from the Dask
  extension: in-process device acceleration vs distributed execution, though both
  consume the same capability declarations.

## Already landed (for reconciliation — not work)

So this index is not misread as a to-do for built things: the join stack
(`comparable_with`, predicates, `dimension_join`, `match`, `implode`), `partition`
+ `reduce` + reducing operators, the lazy/eager duality + bundle scaffolding,
`rows()` streaming, the transition ontology + dispatch, `FieldIdentity`/
`MergedField`, the constrained `.table` escape (`pipe` + the `table_transform`
decorator; `table-escape.md`). See CLAUDE.md "Current Direction" for the state
snapshot.

## Smaller open items (tactical; from the examples)

From `adtech-findings.md` §3 and the join open list, not yet homed in a section
above: `map_fields` return-honesty; `sort` / ordered fibers; output-assignment
sugar; composite-key `partition` → multiindex + nullable index; a **lazy,
column-adding `pipe`/`table_transform` arm** (a deferred `extend` threaded into
the computation graph + a whole-table compute coupling — `table-escape.md` §8,
the eager escape is built); field-expression algebra (UC1); `match`/`join`
reconciliation + strategy-shim deprecation; partition benchmark; `to_index`
boundary rounding; stochastic determinism. These graduate into a section here (or
their own note) when picked up.
