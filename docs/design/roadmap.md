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

## v1 — IO and storage

`patchframe/io/` is empty; `patchframe/storage/` has `array_store.py` only
(`storage/__init__.py` exports nothing yet).

- **IO operators: `load` / `save` / `append`.** The dataset-level IO operator
  family — persist a `Dataset` (schema + table + couplings + provenance) and read
  it back, with `append` for accumulating stores. Must respect: pickle-friendly
  serialized state with no live runtime leak (`design-constraints.md` §7), and the
  source-vs-storage split below. `DataField` columns persist as accessors that
  reopen from descriptors alone (`SourceManager` is not part of portable state).
- **`ArrayStore` (exists) + `MetadataStore` (new).** Two persistence backends with
  different shapes: `ArrayStore` holds the array payloads (the `DataField`
  materializations); `MetadataStore` holds the table/schema/couplings/provenance
  (the dataset's structural state). `ArrayStore` needs completion + export from
  `storage/__init__.py`; `MetadataStore` is unbuilt. Keep the table backend
  swappable (`design-constraints.md` §8) — the metadata store is not pandas-only.
- **`SourceIOAdapter` (design lost — recaptured here).** The seam for
  **source-specific IO optimization**, and the place the **"source we read from"
  vs "storage we own"** split lives (`design-constraints.md` §5): a raster on a
  remote bucket and a Zarr store we wrote outputs to are different categories;
  conflating them forces every adapter to handle every case. An adapter handles
  one source/storage kind's efficient load/save/append/partial-read/sync. **Needs
  a design note before building** — specifically the source-vs-storage taxonomy
  and the adapter protocol — because v1 IO above should not harden the wrong
  shape. This entry is the reference that was missing.

## v1→v2 boundary — lazy Dataset access (the `BundleField` cell)

- **`DatasetAccessor` + `DatasetSource`.** The `BundleField` equivalent of
  `DataAccessor`/`DataSource`: a lazy dataset-valued cell that materializes a
  sub-`Dataset` on access, exactly mirroring how a `DataField` cell is
  array-or-`DataAccessor` (`lazy-and-bundle.md` §3). `fields.py`'s `BundleField`
  marks the eager-only state as future work; the genericity constraints are
  already pinned (`design-constraints.md` §5, `lazy-and-bundle.md` §8: the
  `DataSource` contract stays output-type-generic, `SourceDescriptor`/
  `assert_source_contract` extend without a refactor). The *primitive* is v1-able
  scaffolding; the *streaming executor* that exploits it (out-of-core fiber
  streaming — the adtech journeys example is its gating workload, `adtech-findings.md`
  §3.6) is v2. Scaffold the cell now; build the executor at the workload.

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
