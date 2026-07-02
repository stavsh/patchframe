# Storage Machinery — `save` / `load` / `append`, field-owned IO, the transfer manifest

Status: **design agreed 2026-07-01 (discussion-only; nothing built yet).** The
design note the roadmap's "IO and storage" section (and `SourceIOAdapter`) is
gated behind. Records the converged model and, deliberately, the **capability
spectrum and honest limits** (§11) — this design does *not* promise every IO
feature on every store, and the boundary is stated rather than discovered. The
in-memory halves already exist (`offload`/`MemoryDatasetStore`/`MemoryDatasetSource`,
`dataset-accessor.md`); this note is the disk/persistent build.

Cross-references:

- `dataset-accessor.md` — the lazy cell primitive + `offload` (BUILT); this note's
  `DatasetStore` section is `offload` with a disk backend, and `load` reconstructs
  the lazy cells it defines.
- `design-constraints.md` §5 (source-we-read vs storage-we-own; `DataSource` output
  generic; `SourceDescriptor` worker-reopen; `assert_source_contract`), §7 (pickle-
  friendly serialized state, no live-runtime leak), §9 (a future `execution_context`).
- `field-authoring.md` — the field-owned capability set this extends (the column
  trio, row-exit `register_field_exit`, composition `register_field_policy`);
  storage is the fourth peer.
- `lazy-and-bundle.md` §4 (cardinality; the plan→execute paradigm), the executor.
- `fields.py`, `couplings.py` (`Materialize`, `CallSpec`, `resolve_fiber_cell`),
  `data/{accessor,dataset_accessor,source,array_source}.py`, `storage/array_store.py`.

## Purpose

Persist a `Dataset` — structure, array payloads, and sub-datasets — to disk and
read it back **lazily** (out-of-core on load), with the IO robustness real runs
need (partial-failure isolation, retry, resume, error visibility) *where the
backing store can support it*. The design reuses the framework wholesale: storage
is a field capability, a transfer is a plan `Dataset`, execution is `consume`, and
the progress record is a queryable manifest.

## 1. The surface: one operator each

- **`save(ds, path)`** → writes a container at `path`; returns its descriptor.
- **`load(path)`** → a **lazy** `Dataset` (accessors, not materialized payloads).
- **`append(ds, path)`** → add rows/items to an existing container (capability-gated, §10–11).

Everything below is *internal* to these three. A user never names a store, a
section, an adapter, or a manifest.

## 2. The container: three sections + a manifest, field-routed

A saved container has three payload sections plus a journal, and **which section a
field writes to is the field's own `to_storage` choice (§3), never a storage-layer
`isinstance`:**

- **MetadataStore — the spine.** The encoded table + schema (field-specs) +
  provenance + descriptors, and the **reference-structs** into the other sections —
  a **declarative sidecar, not pickle** (§12). Scalar / index / composite /
  dimension / geometry fields encode *in place* here. **Couplings are not
  persisted** (§3).
- **ArrayStore — the array section.** Where `DataField` owned payloads are
  transferred (§5). Chunked/region-addressable (Zarr/HDF5-like) so the loaded
  accessor stays out-of-core.
- **DatasetStore — the sub-dataset section.** Where `BundleField` fibers persist —
  a fiber is a `Dataset`, so this is a **recursive `save`** (§5). This is `offload`
  with a disk backend.
- **Manifest — the journal (§8).** The transfer plan plus per-unit status/error,
  persisted write-ahead, so a run is resumable and its failures inspectable.

So "metadata- vs array- vs dataset-stored" is the *outcome* of a field's encode,
not a partition the IO layer applies.

## 3. Storage is field-owned — the fourth capability

The peer of the column trio (`table_columns`/`validate_in_table`/
`rename_table_columns`), row-exit (`exit_value`/`register_field_exit`), and
composition (`register_field_policy`). The storage layer **asks the field**:

```python
def to_storage(self, column, ctx) -> tuple[Encoded, list[Transfer]]: ...  # IO-side-effecting
def from_storage(self, stored, ctx) -> column: ...                        # lazy: accessors, no read
```

Resolved by MRO with a `register_field_storage(type, …)` fallback for fields you
do not own. `ctx` carries the container's sections, the `SourceManager`, the
`portable` gate, and the execution context.

**The storable vocabulary (the contract that makes it work).** A field encodes to
a *small closed set* every MetadataStore backend must handle: **native scalar
dtypes + bytes + a reference-struct `{store_id, key, slice}`**. The format only
ever stores those primitives — never a `DataAccessor` or a shapely geometry. Two
consequences:

- **Parquet becomes viable for every field type** (geometry → a WKB *binary*
  column, an accessor → a *struct* column), so we keep parquet's row-group →
  `read_partial(index)` win *and* hold object-cell fields. The "parquet can't store
  object cells" objection dissolves at the field's `to_storage`.
- **Extension fields persist without core dependencies.** A geometry extension's
  `GeometryField.to_storage` = shapely → WKB; `from_storage` = WKB → shapely; core
  never imports shapely (same discipline as "geometry lives in extensions").

`to_storage` is asymmetric on purpose: it *writes* (emits transfers, has IO side
effects) like `offload`; `from_storage` stays lazy (re-points accessors, no read).

### Couplings are not persisted — but stay serializable for execution

The persisted container holds **settled data only; no pending couplings.** A
coupling carries *code* (a `CallSpec` — an operator/fn ref plus args), the
un-migratable part; keeping couplings out of the durable format is what makes it
pure declarative data and **deletes the coupling-spec-serialization branch
entirely**. Two consequences on *different* axes — do not conflate them:

- **Durable, cross-version data format → coupling-free.** During save, couplings
  **settle per-unit as each fiber/column is written** (run, then discard) — not a
  memory-bound all-at-once `consume`. The loaded dataset has no pending couplings;
  couplings are re-applied **on top** of the loaded `DatasetAccessor`s, never inside
  — already the idiom for per-fiber work (a carrier `MapCoupling` over the fiber
  column). Two coupling-free ways to keep a derived column: *settle it* (store the
  values) or *drop it and re-apply on top* (recompute at load).
- **Transient, same-code execution/distribution → couplings stay pickle-serializable,
  and are load-bearing.** `CallSpec` / `warn_if_unpicklable` / design-constraints §7
  are unchanged — they serve worker-shipping, the parallel/distributed executor
  (§7), *the settling phase of a parallel save* (settling couplings fan out to
  workers), and the resume journal (a mid-settle resume re-runs its coupling).
  Dropping coupling serialization would break parallel execution, so it stays.

The axis is **durable-cross-version-*data*** (declarative, coupling-free) vs
**transient-same-code-*computation*** (pickle, couplings required); orthogonal, both
hold. Honest sacrifice: a save→load round-trip drops the pending computation graph —
you re-declare couplings after load.

## 4. Reference vs own — the `portable` gate

Per field cell, `to_storage` decides:

- **`portable` source → reference.** Serialize the accessor as a reference-struct;
  the bytes live externally by contract (a WAV at a path, a remote raster). No IO.
- **non-portable / eager array / explicitly owned → transfer** to the ArrayStore
  (§5), then reference the new owned slot.

`portable` already exists on every `DataSource`, so it *is* the source-vs-storage
switch of `design-constraints.md` §5 for v1. The full typed-category taxonomy
(read-only / mutable / append / queryable / generated / model-output) is a later
*refinement* of this gate for richer adapter behaviour, not a prerequisite.

## 5. A transfer is a plan `Dataset`; the write-coupling is the dual of `materialize`

`materialize` is the *read* half of an accessor. A transfer needs the **dual** —
value → slot (write) — so a block-copy is `write(target, read(source))`, and
**both operands are ordinary fields:**

```python
schema = Schema(fields=(IndexField("block"),  DataField("source"),  DataField("target")))
#   WriteCoupling.apply_row:  target.write(source.materialize())     # the dual of Materialize
```

- `DataAccessor.write(array)` → an ArrayStore region;
- `DatasetAccessor.write(dataset)` → a DatasetStore slot = **a recursive `save`**.

So the transfer plan is a normal `Dataset` (rows = blocks or fibers), which buys
**inspection/caching** (`where`/`sample`/cache before any byte moves),
**distribution** (rows are picklable accessors + `CallSpec`s → ship a row-block to
a worker), and **parallelism** (independent rows). `WholeCopy` is a 1-row plan; no
`BlockCopy`/`WholeCopy` classes survive — they are rows.

**`BundleField` is the recursive arm.** Its write plan has one row per fiber
(`source` = the fiber, `target` = a `DatasetAccessor` slot); the write-call is
`save(fiber, into=target)`. Because a fiber's `save` produces *its* plan, the
overall save plan is a **tree** (a `BundleField` row's fill is a sub-plan). The
tree is the plan; **traversal is the executor's choice** (§7) — depth-first
sequential, or flattened to all leaf block-copies for full parallelism.

## 6. The block grid is the *store's*, not `AxisWindow`; the source keeps its read contract

The transfer imposes no chunker. The **store** owns its native grid; the **source**
uses the read contract it already has:

- **Store (write side)** yields `blocks()` (its native grid — Zarr chunks, one
  whole block, sparse tiles) and accepts `write(region, payload)`.
- **Source (read side)** answers `read_partial` (which already consumes a
  positional `ResolvedSlice` — a raw-array copy is *positional*, so no natural-unit
  round-trip) + `shape`/`extent_for` (size, no read) + `supports_partial_read`.

So a WAV / mock / memory source is transferable **with no new adapter** — it
already has `read_partial`/`shape`. `AxisWindow` stays where it belongs (an
operator-layer *planning* grid for `window_expansion_plan`); storage never depends
upward on `ops`.

**`SourceIOAdapter` is reserved for special cases, both sides**, registered per
kind, an *override* not a gate: a sparse/point source where a region is a
bounding-volume *query*; an opaque payload with no meaningful partial read; a
store that drives the grid from its own chunks; the **mode mechanics** (§10, LMDB
map-resize on append-open). Most sources and the core `ArrayStore` register none.
`Region` is the read/write lingua franca (positional `DimensionedSlice` for arrays;
extension-defined for sparse); the transfer is generic over it, requiring only
that a paired read/write adapter agree on the type.

## 7. Execution: units run by an executor, never a `for`-loop

The transfer *emits independent, executor-agnostic units*; a pluggable executor
runs them — **this is the `execution_context`** (`design-constraints.md` §9):
sequential default now, thread-pool / async / Dask later, over the *same* plan.

- **Independence is structural** (disjoint regions) — that licenses any concurrency.
- **Gated by existing flags:** concurrent reads by the source's `thread_safe` /
  `fork_safe`; concurrent writes by a store-declared write capability + ordering
  (random-access region-write → unordered/parallel; append-only → sequential). The
  executor never parallelises past what a source/store allows.
- **Units are sync-or-async** (`run` / `arun`) so IO-bound sources can be `gather`ed.
- **Eager-allocate / deferred-fill:** `open_writer`/`allocate` assign the target
  descriptor *now*, so `to_storage` returns the `Reference` immediately (pointing
  at an unfilled slot), and the copy is deferred — the null-output-until-consumed
  shape every coupling already has.

`save` = drive `to_storage` across fields → persist the spine → `consume` the one
combined plan with the chosen concurrency.

## 8. Robustness = the manifest (a persisted plan + status/error)

Extend the transfer plan with `status` (pending/running/done/failed) + `error`
(type/message/traceback/item/region/attempts) columns and **persist it write-ahead**
— now it is a **manifest / journal**, and every robustness feature is a dataset
operation:

- **Isolation** — `consume` runs each row in try/except, records the outcome, never
  fail-fast (a `collect-errors` mode).
- **Retry** — re-run a row (safe *because* writes are idempotent, §10); transient
  vs permanent classified by the adapter.
- **Resume** — `where(manifest, status != done)` → re-`consume` (§9 for the two
  granularities).
- **Complete vs failed/interrupted** — the `status` column + a container-complete
  predicate (all rows `done`).
- **Error propagation** — errors are *columns*; after a run `where(manifest, failed)`
  is a queryable `Dataset` of the exact failing (item, region, traceback), so a
  single bad read/write is debuggable, not a lost stack trace.

The manifest is simultaneously checkpoint, failure log, resume source, and
completeness record — one `Dataset`. It is written **as an append-only completion
journal** (append "R done"), which needs a *lower* store capability than in-place
gap-fill (§10) — so progress tracking stays portable even where the array store's
update mode does not.

## 9. Scale, and the two resume granularities

Both regimes resolve to the store-owned grid (§6):

- **Many small arrays** → *many* rows; coarse grid (one block/array); the manifest
  is itself **regime-2** (streams, our own out-of-core case); a `write_batch`
  capability amortises tiny writes.
- **Very large arrays** → *few* rows, *fine* grid (bounded per-block memory, cheap
  per-block retry).

Resume then has **two levels, and they are different mechanisms:**

- **Plan-scale (rows = arrays/fibers)** — *explicit, external*: filter the manifest,
  re-`consume`. Free and uniform; array/fiber granularity.
- **Block-scale (inside one large array)** — the blocks are the store's grid, *not*
  plan rows, so the manifest cannot see them. Resume here is *implicit and
  store-internal*: the array's row is simply re-run, and the store's `missing_blocks(item)`
  yields only the not-yet-written blocks (**idempotent skip**). For a chunk-native
  store this is free (a written chunk file *is* the record). Write-ahead ordering
  ties the levels: a row flips to `done` only after `finalize`, so a mid-array crash
  leaves it `pending` → re-run → store fills the gaps.

We deliberately keep the plan **small (one row per array)** and delegate sub-array
resume to the store, rather than exploding the manifest to one row per block. The
blocks-as-rows form is *available* for a workload that wants uniform plan-scale
block resume, at that manifest-size cost — the exception, not the default.

## 10. The store contract: `open(mode)` + the write side

A container/store is opened in a **mode**, and the store **declares which modes it
supports** — a capability ladder like `portable`/`supports_partial_read`:

```
{create, read}   baseline  →  save + load        (every store)
   + update                →  resume / gap-fill  (idempotent region overwrite)
   + append                →  append operator / grow
```

Mode *mechanics* live in the store's adapter (§6): a filesystem-Zarr `append` open
is a no-op, an **LMDB-Zarr** `append` open resizes the map, a blob store raises
"mode not supported." The generic layer only ever says "open in `update`."

Write side (within a write-capable open):

```python
open(container, mode) -> Container                         # mode-gated
writer = store.open_writer(item, shape, dimensions)        # eager allocate -> descriptor
writer.write(region, array)                                # IDEMPOTENT overwrite
writer.commit(region | item)                               # durable (write-ahead point)
writer.finalize() -> SourceDescriptor
store.missing_blocks(item) -> Iterator[Region]             # block-scale resume
store.write_batch([(region, array), …])                    # optional (many-small)
# DatasetStore analogue: allocate(item) -> (slot, descriptor); fill = recursive save
```

The **read side is unchanged** — the existing `DataSource`/`DatasetSource` contract
(`read_partial`/`materialize`/`shape`). The only genuinely new primitives are the
**`write` duals** (`DataAccessor.write`, `DatasetAccessor.write`) and this write
contract; idempotence + durable `commit` are load-bearing (retry and crash-resume
both rest on them).

## 11. The capability spectrum — what is *not* universal (stated, not discovered)

This design does not force every feature onto every store. Explicitly:

- **Mid-array resume** — chunk-native stores (Zarr/HDF5): ✓ free. A **blob /
  whole-file store**: ✗ — a failure re-does the whole array. (Array-granularity
  resume still works via the plan manifest.)
- **Resume / append** — require ≥ `update` / `append` mode. A **write-once store**
  gives `save` + `load` and nothing else. Accepted.
- **Idempotent writes** — region-write stores: ✓. **Append-only stores**: ✗ (append
  twice = duplicate) — they need transactional commit or dedup, or forgo retry/resume.
- **Portability** — a **non-portable** payload (a live/computed source) *must* be
  owned (copied); it cannot be persisted by reference.
- **Out-of-core transfer** — requires the source's `supports_partial_read`; a
  whole-read-only source is memory-bound (one whole block).

The rule is "the store declares its modes/capabilities; the operator does what the
declaration allows and says so." Not a general guarantee — a documented boundary.

## 12. Backends and dependencies

The store contract is core; **every heavy backend is an opt-in extra or extension,
with a zero-new-dep default so the base always persists** — the pandas-core /
polars-optional, `DataSource`-core / example-extra ethos.

- **Core default (no new deps): a declarative sidecar, not pickle.** The durable
  format is a small JSON/msgpack **structural sidecar** — field-specs (field-type +
  config, recursive) + provenance + descriptors + a `format_version` — plus the
  **encoded columns** from `to_storage` (native `.npy`/memmap arrays via
  `open_memmap`, giving region write/read + out-of-core, zero new deps) and the
  **JSONL manifest** (append-only journal). Robust and migratable *because* nothing
  is a pickled live object: pickle persists class *structure* (patchframe classes,
  and pandas/numpy internals across versions) and cannot be migrated; a declarative
  codec persists type-name + config and can. Pickle stays only for transient
  execution (§3).
- **`[parquet]` → pyarrow:** a durable/cross-language column store with row-group
  partial reads — the scale/interop upgrade for huge flat tables, not a durability
  crutch.
- **`[zarr]` → zarr + numcodecs (+ `fsspec` remote):** the chunked/compressed
  ArrayStore with **free mid-array resume** (chunk presence, §9) and the remote path.
- **`[storage]`** = both; **extensions** own cloud (`fsspec`/`s3fs`), HDF5, TileDB —
  each just implements the store contract.

The dependency you add *is* the capability you buy (the §11 spectrum): the npy
default → out-of-core reads + sidecar-based mid-array resume; `[zarr]` → free
mid-array resume + compression + remote; `[parquet]` → durable/interop spines.
`save(ds, path)` uses the default; `save(ds, path, store="zarr")` opts in — like
`engine="pandas"|"polars"`.

Open detail: **identity persistence** — whether `FieldIdentity`/`IndexIdentity`
round-trip through the field-specs (so reloaded couplings-on-top still match by
identity) or are re-derived on load.

## 13. Reuse, staging, open questions

**Reuse:** `DataField`/`BundleField` + the field-capability pattern; `DataAccessor`/
`DatasetAccessor` (+ the `write` duals); plan `Dataset`s; couplings + `consume`;
`where`/filter (manifest queries); the existing read contract (`read_partial`/
`shape`/`supports_partial_read`/`thread_safe`/`fork_safe`/`portable`); `offload` +
`MemoryDatasetStore` (the memory backend of the DatasetStore section);
`resolve_fiber_cell` (load re-points into it); the `execution_context`.

**Staging.** *v1:* `save`/`load` with the **dep-light declarative default** (§12:
sidecar spine + `.npy`/memmap ArrayStore + JSONL manifest) + the disk DatasetStore,
`portable`-gated reference/transfer, the manifest, a **sequential executor**,
`create`/`read` modes; `[parquet]`/`[zarr]` as opt-in backends.
*Workload/capability-gated:* `append` + `resume` (`update`/`append` modes), the
parallel/async/Dask executor, `write_batch`, the typed-category taxonomy, sparse/
opaque `SourceIOAdapter`s, blocks-as-rows for uniform block-resume.

**Open questions.** The parquet accessor/geometry column encodings (struct vs
sidecar); the exact `commit` granularity and crash-consistency proof; the
manifest's own regime-2 persistence for the many-small case; `assert_store_contract`
(the store-side peer of `assert_source_contract`, checking idempotence + modes +
`missing_blocks`); whether `save_plan(ds, path)` (the inspectable/distributable plan
without `consume`, the join/merge split applied to IO) is worth exposing.

**Anti-patterns.** Storage depending on `ops`/`AxisWindow` (layering); a `for`-loop
transfer (precludes parallel/async); `isinstance`-ing fields in the IO layer
(delegate to `to_storage`); assuming `update`/`append`/idempotence universally
(declare as capability); `materialize`-then-copy for out-of-core arrays (use the
streaming `write`-dual transfer).
