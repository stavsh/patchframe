# Lazy Dataset Access — `DatasetAccessor` / `DatasetSource`

Status: design agreed 2026-06-29; **Phase 1 + `offload` BUILT 2026-06-30** (665
passed). Built: the cell primitive (`DatasetAccessor`/`DatasetSource`, the
no-materialize `length`/`shape` primitives, `MemoryDatasetSource`,
`assert_dataset_source_contract`, the base `DataSource.shape` + `ArrayDataSource`
default; the `SourceManager` genericity audit verified — zero refactor) and the
`offload` producer (`offload(ds, store=, chunk_size=)` → a chunk bundle of
`DatasetAccessor`s, the `DatasetStore` protocol [PROVISIONAL] + `MemoryDatasetStore`,
the `resolve_fiber_cell` resolver at 3/5 bundle-read sites). **Unbuilt:** the disk
`MetadataStore` backend (the storage machinery, §5 — now under discussion), the 2
remaining resolver sites (`ApplyOperator`/`MapCoupling`), the cardinality length-map
/ estimator (§9), the §10 window-expansion streaming pattern, the multi-axis
(patch-dim) accessor slice (§6).
Grows the roadmap entry "lazy Dataset access (the `BundleField` cell)" into its
own note (`roadmap.md` → shrink that entry to a pointer). Records the converged
model and the corrections made along the way, so the choices are not
re-litigated. This is the lazy-*cell* and streaming-*execution* mechanics;
`lazy-and-bundle.md` owns the bundle-substrate rationale this extends.

Cross-references:

- `lazy-and-bundle.md` — the bundle substrate; §2 (planners with billion-row
  materializations — the gating workload), §3 (cell-not-container; the
  `DatasetAccessor` cell), §4 (cardinality + per-row-independence routing), §5
  (the fiber-bundle characterization + the elegance leash), §7 (staging
  discipline), §8 (genericity / determinism / pickle), §9 (anti-patterns).
- `design-constraints.md` §4 (no parallel `LazyDataset`; bundle nesting cap), §5
  (source-vs-storage; `DataSource` output type stays generic; `SourceDescriptor`
  worker-reopen; `assert_source_contract`).
- `partition-aggregate.md` §3 — the lazy *cell* named as the orthogonal axis.
- CLAUDE.md "Source Authoring" — the `DataSource`/`ArrayDataSource` contract this
  mirrors one level up.
- `accessor.py`, `source.py`, `array_source.py`, `manager.py`, `descriptor.py` —
  the array-level machinery this lifts. `couplings.py` (`Materialize`/`BindSlice`/
  `ApplyOperator`/`MapCoupling`/`CallSpec`), `fields.py` (`BundleField`),
  `dimensions.py` (`IndexDimension`/`CategoricalDimension`), `dataset.py`
  (`rows()`/`RowSequence`).

## Purpose

Make a `BundleField` cell hold a lazy, out-of-core sub-`Dataset` — the
dataset-valued sibling of how a `DataField` cell is array-or-`DataAccessor`
(`lazy-and-bundle.md` §3). The forcing workload is **out-of-core window
expansion** (`lazy-and-bundle.md` §2: a compact closed-form plan with a
billion-row materialization), not the journeys grouping (see §4). The design
maximises reuse: the `DataSource` contract is already output-type-generic, so the
whole source/descriptor/manager stack extends without a refactor.

## 1. The shape — `DataAccessor`'s sibling (the mirror)

`DatasetAccessor` is a tiny, frozen, pickle-friendly lazy pointer that
materialises to a `Dataset` instead of an array. Same resolution path as
`DataAccessor` (`accessor.py:25`): `source_desc_id` → `SourceManager`
(`manager.py:34`) → `source.materialize(self)`.

```python
# sketch
@dataclass(frozen=True, slots=True)
class DatasetAccessor:
    source_desc_id: int
    dimensioned_slice: DimensionedSlice           # the selector (see §6)
    manager_hint: SourceManager | None = None
    def slice(self, s): ...                        # == DataAccessor.slice (accessor.py:36)
    def materialize(self, manager=None) -> Dataset: ...
```

`DatasetSource` is a `DataSource` subclass (`source.py:24`) whose `materialize`
returns a `Dataset`. **The base contract needs no change** — `materialize` already
returns `Any` (`source.py:49`), and `SourceDescriptor`/`SourceManager`/leases are
output-agnostic. This is the `design-constraints.md` §5 genericity requirement
satisfied by construction. The one honest gap: `assert_source_contract`
(`testing/source_contract.py`) splits — the descriptor/reopen/identity-roundtrip
half generalises; the accessor/slice/dimension half is array-shaped, so a
`assert_dataset_source_contract` sibling reuses the reopen half rather than
"extends without shimming."

## 2. The cell-union — *no* separate lazy field type

A `BundleField` cell is `Dataset`-or-`DatasetAccessor`. This is a **cell-value**
property, not a **field-type** property — there is no `LazyBundleField`, exactly
as there is no `LazyDataField`. The decision (considered and rejected: a distinct
lazy field type):

- **The `DataField` precedent is dispositive.** A `DataField` cell is
  array-or-`DataAccessor` in one type; every consumer keys on
  `isinstance(cell, DataAccessor)` at the cell. `DataField` wants every
  affordance a "lazy bundle" would (windowing, dimension binding, partial reads)
  and gets them via the *source* + couplings + handle, never a subtype.
- **Generic lifts mint `BundleField`** (`partition.py:204` constructs
  `BundleField(name=into)`). A separate type is *erased* the instant any generic
  lift touches the bundle, or forces pervasive `isinstance` in every lift — the
  thing `field-authoring.md`'s "capabilities are field-owned, no `isinstance`"
  law forbids. The cell-union is the substrate's mechanism for letting
  specialisation (the accessor) ride *inside* the generic container.
- **Cost asymmetry.** The union localises the eager/lazy cost to one resolver at
  ~5 sites (§11); a split spreads type-dispatch across those sites *plus* every
  lift. O(1) vs O(operators).

Specialisation lives on the `DatasetSource` (dimensions + read), the
`DatasetAccessor` (the slice), and lifted `BindSlice` — not a field type.

## 3. Transforming a lazy bundle (materialize-before-call)

The accessor *object* is a **passive pointer** — it holds no behaviour and grows
no transform methods, so it is not a competing transform surface (there is no
`accessor.where(...)`). Its own lifecycle is **create** (by a producer — the
`make_*` analogue), **slice** (`BindSlice` attaches a `DimensionedSlice`),
**materialize** (`Materialize` calls `.materialize()`). Behaviour lives in
`Dimensions` (slicing policy + join commensurability), the `DatasetSource` (read +
extent), and the couplings.

But operators **do** transform an accessor-bearing bundle — *lazily, exactly as
they transform an eager `BundleField` today*: the op is recorded as a per-fiber
coupling on the carrier (`ApplyOperator`/`MapCoupling` via `defer_in_level`) and
runs at `consume`/`collect`. The **only** added requirement over an eager
`Dataset` cell is that the cell is **materialised to a `Dataset` before the
operator call** — the §7 resolver, and the exact analogue of `map_fields` over a
`DataField` needing a `Materialize` upstream so the fn receives a concrete array
(`couplings.py:529`). So the lazy arm is unchanged; the accessor only adds a
resolve step in front of the per-fiber call.

On the lifted frame, **cardinality-preserving operators are allowed too** — the
same-level coupling-able ops (`map_fields`/`rename`/`drop`/`keep`), not only the
per-fiber bundle lifts; they reach an accessor cell only through
materialize-before-call. The **only** operators that interact with a
`DatasetAccessor` *directly* — handling it as an accessor rather than via
materialise — are **`materialize`** (resolve → `Dataset`) and the **slice
operator** (`data_slice` / `slice_data`: attach an `IndexDimension` slice, lazy,
§6). That is the exact mirror of `DataAccessor`, whose only direct interactors are
`Materialize` and `BindSlice` — evidence the two-interactor surface is structural,
not just the current v1 scope.

## 4. When laziness exists — the cardinality law

> **Laziness is *inherited* (from a lazy input) or *expansion-forced*
> (cardinality increase). It is never *introduced* by a cardinality-preserving
> reshape.**

This is the `cardinality` capability (`lazy-and-bundle.md` §4) doing the
"reason-before-computing" work it was declared for: preserve ⟹ output total ≤
input total ⟹ nothing new to stream. Consequences:

- **`partition`/`chunk`/`flatten` never need a lazy capability of their own.**
  They preserve *total* rows (`flatten(partition(ds)) == ds`), so spilling their
  resident result is pointless — you already paid the RAM. (`partition` is
  `cardinality=UNKNOWN` only because the *group count* is data-dependent;
  `partition.py:109`. The *total* is preserved, which is the axis memory cares
  about.) Over a resident input they stay eager; over a lazy input they
  *propagate* the input's laziness.
- **Two regimes, and reshapes never cross them.**
  - *Regime 1 — lazy payloads, resident metadata*: `DataAccessor` cells in an
    eager `Dataset`. The common case ("rows point to big files"); the fiber
    tables are small, the heavy arrays already lazy at the `DataField` level. **No
    `DatasetAccessor`.**
  - *Regime 2 — lazy metadata too*: `DatasetAccessor` cells. Needed *only* when
    the table itself does not fit.
- **The producers of regime-2 cells:** (a) a store-backed base / `load` (the
  *origin*); (b) propagation — a row-preserving op over an already-lazy input
  (inherited); (c) **expansion** — `window_expansion_plan`/`explode`, the only
  transform that *introduces* regime-2 (output ≫ input).

**Gating-workload reframe.** The journeys example is a row-preserving grouping of
resident-able metadata → regime 1; it becomes regime 2 only with a store-backed
base, where the laziness is *inherited* and `partition` adds nothing. The honest
gating workload for `DatasetAccessor` is **window-expansion of large patches**:
expansion forces regime 2 from a trivially-resident input. This is the case
`lazy-and-bundle.md` §2 already names.

## 5. `offload` = `save(store=…)` (the realize primitive)

`offload` is not a `partition` capability and not a way to make a resident
grouping lazy — that is the pointless case the law kills. It is `save(store=…)`,
backend-parameterised, applied to an **expansion-forced lazy** result: it
*realises* the billion expansion rows into a concrete store. The store flavours
(see also §9):

- **`source=memory`** — formalises the slicing paradigm. A **pandas-backed**
  memory store is sufficient: object-dtype columns share their contents by
  reference under `.loc`/explode (the array of references is copied, not the
  referenced objects), so a `DataField` cell — lazy accessor *or* inlined array —
  is shared across exploded window-rows; only the scalar/typed columns are
  byte-copied, and a chunk's gather allocates only an O(chunk × columns) pointer
  block. So the heavy payloads are referenced, not duplicated, with no columnar
  buffer. (Columnar/arrow earns its place only for typed-numeric-heavy tables,
  cross-process zero-copy, and disk — optional backends behind the same seam.)
- **`source=disk`** — the memory-efficiency and parallelism win: the table lives
  off-heap, read back by index range.

`offload` only avoids the 2× if it *replaces* the live representation and the
original is released — a store holding copies *beside* the original buys nothing.

Computed vs stored recipe (the `DatasetSource.materialize` body): **stored** =
read the block for this slice; **computed** = replay a `CallSpec` (`couplings.py:380`)
against a base, e.g. `where(base, key==·)`. The stored flavour is the degenerate
case where the "computation" is a block read. For a *closed-form* expansion the
computed recipe is cheaper (recompute window specs vectorised; no write step);
stored/`offload` wins when recompute is expensive or non-addressable (a filter)
or for multi-epoch array caching.

## 6. The dimensional unification, and the v1 `IndexDimension`-only cut

A dataset-of-patches is a **generalised array**: axis 0 = the table rows, axes
1+ = the patch dims. `Dimensions` is already an *ordered axis layout*
(`dimensions.py`), and the two row-chunking modes already have their dimension
types — neither is new:

- `chunk` (positional row blocks) ↔ **`IndexDimension`** (`dimensions.py:79`).
- `partition` (keyed groups) ↔ **`CategoricalDimension`** (`dimensions.py:147`).

So `FiberSpec` = `WindowSpec` over the row axis, and named dimensions on an
ordered layout are exactly what keep row-axis and patch-axis distinct *while*
sharing `Dimensions.resolve` (`dimensions.py:192`) — the name/axis-role
disambiguates. (This supersedes an earlier "never conflate the two axes"
warning: the conflation is unsafe only when *implicit*.)

**The v1 cut (decided):** force the `DatasetAccessor` to carry **only an
`IndexDimension`** (a positional row-range). The window expansion *flattens* the
multi-dim window grid into a 1-D indexed sequence, so positional chunking of that
sequence is all that is needed. The full multi-axis accessor slice is
**redundant**, not merely deferred — dense windowing is the inner expansion;
uniform batch-cropping is `BindSlice` on the patch `DataField` (regime 1). So the
"Dimensions slice datasets" generalisation buys no v1 workload and is dropped
(the elegance leash, `lazy-and-bundle.md` §5). What this gives up: lazy keyed/
semantic-group chunking (regime-2 keyed → v2), uniform chunking *after* a
non-closed-form op (→ §10 boundary), and a scale ceiling on the eager outer plan
(§10).

## 7. `materialize` semantics

- **Single-level.** `DatasetAccessor.materialize()` realises the fiber's
  *structure* (table + schema + couplings); inner `DataField` cells stay lazy
  `DataAccessor`s — the nested-laziness win. Recursion to inner arrays is opt-in
  via the inner accessors' own `Materialize`.
- **Windows attach, do not materialise.** A patch-axis slice pushes down onto the
  inner accessors via `BindSlice` (attach), not a read.
- **One polymorphic `Materialize`, not a split.** Both accessor types share a
  `.materialize()` method, so `Materialize` (`couplings.py:329`) generalises to
  "any accessor → `.materialize()`." The result-type difference (array vs
  `Dataset`) is absorbed by the cell-union and handled at **`exit_value`**
  (`fields.py:396`: a fiber → records), not in the coupling. Type-agnostic in,
  type-specific out.
- **Two consumption modes, the same dual as arrays:** *resolve-on-read*
  (implicit, the §11 sites) + the `Materialize` coupling (explicit bulk realise).

## 8. The reader contract — `read_partial(index)` by default

`DatasetSourceReader.read_partial(index_slice)` is the **mandatory baseline** of
the reader contract — the dataset-level mirror of `ArrayDataSource`'s partial
read (`array_source.py:125`), with two refinements:

- **The index dimension is the privileged default axis** — the one *universal*
  axis (every dataset has a row index; categorical keys and patch dims are
  schema/source-specific).
- **Availability mandatory, efficiency declared.** Fallback = full-load + `iloc`
  (mirrors the array full-read-plus-slice fallback, `array_source.py:132`); a
  `supports_partial_read` analogue declares whether it is cheap (closed-form /
  row-group) rather than discovering it by scan.

Minimal reader: `open` + `length` + `read_partial(index_slice)` — the whole
streaming substrate. (`length` must be answerable without materialising — true
for a closed-form expansion; see §9.)

## 9. Output chunk size — the estimator (and pad / rechunk / fusion)

The raggedness problem: chunking an expansion's *output* by index is uniform `B`
by construction, but a **per-chunk cardinality-changing op** (a `filter` drops
rows, a nested `expand` grows them) makes the *output* size unknown
post-execution. The design (build order = the priority):

1. **Estimator (foundational).** Extend the existing `cardinality` ClassVar with
   an optional **length-map** (`window_expansion_plan` knows windows-per-parent
   from extents; `map`/`rename` = identity; `filter` = "≤ input, unknown";
   data-dependent `explode` = unknown). Propagating it along the deferred chain
   classifies output as **exact / bounded / unknown** without running — building
   out `lazy-and-bundle.md` §4's `len(lazy): Optional[int]`. This *decides the
   strategy*; it does not itself give exact uniform output.
2. **Execution strategy, chosen by the estimator's verdict:**

   | verdict | strategy | machinery |
   |---|---|---|
   | exact (preserve / known-expand) | uniform falls out | none |
   | exact, closed-form | **fusion** — size the input chunk to the known factor | the closed-form optimisation |
   | bounded (filter ≤ B) | **pad to the bound** | simplest; slight waste |
   | exact-no-waste, or unknown | **rechunk buffer** | accumulate → emit uniform → carry remainder |

3. **Fusion is the exact-known closed-form *branch*, not the general mechanism.**
   A chunk-fuser that rewrites `window_expansion_plan` is the
   "plan-specific lazy machinery" anti-pattern (`lazy-and-bundle.md` §9) if made
   central; apply it only when the estimator green-lights it. The **rechunk
   buffer** is the general closer (a filter's exact yield is unknowable by
   sizing); a small stateful streaming node.

So the fork "fusion vs estimate length" dissolves: **build the estimator**, and
it routes each chain to the cheapest valid strategy.

**The streaming sandwich:** mandatory `read_partial(index)` (input, §8) →
estimator (planner) → pad / rechunk (output), with `offload = save(store=…)`
(§5) as the sliceable substrate.

## 10. The v1 streaming pattern (worked: out-of-core window expansion)

```
base                                  (resident, small)
  → window_expansion_plan  (LAZY)     closed-form; never materialised
  → window over IndexDimension (EAGER) the chunk plan: small, uniform B
  → per chunk: explode that index-range (LAZY) → process → discard
```

Works because a closed-form expansion is **positionally addressable** (row `k`
computable without `0..k-1`) and **length-known without materialising**. Uniform
batches fall out (chunking the expansion output directly). No offload, no fusing
executor — the outer chunk plan is eager and small, the laziness is entirely
per-chunk.

**The boundary (the one rule to hold):** a non-closed-form op (`filter`/`sort`/
`join`/keyed-group) *between* the expansion and the index-chunk breaks
addressability — the filtered length/positions are not closed-form. Keep filters
*inside* the per-chunk execution. Anything that needs filter-then-uniform-batch
is the `offload`-realise barrier (assign a global index, then positional-chunk),
by construction. Mechanical gap: `window_expansion_plan` today windows slice
columns, not a scalar row-extent; chunking the `IndexDimension` needs it to
accept `extent = len(expansion)`.

## 11. Reuse inventory + genericity audit

**Reused unchanged:** `SourceDescriptor`/`SourceManager`/desc-id resolution; the
`DataSource` contract (output-generic); pickle-friendly serialized state (§8);
`partition`/`chunk` (now `FiberSpec` instances); `ApplyOperator`/`MapCoupling`
(resolve accessors on read); `rows()`/`RowSequence.__getitems__` (`dataset.py:259`
— out-of-core *for free*: a batch's transient consumes only its fibers);
`BundleField` + `exit_value`; the `cardinality` ClassVar; `DimensionedSlice`/
`Dimensions`/`IndexDimension`/`DimensionedSliceArray`/`BindSlice`; `CallSpec`
(the computed recipe); `window_expansion_plan`/`explode_windows`.

**Generalised, not split:** `Materialize` (any accessor → `.materialize()`).

**New:** `DatasetAccessor` + `DatasetSource` ABC (+ pandas-memory and a
store-backed source); the resolver helper; `read_partial(index)` on readers; the
cardinality length-map + the estimator; pad / rechunk buffer; a
`assert_dataset_source_contract` sibling.

**Cell-read resolution sites** (the "audit as touched", `lazy-and-bundle.md` §8 —
centralise as one `fiber(cell, manager) -> Dataset` helper):
`BundleField.exit_value` (`fields.py:396`); `ApplyOperator.compute`/`apply_row`
(`couplings.py:498`/`505`); `MapCoupling.compute`/`apply_row` (`couplings.py:555`/
`562`); `bundle.py` `_flatten_cells`/`_extract_cell` (`bundle.py:340`/`321`).
The resolver calls `cell.materialize()` with no manager arg, relying on
`manager_hint`/default exactly as `Materialize.compute` does today
(`couplings.py:354`).

## 12. Staging

**v1 (build now):** the cell primitive (`DatasetAccessor` + `DatasetSource` ABC,
`IndexDimension`-only) + the resolver + `read_partial(index)` baseline +
`offload = save(store=…)` with a pandas-memory backend + the §10 closed-form
streaming pattern + the cardinality length-map / estimator + pad-to-bound. This
streams out-of-core window expansion and measures the win — without a fusing
executor.

**v2 (workload-gated):** the computed-recipe source over a store-backed base
(lazy keyed `partition`, predicate pushdown — the blocking node); the rechunk
buffer; the fusion optimisation; the multi-axis accessor slice (Dimensions-slice-
datasets, with per-`DataField` pushdown by axis membership); arrow/columnar +
disk backends; the GPU `execution_context` consuming the sliced lazy batches.

## 13. Open questions and anti-patterns

Open:

- **Scale ceiling on the eager outer plan** — #chunks rows must be resident
  (1M chunks at billion-windows / batch 1000 is fine; extreme scale → a
  hierarchical / lazy outer).
- **Determinism** (`lazy-and-bundle.md` §8) — same slice → same labels/order; an
  `IndexDimension` (positional) chunk needs a stable row order; a
  `CategoricalDimension` (keyed) chunk is order-free.
- **Identity vs slicing** — `IndexField` owns row identity (schema); a row
  `Dimension` is a *source-access* protocol, never on the `Schema`/`Dataset`
  (the discipline that keeps `Dimensions` off `DataField`).
- **`assert_dataset_source_contract`** scope (the reusable reopen half vs the
  array-shaped slice half).
- **`FiberSpec` = `WindowSpec`** protocol merge — the `IndexDimension`/
  `CategoricalDimension` reuse is solid now; the shared protocol is
  validate-at-workload.

Anti-patterns (carried from `lazy-and-bundle.md` §9, sharpened here):

- A separate lazy field type (§2).
- `offload`/`partition(spill=)` of a resident reshape (§4).
- Fusion-into-`window_expansion_plan` as the central uniform-batch mechanism (§9).
- A `lazy=` flag — laziness is operand/input-driven, never a boolean mode.
