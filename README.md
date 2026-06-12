# patchframe

`patchframe` is a dataframe-first infrastructure for datasets that combine:

- tabular metadata and annotations
- lazy access to multidimensional array data
- typed schema
- explicit dataset operations
- source tracking

For a user-facing overview of why this package exists and where it is useful,
start with [Intro / Motivation](docs/intro.md).

The core package is intentionally small and non-geometric. It models datasets as typed **fields** transformed by **operators** and related through explicit **couplings**. Structural changes are declared through explicit **transitions**.

## Status

Early development.

## Design posture

Patchframe is built by using its abstractions hard enough that weak ones
"shatter in your hands." Real examples drive the shape of the package; when a
tool cracks under use, the surviving pieces become the next design.

## Core ideas

A dataset is defined by four top-level aspects:

1. **Schema** — field definitions and column types
2. **Table** — the underlying `pandas.DataFrame`
3. **CouplingSet** — serializable declarations of relationships between fields
4. **Sources** — `DatasetSourceInfo` records tracking where data came from

Sources are preserved through operations by default. N-ary operators (e.g. concat) union the source sets of their inputs. Creation operators introduce new source records. This enables workflows such as upsert-back-to-source and multi-source labeling UIs.

The persistence layer is built from:

- **ArrayStore**
- **MetadataStore**
- **SourceIOAdapter**

The lazy data layer is built from:

- **DataAccessor** — tiny lazy object stored in table cells
- **DataSource** — runtime object that interprets and materializes accessors
- **SourceDescriptor** — durable reopen recipe for a source
- **SourceManager** — process-local manager of live source handles

### User-facing data sources

`DataSource` is the low-level runtime contract, but it should not be the only
extension point users are expected to implement directly. Source authors should
not have to repeatedly hand-write descriptor roundtrips, dimension-name
validation, or generic slice application.

Source capability flags make the expected behavior explicit:

- `runtime`: usable through a live `SourceManager` in the current process.
- `reopenable`: `describe()` returns a `SourceDescriptor` complete enough for
  `type(source).open(descriptor)` to reconstruct an equivalent source.
- `portable`: the descriptor is serializable enough for worker/process transfer.

Plain `DataSource` defaults to runtime-only. `ArrayDataSource` defaults to all
three capabilities because it generates `describe()` and `open()` from declared
configuration fields. Sources that cannot provide persistent or portable
configuration can still participate in patchframe as runtime sources, but they
should fail clearly if used in contexts that require reopening or worker
transfer.

The intended direction is a higher-level array/source base class for common
external data sources. That base should own:

- `describe()` and `open()` roundtrip boilerplate from a serializable source
  configuration
- stable `source_id` construction from source identity/configuration
- `SourceDescriptor.open_config` completeness checks
- dimension-name validation in `slice_accessor`
- generic `DimensionedSlice` resolution and NumPy-style slice application

User implementations should usually provide only source-specific IO:

- `read_full(item_id, accessor)` for simple sources
- `read_partial(item_id, resolved_slice, accessor)` for sources that can read a
  slice efficiently without loading the full item

Partial-read support should be explicit, for example through a
`supports_partial_read` flag, rather than discovered by exception handling on
the materialization path. The default `read(...)` behavior can load the full
item and apply the resolved slice when partial reads are not supported.

A testing utility should also exist for source authors, tentatively
`assert_source_contract(...)` or `validate_source(...)`. It should verify that:

- `source.describe()` returns a valid `SourceDescriptor`
- `type(source).open(source.describe())` can reopen an equivalent source when
  `reopenable=True`
- `open_config` is complete and serializable enough for worker processes when
  `portable=True`
- source identity and dimensions survive the descriptor roundtrip
- unknown slice dimensions are rejected
- full and sliced materialization work
- when feasible, partial-read output matches full-read-plus-slice output

This preserves `DataSource` as an escape hatch for unusual backends while giving
normal examples such as WAV, raster, Zarr, or image-folder sources a safer and
smaller implementation surface.

### Dimensional slicing note

Patchframe currently has several objects related to dimensional slicing:
`Dimension`, `Dimensions`, `DimensionIndex`, `DimensionedSlice`, and the
dimensioned-slice extension array. `ResolvedSlice` is currently a thin helper
that keeps dimension names available through slice resolution and source
materialization. Treat it as provisional: the dimensional slicing model may
need consolidation before that helper becomes a long-term public concept.

## Dataset identity invariant

The table index is a hard dataset invariant: it must be unique, and it is named
after the schema's primary `IndexField` so row identity is self-describing.
`IndexField.validate_column` enforces the name match, and `make_from_dataframe`
names the index for you (operators that rename or rebuild the index keep the two
in sync). In patchframe, the DataFrame index is row identity, not only a pandas
alignment label. This constraint may be relaxed around the number of index-like
fields in the future, but the primary dataset row identity must remain unique.

## Semantic state propagation

Patchframe state should describe semantic meaning, not only the current
materialized content. Semantic state is minted at creation boundaries and then
preserved, replaced, or invalidated by operator-specific transition rules.

Content-derived fingerprints are still useful for validation, caching, and
diagnostics, but they should not define semantic identity. For example, a
filtered dataset can keep the same row identity namespace as its parent even
though its row content changed, while a newly generated plan dataset should
create its own row identity namespace and carry `ForeignIndexField` columns
back to the source namespace.

This propagation-first model is why schema fields, couplings, sources, and
index identities should be explicit state carried through operators rather
than repeatedly inferred from table values.

## Ergonomics Principle

Patchframe should keep common dataset usage natural while preserving explicit
structure where it matters. Complexity should be paid once, at the layer that
actually owns it:

- **Source and creation layer:** highest complexity is allowed here. A
  `DataSource` and its `make_*` creation operator should own source-specific
  details such as file layouts, metadata parsing, source descriptors,
  dimensions, asset IDs, reopen logic, and source-native validation.
- **Dataset definition layer:** moderate complexity is acceptable, but it
  should be applied once when defining the dataset. This includes schema
  fields, dimension bindings, couplings, label normalization, and
  source-specific convenience binding operators.
- **Conventional usage layer:** complexity should be very low. Filtering,
  joins, concat, merge, sampling, loading data, and training integration should
  feel close to normal dataframe/dataloader workflows. Users should not have
  to repeatedly manage source internals, dimension-binding boilerplate, or
  coupling mechanics.

Examples should demonstrate this hierarchy. Domain-specific examples may expose
helpers such as `make_audioset(...)`, `bind_audio_segments(...)`, or
`bind_inria_patches(...)` so that source-specific ceremony is written once and
ordinary use stays concise.

## Naming model

Internally, patchframe draws on QFT-inspired terminology:

- **Fields** — typed schema entities
- **Operators** — dataset transformations
- **Couplings** — explicit relationships between fields
- **Transitions** — structural effects declared by operators

## Operator families

Four operator base classes cover the full construction and transformation surface:

### `DatasetOperator`

Unary dataset-to-dataset transformer. Subclasses declare which aspects they modify via `transitions` (default: preserve everything) and override only the relevant `apply_*` hooks. Aspects not declared are passed through automatically with no code required.

### `CreationOperator`

Creates a dataset from external input. Subclasses must implement `generate_source_info` and `build`. The framework injects the source info into the state returned by `build` before assembling the `Dataset`.

### `PlanOperator`

Creates explicit plan datasets. Subclasses normalize their call signature into
`OperatorCall` and return a concrete plan dataset from `run`. Plan outputs are
sibling artifacts by default: they do not advance `DatasetContext`.

### `CompositionOperator`

Combines multiple datasets into one. All three structural hooks (`apply_schema`, `apply_table`, `apply_couplings`) are required — there is no sensible default for N-ary composition. Sources are unioned by default.

Operators declare their operands and call structure through an
`OperatorSignature`, and a shared interpreter handles `FieldHandle` inputs,
eager/deferred dispatch, and the returned result — see
[Lazy and eager duality](#lazy-and-eager-duality). Execution hooks (`apply_*`)
receive normalized datasets and local field selectors only. Dataset-level
operators such as `concat(...)` take whole-dataset operands; pass `ctx.dataset`
when the dataset itself is the operand.

All operator families route through `OperatorCall`. `field_handle_inputs` is the
legacy declaration seam, superseded by the signature for migrated operators;
operators no longer hand-write `__call__` for handle dispatch.

### Transition ontology TODO

`AspectTransition("derive")` is currently overused as a catch-all for "this
operator changes the aspect somehow." That is useful during early development
but should not become the final ontology. Transitions should move toward a
small, explicit vocabulary whose modes can be checked mechanically.

The long-term goal is operator contract testing: given an operator's declared
transition plan, test utilities should be able to validate that an
implementation preserves, mints, unions, drops, or rewrites each aspect
according to its declaration. This should let extension authors verify custom
operators against patchframe's structural invariants without hand-writing every
contract test.

## Lazy and eager duality

Every transform operator has two call forms, selected by operand type — there is
no separate lazy API and no `.lazy()` switch:

- **Eager:** a `Dataset` operand produces *and* applies the operation now,
  returning a new `Dataset`.
- **Deferred:** a `FieldHandle` operand records the operation and returns a
  handle, so calls chain; `consume` / `collect` applies it later.

```python
kept = where(ds, lambda df: df["score"] > 0)       # eager    -> Dataset
kept = where(b.field("cell"), pred, out="kept")     # deferred -> FieldHandle
result = kept.collect()                              # apply now -> Dataset
```

Handles come from the dataset entry bridge — `ds.field("image")` /
`ds.fields(["a", "b"])` for existing fields, or `ds.new_field(field_def)` to
declare a new (null-filled) field and get its handle — returning context-bound
`FieldHandle` / `FieldSelection` values that follow a field's *identity* across
operators (not its name). A handle resolves only inside its owning
`DatasetContext`. A deferred op threads one context forward so chains compose; an
eager op forks a new `Dataset` facade. `new_field` is a *cursor* operation (it
advances the shared context), so successive `new_field` calls accrete and their
handles co-resolve — ready to pass together to an operator.

A `FieldHandle` is also iterable for the eager per-row scan:
`for item_id, mask in materialize(ds.field("mask")).items(): ...` drives
coupling-aware row access (`dataset[item_id][name]`), materializing one value at
a time — the training memory profile — rather than bulk-loading the column.
(`__iter__` yields just the values.)

### Where a deferred operation lives

A deferred operation is recorded as a coupling. Operators that only add or fill a
field — schema `preserve`/`extend`, one output row per input row, and
per-row-independent (together: `coupling_able`) — record it directly on the
dataset (same level). Everything else (`where`, `merge`, `concat`, …) has no
honest same-level deferred form, so it lands on a one-row **bundle**: a
`BundleField` carrier whose cells hold whole datasets, plus an `ApplyOperator`
coupling that runs the operator between the cells (per row = per fiber). The
flat↔bundle morphisms are themselves ordinary operators:

- `bundle(left=…, right=…)` — lift datasets into `BundleField` cells.
- `extract(b.field("left"))` — pull one cell back out.
- `flatten(b)` — row-stack every cell (the total space).

`FieldHandle.collect()` is the terminal that materializes a pending field; it is
the single carve-out to "handles do not execute".

A recorded `ApplyOperator` carries a serializable `CallSpec` — the operator
(by class reference) plus normalized `args`/`kwargs` — and names its input/output
cells rather than holding them, so a deferred chain pickles independently of its
datasets (for persistence or worker dispatch). If a deferred call carries
something unpicklable (typically a `lambda` predicate), patchframe raises
`UnpicklableCallWarning` *at the moment the coupling is recorded*, not later at
`collect()` or save time; it still replays in-process, and the category can be
filtered to an error to require persistable chains. Pass a module-level function
instead of a `lambda`.

### Declaring operands

Operators declare their call structure dataclass-style, as class attributes the
metaclass collects into an `OperatorSignature` (the same mechanism that collects
`Parameter`s):

- `DatasetInput` — a whole-dataset operand (a `Dataset`, or a bundle handle for
  the deferred per-fiber arm); `variadic=True` for N operands (`merge`).
- `FieldInput` — a single field operand (a typed handle or a name);
  `output=True` marks a field that is also an in-place output
  (`slice_data.data_field`).
- `SelectionInput` — a multi-field operand.
- `ParamInput` — a per-call positional-or-keyword parameter that is *not* an
  operand (a predicate, a collision strategy). Distinct from `Parameter`, which
  is instance-level behavioural config.
- `FieldOutput` — a caller-named produced field (`merge(…, out="merged")`); the
  chaining point for lifting operators.
- `returns` — `FieldReturn` / `SelectionReturn` / `DatasetReturn`, the eager↔lazy
  seam. A handle return marks a coupling-producer whose deferred arm hands back a
  handle; `DatasetReturn` is for eager-only operators and the bundle entry/exit
  bridges.

A shared interpreter reads the signature, binds the call, selects eager vs
deferred by operand type and same-level vs bundle by `coupling_able`, and returns
the declared handle or `Dataset`. Operators implement only their `apply_*` hooks;
the duality is provided for free.

The duality is wired across the transform operators. `where`, `merge`, `concat`
(and `concat_rows`/`concat_columns`), `rename`, `drop`, `keep`, `set_index`,
`join`, `link`, and `explode` defer onto a `BundleField` carrier; `slice_data`,
`materialize`, and `compose_slice` defer in place as same-level couplings.
`assign`/`add_column` produce *named* columns from values rather than from a field
handle, so their handle form creates the targets first with `new_field`:
`assign([h_a, h_b], values)` fills the (null-filled) fields from a frame keyed by
name and returns a `FieldSelection`; `add_column(handle, values)` is the
single-field counterpart, returning the handle — no `out` needed, since the
outputs are already named.

Assignment follows pandas' two conventions, split by mutability. The functional
form is `ds.assign(c=values)` — returns a new dataset; bare values infer a
`ValueField`, `(field_def, values)` adds a typed field. The in-place forms live
on the mutable session types: `ctx["c"] = values` on the cursor (a `Field` key —
`ctx[field_def] = values` — adds a typed field stating the name once), and
`handle.loc[ids] = values` for label-scoped updates (scalar label, label list,
or boolean mask) — each desugars to the `assign` operator and advances the
shared context. `Dataset` itself is immutable and supports no item assignment.
assign assigns values to fields; a coupling's output field is not special-cased
— values land, and while the coupling is still pending, a later `consume` or
coupling-aware row access recomputes the field over them.

Consume is literal: a coupling is *pending work*, and `consume`/`collect`
complete it and discharge the couplings they ran — so consuming a chain twice
is work-idempotent, and assigning after consume is safe by construction. Row
access (`ds[item_id]`, `.items()`, `handle.loc`) is *evaluation*, not
consumption: it computes a row's pending work ephemerally — nothing is
persisted or discharged — which is exactly the per-row training profile;
`ds["col"]` reads storage.

Row access is also the *exit point* from the dataset world: after evaluation,
every value converts to plain Python through its field's exit conversion
(`Field.exit_value`, or `register_field_exit` for field types you do not own)
— a `BundleField` fiber leaves as a list of records, recursively. The storage
surface keeps framework objects; hold a fiber as a `Dataset` there when you
want lazy navigation. For training loops, `ds.rows()` is the positional view:
a duck-typed map-style dataset (`len` + integer indexing + batched fetch), so
`DataLoader(ds.rows(), ...)` works directly with no torch dependency in
patchframe; `ds.rows("sample")` makes each item one field's value.
`window_expansion_plan` is a transform (source → plan) and a bundle-lifter like
`explode`: a handle operand defers it onto a `BundleField` carrier, and its
`field`/`bindings` are passed as *names* — replay data resolved against the
(possibly deferred) source at run time. There is no eager handle resolution
anywhere on the transform surface: a `FieldHandle` input always selects the
lazy arm. Only the true creation operators (`make_from_dataframe`,
`make_plan`) and the terminals (`extract`/`flatten`/`consume`/`collect`) take
handles without deferring; deferred creation is a future workload-gated tier.

See `examples/lazy_eager_duality_usage.py` for a runnable end-to-end pipeline
(`bundle` → deferred `merge`/`where`/`drop` → one `collect()`), checked against
the eager equivalent.

## Composition policies

This section documents the intended composition model. The concat behavior is
implemented first; the join/merge split is the design target that merge should
move toward.

Composition in `patchframe` is not just dataframe concatenation. A composition
operator must combine:

- table values
- typed schema fields
- coupling declarations
- source/provenance records

This is the main reason patchframe composition is stricter than `pandas`.
Pandas can keep duplicate column names, append suffixes, or silently widen
dtypes. Patchframe has to preserve a predictable dataset schema and coupling
graph, so ambiguous cases raise by default.

### Policy layers

Composition is organized as several policy layers:

1. **Preparation policy**: rename, drop, or prefix fields before composition.
   This must rewrite schema fields, table columns, and coupling `FieldRef`
   references together. Automatic preparation is not implemented yet; rename
   collisions currently raise.
2. **Row field compatibility policy**: decide whether fields with the same name
   can be stacked by rows.
3. **Column bucket policy**: decide how fields behave when they coexist in the
   same output schema, even if their names do not collide. Primary-field
   downgrade is a bucket policy.
4. **Column collision value policy**: decide how same-name table columns are
   resolved when both sides contribute values.
5. **Coupling policy**: decide which couplings survive composition and whether
   coupling references must be rewritten.
6. **Join policy**: decide which rows from two datasets correspond. This is
   intentionally separate from merge.

### Field policies

Field composition is implemented through an external policy registry. The
current built-in policies are:

- `Field`: row stacking requires the same concrete field type and same dtype.
  A type or dtype mismatch raises `TypeError`.
- `ValueField`: row stacking requires the same concrete field type, but dtype
  differences are allowed. If dtypes differ, the output field has `dtype=None`
  and pandas owns the resulting value dtype.
- `DimensionField`: row stacking and key coalescing require the same concrete
  field type, same dtype, and same `dimension` object. Dimension mismatch raises
  `TypeError`.
- `IndexField`: the first output index remains an `IndexField`. Additional
  dataset indexes are downgraded into nullable, table-backed
  `IndexColumnField` fields.

All normal table columns are nullable by design. `IndexField` is special because
it represents the DataFrame index itself. `IndexColumnField` stores index values
that were downgraded into an ordinary table column during composition.
`ForeignIndexField` is the table-backed index-reference field for labels that
point to another dataset's index identity.

### Column collision values

When two table columns map to the same output field, the value policy is
configured with `ColumnCollisionStrategy`:

- `mode="error"`: raise on collision. This is the default.
- `mode="keep"`: keep the chosen side and discard the other side.
- `mode="update_missing"`: start with the chosen side and fill missing values
  from the other side.
- `mode="coalesce"`: currently equivalent to `update_missing`.
- `mode="rename"`: raise for now. Rename/prefix/drop preparation is not
  implemented yet.

`side="left"` chooses the existing/left column. `side="right"` chooses the
incoming/right column.

`on_conflict="raise"` raises when both sides have non-null, unequal values in
the same row. `on_conflict="keep_chosen"` keeps the chosen side in that case.

The important difference from pandas is that patchframe does not use automatic
suffixes as the default conflict resolution. A same-name field collision is a
schema decision, not just a dataframe-label issue.

### Coupling policy

Couplings are declarations over field names. If a composition preparation step
renames a field, coupling `FieldRef`s must be rewritten through the same mapping.
The existing `rename` operator already does this through
`CouplingSet.rewrite_field_names(...)`. Composition rename/prefix preparation
must use the same rule.

Current coupling defaults:

- Column concat: union coupling sets after any field-reference rewrites. Exact
  duplicate couplings are deduplicated. Coupling-specific simplification is not
  implemented yet.
- Row concat: preserve couplings only when all inputs have exactly the same
  `CouplingSet`. If coupling sets differ and any are non-empty, row concat
  raises. The user should consume the coupled fields first, then reapply
  couplings after concat.
- Merge: expected to use conservative coupling rules over the composed output.
  Coupling-specific merge policies are intentionally left as future extension
  points.

The row-concat rule is stricter than pandas because pandas has no equivalent of
a coupling graph. A single output `CouplingSet` applies to every row. Blindly
unioning different input couplings could make a coupling from one dataset affect
rows that came from another dataset.

### `concat_rows`

`concat_rows` stacks datasets by rows.

Schema behavior:

- Fields are matched by name.
- Field compatibility is checked through the row field policy.
- Missing columns are added as nullable columns filled with missing values.
- Primary fields are bucketed through column policy; later primaries of the
  same field type are downgraded.
- Additional dataset indexes become `IndexColumnField` columns.

Raises:

- Raises if no datasets are provided.
- Raises if same-name fields are incompatible by policy.
- Raises if `DimensionField` dimensions differ.
- Raises if non-empty coupling sets differ across inputs.

Differences from pandas:

- Pandas row concat accepts many dtype coercions. Patchframe only relaxes dtype
  where field policy allows it, currently `ValueField`.
- Pandas has no dimension compatibility check.
- Pandas does not model couplings, so it cannot detect unsafe coupling unions.

### `concat_columns`

`concat_columns` composes datasets by columns and aligns rows by DataFrame
index.

Schema behavior:

- Fields are added in input order.
- Unique field names still pass through bucket policy.
- Primary downgrade can happen even without name collision.
- A second `IndexField` becomes an `IndexColumnField`; its values are
  materialized from that input index into the output table.
- Same-name fields are handled by `ColumnCollisionStrategy`.

Raises:

- Raises if no datasets are provided.
- Raises on same-name collisions with the default `mode="error"`.
- Raises for `mode="rename"` because automatic rename preparation is not
  implemented yet.
- Raises when `on_conflict="raise"` and both sides contain unequal non-null
  values.
- Raises if field policies reject the collision field composition.

Differences from pandas:

- Pandas `concat(axis=1)` can produce duplicate column names. Patchframe schemas
  cannot.
- Pandas does not have primary field buckets. Patchframe downgrades later
  primary fields even when names differ.
- Pandas treats indexes only as alignment labels. Patchframe also models index
  identity in the schema, so secondary indexes are explicit table columns.

### `join`

Join should be separate from merge. A join operator decides row correspondence
and outputs a simple dataset representing the mapping between two input
datasets.

The join-plan dataset has its own unique row index, usually represented by
`IndexField(name="join_id")`. Its canonical mapping columns should be index
labels from the input datasets:

- `left_index`: nullable index value into the left dataset
- `right_index`: nullable index value into the right dataset

Optional columns may include:

- `left_pos` and `right_pos` as cached positional lookups
- score, rank, distance, overlap, or strategy-specific metadata

Index labels are canonical because dataset indexes are unique by invariant.
This lets a join-plan dataset behave like any other patchframe dataset: users
can filter it, concatenate it with other compatible join plans, inspect it,
persist it, and then pass the resulting plan to merge. Positional columns can
still be useful as an optimization, but they are not the semantic identity of
the join.

Possible join strategies:

- equality on one or more fields
- index alignment
- left, right, inner, or outer inclusion rules
- nearest temporal match
- interval overlap
- spatial join in an extension package
- scored or ranked candidate retrieval

Raises:

- Raises if a strategy references missing fields.
- Raises if a strategy requires compatible field semantics and the fields are
  incompatible.
- Raises if a strategy cannot represent the requested match type without unique
  row identity.

Difference from pandas:

- Pandas hides the row-matching plan inside `merge`. Patchframe exposes it as a
  dataset so users can inspect, filter, concatenate, visualize, cache, sample,
  or reuse the match plan before materializing a merge.

### `merge`

Merge should consume two datasets and a join-plan dataset. It should not decide
which rows match.

Expected merge behavior:

- Validate that the join plan contains valid `left_index` and `right_index`
  mappings.
- Gather rows from the left and right datasets by unique index label according
  to the join plan.
- Compose output fields through column bucket policy.
- Resolve same-name non-key fields through `ColumnCollisionStrategy`.
- Apply conservative coupling policy.

Raises:

- Raises if the join plan is missing required mapping columns.
- Raises if non-null index labels in the join plan are missing from the
  corresponding input dataset.
- Raises on same-name collisions with default `mode="error"`.
- Raises for `mode="rename"` until rename/prefix preparation is implemented.
- Raises when field policies reject a composed field.
- Raises when coupling policy cannot preserve semantics safely.

Differences from pandas:

- Pandas `merge` combines join planning and row materialization in one call.
  Patchframe separates them.
- Pandas allows duplicate index labels. Patchframe requires unique dataset
  indexes, so merge can use index labels as stable row identity.
- Pandas automatically suffixes overlapping non-key columns. Patchframe raises
  unless an explicit collision policy resolves the collision.
- Pandas does not validate typed schema semantics or coupling safety.

## Plan datasets

Some dataset operations have two separable phases:

1. decide what should happen
2. materialize the result

Patchframe should make the first phase explicit when it is useful to inspect,
filter, sample, cache, concatenate, or replace before materialization. The
intermediate artifact is a **plan dataset**: an ordinary `Dataset` whose rows
describe a later operation.

The implemented join/merge split is the first example:

```python
plan = join(left, right)
merged = merge(left, right, plan)
```

The plan is not a hidden dataframe inside `merge`. It is a dataset with its own
unique row index and mapping columns. Users can inspect or modify it before
applying it.

The naming convention should be:

- Specific planner names create plan datasets.
- Generic operators apply plan datasets when their schema contains the required
  planning fields.
- Plan datasets remain normal datasets unless a future execution layer adds an
  explicit lazy/chunked plan object.

### Window expansion plans

The next planned use of this idea is dimensional expansion: a general form of
tiling, patch extraction, clipping, video windowing, DAS windowing, or other
N-dimensional row expansion.

A dimensional expansion plan describes output rows by mapping each row back to
a source dataset row plus a slice. The current core shape is:

- `source_index`: index label of the source row
- `slice`: a `DimensionedSliceField` backed by `DimensionedSliceArray`
- optional plan metadata such as sampling reason, score, label id, overlap, or
  strategy

`window_expansion_plan` is the first implementation. It accepts extents from
either a single `DimensionedSliceField` or from explicit `DimensionField`
bindings in the same style as `compose_slice`. Single-column inputs may
contain null rows, which are skipped. Multi-field bindings reject nulls because
partial bounds across multiple columns are ambiguous.

The default core planner is deliberately narrow: axis-independent regular
windows over any number of sliceable dimensions. `AxisWindow(size=..., stride=...,
offset=..., include_partial=...)` only answers: "given an extent on one
dimension, which start/stop intervals should be emitted?" The performance
critical expansion lives on `DimensionedSliceArray.explode_windows(...)`, so
the operator wrapper normalizes inputs into a slice array and keeps the window
grid construction NumPy-based. `DataField`-backed planning is intentionally not
accepted yet because source extent lookup is not columnar until a future
`DataAccessorArray` or batch extent API exists. Label-driven, geometry-driven,
source-native, random, adaptive, or multiscale planning can produce the same
plan dataset through extension-owned code.

`DatasetState.metadata` can carry lightweight, non-executable dataset metadata.
Window expansion plans mark themselves under `patchframe.plan`; for now this is
used to warn when `window_expansion_plan` is called on an existing plan dataset.
Planning over a plan currently means the new `source_index` values point to the
input plan rows, not to the original source dataset. Dedicated plan-refinement
semantics should be designed before relying on repeated planning.

Applying such a plan is a generic operation named `explode`, not an aerial- or
raster-specific `retile`. It gathers source rows by a `ForeignIndexField`,
overlays compatible plan fields, and preserves source couplings. It does not
invent new slice bindings or materialize arrays unless a materialization
coupling is explicitly present or consumed. If a plan dataset has couplings,
`explode` warns that plan couplings are ignored and that the user should
consider consuming plan bindings before applying the plan.

`make_plan(target, source_index=...)` is the minimal generic plan constructor.
It creates a fresh plan index plus one validated `ForeignIndexField` into the
target dataset. `assign(...)` can then add multiple payload columns, inferring
`ValueField` by default or accepting `(Field, values)` for explicitly typed
columns. `make_plan.from_dataframe(...)` and `make_plan.from_series(...)`
provide short paths for table-shaped and one-column inputs. This keeps sparse
extension planners focused on generating records rather than rebuilding plan
schema and index-identity boilerplate.

For imperative pipeline authoring, `Dataset.context()` provides an explicit
mutable cursor over immutable dataset snapshots. Unary operators may omit their
dataset while a context is active, and `FieldHandle`s follow surviving fields
through `FieldIdentity`:

```python
with patches.context() as ctx:
    patch = ctx.field("patch")
    image = ctx.field("image")
    slice_data(patch, image)
    materialize(image)
    consume(image)

patches = ctx.dataset
```

`make_plan(ctx.field("item_id"), source_index=...)` accepts an index handle but
does not advance the context: the returned plan is a sibling artifact.
Creation operators and `join(...)` follow the same sibling-output rule.
Unary operators advance automatically. Composition operators advance only
when the current context dataset is passed as an explicit input. `explode(plan)`
uses the active context dataset as its source and advances that context.

Field handles are intentionally not generic dataset pointers. Use them where an
operator parameter names a specific field, such as `slice_data(patch, image)`,
`consume(image)`, `make_plan(ctx.field("item_id"), ...)`, or
`window_expansion_plan(ctx.dataset, field=ctx.field("extent"), ...)`. Use
`ctx.dataset` for whole-dataset operations such as `concat(...)`.

Keeping the source mapping column, for example `source_index`, may be useful
for traceability but changes the output schema. This is a concrete case where
an operator's transition contract may depend on user flags, so it should wait
until transition declarations can express flag-dependent schema behavior.

This is important for sparse-label and large-extent datasets. Patchframe should
not require the user to generate every possible tile and then spatially join
labels just to discover the small subset of useful patches. Dense tiling is one
valid plan creation strategy, but label-driven, sampled, indexed, or extension
generated plans should feed the same apply operator.

### Lazy and batched plans

The first implementation should keep plan datasets concrete and in memory. A
generator-like or lazy plan object is a future execution-layer decision, not a
`DatasetState` feature yet.

Putting unmaterialized plans inside `DatasetState` would affect core
invariants such as `len(dataset)`, `dataset.table`, schema validation, coupling
execution, source propagation, row access, and composition. If lazy planning is
needed later, it should likely be explicit, similar to a groupby/planner object
that can yield normal plan datasets or plan chunks.

For now, the batch-friendly constraint is simple: plan application should accept
any valid plan dataset. Future batching can feed smaller plan datasets into the
same apply operator without changing the meaning of the operation.

## Current Development Status

Recent implementation work established the current composition and benchmark
baseline:

- Unique dataset indexes are enforced at `DatasetState` construction.
- `concat_rows` and `concat_columns` use field composition policies,
  nullable-column semantics, collision strategies, and conservative coupling
  rules.
- `join` now produces an explicit join-plan dataset with `left_index` and
  `right_index` mapping columns.
- `merge` consumes left/right datasets plus a join-plan dataset. It validates
  mapping labels, preserves join-plan metadata columns, gathers rows by index
  label, applies collision policy, and unions couplings like column
  composition.
- `BindDimensions` supports generic dimension bindings and writes
  `DimensionedSliceArray` columns through `consume` without row-level
  `DimensionedSlice` materialization.
- Benchmark scaffolding exists under `benchmarks/` for concat, join, merge,
  window expansion plan creation, and non-materializing consume paths. Benchmark
  results are local artifacts and are ignored by git.
- `DimensionedSliceArray` missing-mask construction uses vectorized
  `pd.isna`, which keeps million-row `consume(BindDimensions)` runs in the
  sub-second range for the current benchmark shape.
- `window_expansion_plan` produces concrete window expansion plans with
  `source_index` and columnar `slice` fields. It can plan from
  `DimensionedSliceField` extents or explicit `DimensionField` bindings, with
  window expansion handled by `DimensionedSliceArray`. Its field-scoped
  `field` and `bindings` parameters accept `FieldHandle`s.
- `make_plan` and `assign` provide the low-ceremony path for sparse or
  extension-owned plan construction. Dense `window_expansion_plan` uses the
  same primitives.
- `DimensionJoin` produces join plans for bounded half-open interval overlap
  across named `DimensionedSliceField` dimensions. Optional equality scope
  fields keep local coordinate systems, such as per-tile pixel coordinates,
  from matching across unrelated rows.
- The AudioSet example demonstrates the intended ergonomics and source
  decoupling: `make_audio_files` parses the WAV source, `make_audioset_labels`
  parses labels/segments, `merge_audio_labels` composes them through
  `join`/`merge`, and `bind_audio_segments` applies dataset-definition
  couplings once. Conventional usage gets implicitly sliced audio accessors
  through row access. It also includes waveform/spectrogram/label-count
  visualizations and a basic optional PyTorch `DataLoader`.
- The Inria Aerial Image Labeling example demonstrates raster-native expansion
  on a concrete building-segmentation dataset without geometry-specific core
  logic: `make_inria` discovers RGB GeoTIFFs and aligned training masks,
  `make_inria_mask_bbox_plan` emits connected-component bounding boxes,
  `make_inria_patch_plan` produces candidate windows through
  `window_expansion_plan`, scoped `DimensionJoin` selects windows overlapping
  mask boxes, and `bind_inria_patches` applies filtered plans through `explode`
  before binding deferred GeoTIFF window reads.
- Transform operators expose a lazy ↔ eager duality through operand-type
  dispatch: a `Dataset` operand applies eagerly and returns a `Dataset`; a
  `FieldHandle` operand defers and returns a handle. A signature-driven
  interpreter (`OperatorSignature` with `DatasetInput` / `FieldInput` /
  `SelectionInput` / `ParamInput` / `FieldOutput` slots) routes same-level
  couplings vs `BundleField` bundles by the derived `coupling_able` test, so
  operators get both arms from declarations alone — no hand-written `__call__`
  dispatch. Wired across the transform operators (`where`, `merge`, `concat`,
  `rename`, `drop`, `keep`, `set_index`, `join`, `explode`, and the coupling-
  authoring `materialize`/`slice_data`/`compose_slice`); the entry/exit bridges
  (`bundle`/`extract`/`flatten`/`collect`) and `Dataset.field()`/`fields()`
  complete the surface. A runnable end-to-end example lives in
  `examples/lazy_eager_duality_usage.py`. See
  [Lazy and eager duality](#lazy-and-eager-duality).
- The coupling-authoring operators dropped their `bind_` prefix now that the
  duality reads on a bare verb: `bind_materialize` → `materialize`, `bind_slice`
  → `slice_data`, `bind_dimensions` → `compose_slice`. The old names remain as
  deprecated `pf.*` aliases that warn and forward.
- Deferred operator calls are recorded as `ApplyOperator` couplings carrying a
  serializable `CallSpec` (operator-by-class + normalized `args`/`kwargs`),
  referencing their cells by name so a deferred chain pickles independently of
  its datasets. An unpicklable argument surfaces as `UnpicklableCallWarning` at
  defer time (when the coupling is recorded), not at `collect()`/save; the
  category is filterable to an error to require persistable chains.

## Performance Direction

Patchframe should keep operator APIs independent from the table execution
engine where practical. The default in-memory table remains pandas because it
supports nullable columns, object columns, extension arrays, and the current
schema validation model directly.

For table-heavy relational operators, a future `engine=...` parameter may be
added after benchmark validation. Candidate operators include:

- `concat_rows`
- `concat_columns`
- `join` for field equality joins
- `merge` for join-plan materialization

Possible engine values:

- `engine="pandas"`: default behavior, preserves current semantics.
- `engine="polars"`: optional acceleration path for supported schemas.
- `engine="auto"`: future dispatch once benchmark data and compatibility rules
  are mature enough.

The first Polars integration should be benchmark-only. It should measure:

- pandas end-to-end runtime
- pandas-to-Polars conversion time
- Polars operation runtime
- Polars-to-pandas conversion time
- result equality against the pandas implementation

Polars must treat the pandas index explicitly as a reserved identity column,
then restore it and validate uniqueness when converting back. This matters
because patchframe uses the DataFrame index as dataset row identity, while
Polars has no pandas-style index.

Known constraints before promoting `engine="polars"` into core operators:

- `DataAccessor` columns are Python object columns and may reduce or eliminate
  Polars benefits.
- `DimensionedSliceArray` is a pandas extension array and should not be routed
  through Polars unless it can be preserved or rebuilt without scalar
  object materialization.
- `consume(BindDimensions)` is primarily a NumPy/extension-array construction
  path and is not expected to benefit from a Polars backend.
- Any alternate engine must preserve schema policies, nullable semantics,
  collision behavior, coupling field references, and the unique-index
  invariant.

## Future Extensions And TODOs

### Field Handle Ergonomics

The symbolic field-handle layer has landed. `ds.field("image")` /
`ds.fields([...])` return context-bound `FieldHandle` / `FieldSelection` values
that follow a field's identity and resolve only inside their `DatasetContext`,
with no global pointer manager of live datasets or fields — see
[Lazy and eager duality](#lazy-and-eager-duality).

Remaining directions:

- Make distinct operator call structures **explicit overloads** (in the spirit
  of `typing.overload`) instead of one signature plus a binding interpreter; the
  binding is already shaped to generalize to "try each overload".
- The **lift** case for mixed `Dataset`/handle operands (mint a carrier on the
  fly), currently an explicit error in favour of bundling first.
- A workload-gated **deferred-creation** (`DatasetAccessor`) tier and the
  tall/streaming bundle substrate.

### Dask Execution Extension

Dask support should be explicit and extension-owned. The preferred user-facing
shape is to return a Dask collection or delayed graph and let the user call
`.compute()` directly:

```python
lazy = patchframe_dask.map_field(ds, input_field="audio", output_field="snr", fn=compute_snr)
result = lazy.compute()
```

Core operators should not hide Dask execution behind normal in-memory operator
calls. Candidate extension workloads:

- large-scale materialization of `DataAccessor` columns
- partitioned feature computation, such as SNR over audio segments
- local reductions over sliced arrays, such as argmax over model likelihood maps
- batch embedding or quality-metric computation

Design constraints to preserve now:

- `DataAccessor`, `DimensionedSlice`, coupling declarations, and operation
  specs should stay small and pickle-friendly.
- `SourceDescriptor` must be sufficient for a worker process to reopen a
  `DataSource`; workers should not depend on a process-shared `SourceManager`.
- Live file handles, caches, locks, GPU handles, and runtime managers should
  stay out of serialized dataset state and task graphs.
- Dask work should be partitioned by row blocks or source-native chunks, not
  one task per row.
- Outputs should retain dataset index labels so computed results can be joined
  or assigned back deterministically.

### Access Path And Async IO

Future row access may need async support for IO-heavy sources. Keep this
separate from table composition operators:

- `dataset[row_id]` can remain synchronous for the core API.
- An extension or alternate access API may expose async/concurrent row reads.
- Async should target IO-bound materialization and inspection, not concat,
  join, merge, or `BindDimensions` metadata construction.

### Dimension And Join Extensions

Dimension-aware joins remain a major future design area:

- generic dimension-scope join dispatch
- interval overlap joins
- nearest temporal joins
- geometry/spatial joins in an optional extension package
- scored or ranked retrieval joins

The built-in package should stay non-geometric. Geometry dependencies belong
in extensions.

### Benchmark TODOs

- Add benchmark-only Polars candidate paths before adding `engine="polars"` to
  any core operator.
- Add larger manual benchmark presets for `1_000_000+` rows and many nullable
  columns.
- Add access-path benchmarks for row access, coupling application, and future
  async/materialization workloads.
- Keep CI tests small and correctness-oriented; do not enforce wall-clock
  thresholds until stable baseline data exists.

## Development setup

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

To run examples that need visualization, WAV IO, or PyTorch:

```bash
pip install -e ".[examples]"
```
