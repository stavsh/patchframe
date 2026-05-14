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

## Dataset identity invariant

The table index is a hard dataset invariant: it must be unique. In patchframe,
the DataFrame index is row identity, not only a pandas alignment label. This
constraint may be relaxed around the number of index-like fields in the future,
but the primary dataset row identity must remain unique.

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
`bind_aerial_patches(...)` so that source-specific ceremony is written once and
ordinary use stays concise.

## Naming model

Internally, patchframe draws on QFT-inspired terminology:

- **Fields** — typed schema entities
- **Operators** — dataset transformations
- **Couplings** — explicit relationships between fields
- **Transitions** — structural effects declared by operators

## Operator families

Three operator base classes cover the full construction and transformation surface:

### `DatasetOperator`

Unary dataset-to-dataset transformer. Subclasses declare which aspects they modify via `transitions` (default: preserve everything) and override only the relevant `apply_*` hooks. Aspects not declared are passed through automatically with no code required.

### `CreationOperator`

Creates a dataset from external input. Subclasses must implement `generate_source_info` and `build`. The framework injects the source info into the state returned by `build` before assembling the `Dataset`.

### `CompositionOperator`

Combines multiple datasets into one. All three structural hooks (`apply_schema`, `apply_table`, `apply_couplings`) are required — there is no sensible default for N-ary composition. Sources are unioned by default.

All three families support a full escape hatch by overriding `__call__` directly.

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

## First concrete operator

- `make_from_dataframe` — turns an existing dataframe and explicit schema into a dataset

## Planned first sources

- `MemorySource`
- `DataFrameSource`
- `MockSource`

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
  and non-materializing consume paths. Benchmark results are local artifacts
  and are ignored by git.
- `DimensionedSliceArray` missing-mask construction uses vectorized
  `pd.isna`, which keeps million-row `consume(BindDimensions)` runs in the
  sub-second range for the current benchmark shape.
- The AudioSet example demonstrates the intended ergonomics and source
  decoupling: `make_audio_files` parses the WAV source, `make_audioset_labels`
  parses labels/segments, `merge_audio_labels` composes them through
  `join`/`merge`, and `bind_audio_segments` applies dataset-definition
  couplings once. Conventional usage gets implicitly sliced audio accessors
  through row access. It also includes waveform/spectrogram/label-count
  visualizations and a basic optional PyTorch `DataLoader`.

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
