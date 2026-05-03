# patchframe

`patchframe` is a dataframe-first infrastructure for datasets that combine:

- tabular metadata and annotations
- lazy access to multidimensional array data
- typed schema
- explicit dataset operations
- source tracking

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

## First concrete operator

- `make_from_dataframe` — turns an existing dataframe and explicit schema into a dataset

## Planned first sources

- `MemorySource`
- `DataFrameSource`
- `MockSource`

## Development setup

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```
