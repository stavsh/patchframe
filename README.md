# patchframe

`patchframe` is a dataframe-first infrastructure for datasets that combine:

- **tabular metadata and annotations**
- **lazy access to multidimensional array data**
- **typed schema**
- **explicit dataset operations**
- **source provenance**

The goal is to provide a clean foundation for datasets that need both:

1. the expressive power of a table-oriented workflow (`pandas`)
2. the lazy data-access behavior typically associated with ML / dataset pipelines

## Status

`patchframe` is in early development.

The initial implementation is focused on a small, general core with:

- no geometry in the core package
- explicit schema / bindings / provenance
- lazy data access through tiny `DataAccessor` objects
- generic dataset operations
- pluggable runtime and storage backends

Geometry, DAS-specific slicing, SQL-backed dataset makers, and other domain-specific functionality are intended to live in extension packages rather than in the core.

---

## Design goals

### Dataframe-first
Datasets should remain easy to inspect, filter, join, concatenate, and transform using familiar dataframe-style operations.

### Lazy data access
Rows may contain lazy references to array-like data without materializing that data until needed.

### Typed schema
A dataset has an explicit schema, with field types such as:

- index fields
- regular value fields
- data fields
- slice-spec fields

### Explicit bindings
Relationships between fields are represented explicitly through binding specs rather than implicit coupling.

### Provenance-aware
Datasets carry information about where they came from and how they were derived.

### Operator-driven design
Operations should define their effects on dataset structure explicitly, rather than relying on ad hoc dataframe mutation.

---

## Core concepts

### Dataset state

A dataset is defined by four top-level parts:

1. **Schema**  
   Field definitions and column types.

2. **Table**  
   The underlying `pandas.DataFrame`.

3. **BindingSpecSet**  
   Serializable declarations of relationships between fields.

4. **Provenance**  
   Source information and lineage metadata.

### DataAccessor

A `DataAccessor` is a tiny lazy object stored in a data column.

It identifies:

- which source a row refers to
- which logical item inside that source is being accessed
- which asset is being referenced
- which lazy view / slice should be applied

### DataSource

A `DataSource` is the runtime object that knows how to interpret and materialize `DataAccessor`s.

### SourceDescriptor

A `SourceDescriptor` is the durable, serializable identity + reopen recipe for a source.

### SourceManager

A `SourceManager` is a process-local runtime manager for live `DataSource` handles.

### ArrayStore / MetadataStore / SourceIOAdapter

Persistence is decomposed into:

- **ArrayStore**: storage of lazy array assets
- **MetadataStore**: storage of non-lazy metadata
- **SourceIOAdapter**: dataset-level save/load/append integration

---

## Planned source types

The first source types planned for the core package are:

- **MemorySource**  
  In-memory numpy-backed source for development and testing.

- **DataFrameSource**  
  Persistence-oriented source using dataframe-friendly serialization (pickle / Arrow).

- **MockSource**  
  Deterministic synthetic source for testing slicing and data access behavior.

---

## Planned package layout

```text
patchframe/
  dataset/
  data/
  storage/
  ops/
  io/
  sources/