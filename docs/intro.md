# Intro / Motivation

`patchframe` is for datasets where a row in a table is only the beginning.

Many ML and scientific datasets start as a dataframe-shaped object: one row per
clip, image, tile, patient, event, patch, or sample. But the actual data often
lives somewhere else: WAV files, video frames, arrays in a store, remote sensing
rasters, microscope images, simulation outputs, or model-generated embeddings.

Pandas is excellent at the table part. `patchframe` keeps that familiar table
center, then adds the dataset semantics pandas intentionally does not model:
typed fields, lazy multidimensional data access, explicit relationships between
columns, source tracking, and conservative dataset composition.

The goal is simple: let users work with complex datasets using dataframe-like
operations, without repeatedly rebuilding the fragile glue between metadata,
labels, file paths, array slices, and training code.

This matters most when datasets are not single monoliths. Real projects often
need to reuse the same raw data with multiple label sources, or reuse the same
labels with multiple data representations. Patchframe is designed around that
kind of data fusion.

## Who is this for?

`patchframe` is aimed at users who build, inspect, join, filter, label, and train
from datasets with external or multidimensional data:

- ML data engineers assembling training sets from files, annotations, and model
  outputs.
- Researchers working with audio, images, video, sensor data, simulation arrays,
  or scientific measurements.
- Labeling and active-learning tool builders who need rows, sources, labels, and
  array previews to stay connected.
- Dataset infrastructure authors who want reusable domain-specific loaders
  without baking one modality into the core package.
- Teams that already like pandas, but need dataset identity, provenance, lazy
  access, and safer joins than raw dataframe operations provide.

If your data is a small CSV and all values fit naturally inside scalar columns,
pandas is the right tool. `patchframe` starts to matter when a table row points
to something larger, sliced, source-backed, or expensive to materialize.

## The gap between pandas and datasets

In pandas, a dataset like AudioSet usually becomes a dataframe with columns such
as:

- `ytid`
- `start_seconds`
- `end_seconds`
- `labels`
- `audio_path`

That representation is useful, but it leaves important rules in user code:

- Which columns define the audio slice?
- Is `audio_path` a copied file, a read-only source, or a generated artifact?
- Does `start_seconds` mean seconds, samples, frames, or something else?
- After a join, do the labels still point at the correct audio file?
- After concat, are two fields with the same name actually semantically
  compatible?
- Can a row be loaded directly by training code without rewriting path and slice
  logic?

`patchframe` makes those rules part of the dataset state.

At the core, a `Dataset` is:

- a pandas-backed table
- a typed schema
- a set of couplings between fields
- source records describing where data came from

That extra structure is what lets the table stay easy to use while the dataset
remains explicit enough to survive joins, merges, filters, slicing, and lazy
row access.

## Where patchframe shines

### Lazy external data

Large arrays do not need to live inside dataframe cells. A data column can hold
small `DataAccessor` objects that point to a `DataSource`. The source knows how
to reopen and materialize the data only when needed.

For AudioSet, the table can hold audio accessors instead of waveform arrays.
The WAV file is read only when a row, plot, or batch actually needs audio.

### Reusable source loaders

Patchframe encourages reusable dataset loaders instead of one giant dataset
class per final training table.

In the AudioSet example, the WAV files and the label CSV are separate creation
operators:

- `make_audio_files(...)` knows how to represent the actual audio source.
- `make_audioset_labels(...)` knows how to parse one label/segment source.
- `merge_audio_labels(...)` uses normal patchframe join and merge machinery to
  compose them.

That split is important. If a new label format appears later, you do not need a
new all-in-one `AudioDatasetWithThreeLabelTypes` class. You write a new
`make_*_labels(...)` operator for that label source, then reuse the existing
audio loader and composition operators.

The same works in the other direction. If the labels stay the same but the data
source changes from WAV files to precomputed embeddings, spectrogram arrays, a
remote store, or a denoised audio collection, you can add a new source maker and
reuse the label loader.

This is the data-fusion case patchframe is built to make easier:

- fuse different label sources over the same underlying data
- fuse different data representations under the same labels
- keep each source loader small, testable, and reusable
- rely on shared join, merge, concat, schema, coupling, and provenance behavior
  instead of reimplementing dataset plumbing for every combination

### Dimension-aware slicing

Slice values can stay in natural dataset units. Audio segments can be described
in seconds. A temporal dimension can resolve those seconds into sample indices
at materialization time.

That is different from scattering `int(start * sample_rate)` across notebooks,
dataloaders, plotting helpers, and preprocessing scripts.

### Couplings instead of repeated glue code

Patchframe can record that one field slices another field. In the AudioSet
example, segment start/end columns are bound once to the audio field. After
that, ordinary row access can return the sliced audio segment directly.

The dataset definition pays the complexity once. Conventional usage stays
small.

### Safer composition

Pandas is permissive because it is a table library. It can suffix overlapping
columns, accept duplicate labels in many places, and coerce dtypes liberally.

Patchframe is stricter because it is carrying dataset meaning:

- row identity must be unique
- fields have semantic types
- dimension fields must be compatible
- coupling references must remain valid
- source records should be preserved
- ambiguous column collisions should raise unless explicitly resolved

This strictness is useful when a bad merge can silently corrupt a training set.

### Inspectable joins

Patchframe separates join planning from merge materialization. A join can
produce a join-plan dataset with `left_index` and `right_index` mappings. That
plan can be inspected, filtered, cached, visualized, sampled, or reused before
the final merge.

For dataset work, the join plan is often as important as the merged table.

### Source tracking

Creation operators introduce source records, and composition operators preserve
or union them. This is the start of practical provenance: a dataset can keep
track of which metadata file, array store, file directory, or generated source
contributed to it.

## How it differs from pandas

`patchframe` uses pandas; it does not try to replace it.

| Use pandas when... | Use patchframe when... |
| --- | --- |
| Your data is primarily scalar table data. | Rows point to files, arrays, slices, or external sources. |
| You want maximum flexibility for quick table manipulation. | You want dataset operations to preserve schema, couplings, sources, and row identity. |
| Duplicate column names or automatic suffixes are acceptable. | Ambiguous field collisions should be treated as schema decisions. |
| Joins are just a way to produce another table. | Join plans are artifacts you may inspect, filter, cache, or reuse. |
| Object columns are enough to hold paths or arrays. | Data access should be lazy, typed, reopenable, and source-aware. |
| Downstream loading code can own all file/slice logic. | Dataset construction should define loading and slicing once. |
| Each final dataset gets a custom class. | Small source makers can be fused into many final datasets. |

A short version:

- pandas answers: "What table operation do you want?"
- patchframe also asks: "What dataset meaning must survive that operation?"

## AudioSet in one sketch

The AudioSet example demonstrates the intended ergonomics. Source-specific
details live in `examples/audioset.py`; normal usage remains concise.

```python
import patchframe as pf
from examples.audioset import make_audioset, make_torch_dataloader, plot_spectrogram

ds = make_audioset(
    "balanced_train_segments.csv",
    "wavs/",
    audio_layout="segments",  # use "full" when WAVs are full source media
)

# Keep dataframe-like filtering.
speech = pf.where(ds, ds.table["labels"].str.contains("/m/09x0r", regex=False))

# Row access materializes the implicitly sliced audio segment.
item_id = speech.table.index[0]
row = speech[item_id]

audio = row["audio"]       # numpy array, loaded lazily from the WAV source
labels = row["labels"]     # AudioSet label string
sample_rate = row["sample_rate"]

plot_spectrogram(speech, item_id)

# Training integration can use the dataset rows directly.
loader = make_torch_dataloader(speech, batch_size=16, shuffle=True)
batch = next(iter(loader))
```

Under the hood, the convenience loader is doing the structured work once:

```python
from examples.audioset import (
    SOURCE_AUDIO_ID_FIELD,
    make_audio_files,
    make_audioset_labels,
    merge_audio_labels,
)

labels = make_audioset_labels("balanced_train_segments.csv")
audio_files = make_audio_files(
    labels.table[SOURCE_AUDIO_ID_FIELD].drop_duplicates(),
    "wavs/",
)

ds = merge_audio_labels(labels, audio_files)
```

The label CSV and WAV directory are separate sources. They are parsed
separately, joined by audio id, merged into one dataset, then bound so segment
columns slice audio accessors. Users still get a dataframe-like object for
filtering and inspection, but row access knows how to retrieve the relevant
waveform segment.

This separation is the reusable-loader story in miniature. The audio source
maker can be reused with another label source. The label maker can be reused
with another audio representation. New combinations are assembled by dataset
operators, not by writing a new dataset class for every product of
`data_source x label_source`.

That is the core patchframe pattern:

1. Put source-specific complexity in `DataSource` and `make_*` operators.
2. Define schema, dimensions, and couplings once at dataset construction time.
3. Keep ordinary filtering, visualization, batching, and training code short.

## A useful mental model

Think of patchframe as a dataframe with a memory of what its columns mean.

It does not want to be a universal storage format, a full compute engine, or a
modality-specific toolkit. Instead, it is an infrastructure layer for building
dataset-specific tools:

- `make_audioset(...)` for AudioSet-style audio metadata and WAV files
- `make_satellite_patches(...)` for raster tiles and geospatial labels
- `make_video_clips(...)` for frame ranges and annotations
- `make_microscopy_dataset(...)` for image stores and region annotations
- `make_simulation_runs(...)` for parameter tables and array outputs

The core stays non-geometric and small. Domain-specific packages can provide
their own sources, dimensions, creation operators, visualization helpers, and
training adapters.

## Why this can be useful in practice

Patchframe is strongest when datasets evolve.

Early in a project, a CSV plus file paths may be enough. Later, labels are
corrected, clips are resegmented, model scores are joined back in, multiple
annotator sources are compared, embeddings are cached, training subsets are
sampled, and quality metrics are computed over slices.

Without dataset-level structure, those steps tend to become fragile conventions:
column names, path templates, slice math, and merge assumptions spread across
notebooks and scripts.

Patchframe gives those conventions names and places:

- fields for what columns are
- dimensions for how slices are interpreted
- couplings for how fields relate
- sources for where data came from
- operators for how datasets change

The payoff is not just nicer code. The payoff is fewer silent dataset mistakes.

## Current status

Patchframe is in early development. The core ideas are present, and the AudioSet
example shows the desired user experience, but APIs may still move as more
examples are extracted.

The current package is best evaluated as a dataset infrastructure experiment
with a concrete direction:

- dataframe-first workflows
- lazy multidimensional access
- explicit schema and couplings
- conservative composition semantics
- source-aware dataset construction
- extension-friendly domain examples

## Future directions

Potential development areas that fit the current design:

- More domain examples: image patches, video clips, tabular-plus-embeddings,
  scientific arrays, and remote-sensing tiles.
- Persistent dataset save/load through source IO adapters, without forcing
  external read-only sources into patchframe-managed storage.
- Dask-style extensions for large materialization and feature computation,
  returning explicit lazy graphs rather than hiding distributed execution inside
  normal in-memory operators.
- Benchmarked Polars paths for table-heavy operators such as concat, equality
  joins, and merge materialization, while preserving patchframe schema and index
  semantics.
- Async or concurrent row access for IO-heavy sources.
- Dimension-aware joins: temporal overlap, nearest-neighbor time joins,
  interval joins, scored retrieval joins, and optional geometry/spatial joins
  in extension packages.
- Labeling and active-learning integrations where rows, previews, model scores,
  annotator decisions, and source provenance stay connected.
- Dataset quality workflows that compute metrics over lazy slices and join the
  results back by stable row identity.

The common theme is the same as the AudioSet example: keep source-specific
knowledge close to the source, keep dataset semantics explicit, and keep the
ordinary user path close to the dataframe workflows people already know.
