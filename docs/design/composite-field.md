# Composite Fields — Column Grouping, and the Composite Index

Status: design direction agreed 2026-06-22; **Stages A–D BUILT 2026-06-24**
(adtech finding 3.5 resolved for the forcing function — `audience_overfit` runs on
real composite keys, the `creative|segment` string hack gone; 653 passed). The
originating driver is finding 3.5 (composite-key `partition`/`reduce`), but the
direction is a *general* `CompositeField` (column grouping), of which the
composite index is one variant. Records the converged model and the alternatives
explored, so the choice is not re-litigated. Deferred (fail-loud, no current
consumer): composite `concat`/`merge` composition, composite `domain=`,
`null_keys="group"`, sub-structure operators, identity-as-structured-product
(§3, §6).

Cross-references:

- `adtech-findings.md` §3.5 — the (creative × segment) grain, today faked with a
  `creative|segment` string key.
- `partition-aggregate.md` — `partition`/`reduce` `by=`/`domain=`/null policy this
  extends; `reduce` delegates key dispatch to `partition`.
- `design-constraints.md` §1 (`IndexIdentity` shape-generic; row identity may be a
  tuple-of-namespaces), §4 (the bundle substrate — `CompositeField` is its
  un-boxed, column-axis sibling).
- `lazy-and-bundle.md` §3 — `BundleField` (the row-bundling field this mirrors).
- `fields.py` — `Field` / `IndexField` / `ForeignIndexField` / `BundleField`.

## 1. CompositeField — a column-grouping field

A `CompositeField` groups **N columns** under one logical schema field, described
by an **index-less sub-schema**. It is **cardinality-preserving** (it groups
columns, not rows) — the **un-boxed sibling of `BundleField`** across the
row/column axis:

| | bundles | realized as | cardinality |
|---|---|---|---|
| `BundleField` | rows (a sub-dataset per cell) | **one** column of boxed `Dataset`s | changing (rows nest) |
| `CompositeField` | columns (a sub-schema span) | **N** native columns (nothing boxed) | preserving (same rows) |

Because nothing is boxed, the table stays pandas-native — the decisive criterion
(no unnatural object-column, no repack after a group-by). The sub-schema is
*index-less*: it describes the spanned columns, not a standalone dataset.

**Native column names are namespaced** as `f"{composite_name}.{subfield_name}"`
(a `location` field over `{lat, lon}` → table columns `location.lat`,
`location.lon`). Namespacing is collision-free (two composites may share a
subfield name), discoverable (the prefix is the group), and still native
(`ds.table["location.lat"]` works).

**Encapsulated.** Only the `CompositeField` is a top-level schema field; the
sub-schema fields are reached *through* it. `validate_table` validates the N
dotted columns against the sub-schema; everything iterating `schema.fields` sees
one field. This is what makes it **additive** — code that does not know the type
is unaffected, and `validate_table` already tolerates table columns with no
top-level field, so the dotted columns pass through existing datasets unchanged.

## 1a. Composition, coupling, and the 1:1 field↔column assumption (probed)

**Governing principle: a `CompositeField` is atomic for all purposes** (user,
2026-06-22). The framework treats it as one indivisible schema entity — a
coupling may address the *whole* sub-schema only (never a sub-column), and
generic ops (`keep`/`drop`/`rename`/`concat`/`merge`) act on the composite as a
unit. Reaching *into* the sub-schema — renaming, dropping, or retyping a
component — is **not** generic-op behavior; it requires dedicated composite
operators (future work). Atomicity is what makes the substrate below *determined*
(not speculative): the correct behavior is fixed by the principle + the table
layout, not by a workload. "Build at the forcing function" gates *under-
determined* work (executor policies, user-facing ergonomics); internally-
determined mechanism with a conceivable consumer (e.g. 3D-point dimension
columns) does not need a current example.

Probing Stage A surfaced that the framework pervasively assumes **one schema
field == one table column** (`concat` by `field.name`; `drop`/`keep`/`rename`/
`validate` by name). A `CompositeField` is **1→N**, so it violated that
everywhere. The fix — **built** — is a centralized, polymorphic **field↔column
trio** every site routes through, with no `isinstance` in schema or operators
(the field owns its physical-table relationship):

- `table_columns()` — the table columns a field occupies (default `(name,)`;
  index → `()`; composite → its dotted columns).
- `rename_table_columns(new)` — the column renames implied by renaming the field
  (a composite re-prefixes its dotted columns; an index field → none, its axis
  renames separately). `Schema.table_column_renames(mapping)` aggregates it; `rename` calls that.
- `validate_in_table(table)` — validate the field against the table (index →
  against the index; composite → each dotted column vs the sub-schema).

With these, `validate_table` / `keep` / `drop` / `rename` treat a composite
atomically with no type-branching. The remaining fail-loud cases:

- **Composition (`concat`/`merge`) — fail-loud (scope cut, not a design gap).**
  Unguarded, `concat_rows` would map `location` to a non-existent column → a
  phantom NA column plus the real `location.lat`/`.lon` silently dropped. A
  registered `CompositeFieldCompositionPolicy` raises at `apply_schema` before any
  table work. Under atomicity the compat rule is *determined* (composites compose
  iff their sub-schemas match as whole units), so building it later is mechanical;
  it stays fail-loud only as a scope cut. `CompositeField` and the future
  `CompositeIndexField` register **distinct** policies — the registry resolves by
  MRO and the two compose differently; this is why they are separate types.
- **Coupling — atomic, not addressable.** Sub-column refs (`location.lat`) are
  rejected by `schema.has` (top-level only); the composite name (`location`) is
  rejected by `map_fields`. So a composite cannot be a coupling input in v1;
  whole-composite coupling access is future work.
- **Sub-structure ops** — renaming/dropping/retyping a *component* is not
  generic-op behavior (a dotted name is not a top-level field); it needs a
  dedicated composite operator (future).
- **Non-index guard.** `CompositeField` cannot be `primary`/the index (it groups
  columns); a composite *index* is the separate `CompositeIndexField` (§2).

**Operator audit + the schema invariant.** Ops that *add* or *target* a column by
name were audited (`assign` was the worst — it **silently** created a phantom
`location` column, or a 2nd field claiming `location.lat`). Two layers handle it:

- **A schema invariant — no two fields may claim the same table column** (the
  union of `table_columns()` is unique; `Schema.__post_init__`). This catches any
  field overlapping a composite's dotted span (the stray-`location.lat` case),
  systematically, for every operator.
- **Single-target ops** (`assign` / `set_index` / `partition` need one scalar
  column) ask the field its column count — `len(table_columns()) > 1` rejects a
  composite. This is a **field-owned capability**, queried not `isinstance`-d:
  capabilities are the field's responsibility (user, 2026-06-23). A *formal*
  capability ontology (`is_key` / `is_assignable`) is the eventual home but a
  deliberate diversion (§6). Specialized ops (`link`/`explode`/`dimension_join`/
  …) are not deeply audited — a composite reaching them is unlikely and fails
  loud (`KeyError`); the schema invariant backstops any that add fields.

## 2. CompositeIndexField — the index variant

The same mechanism pointed at the **index** instead of at columns: the sub-schema
fields *are* the pandas `MultiIndex` levels. It is a **subclass of `IndexField`**,
so `isinstance(., IndexField)` and `primary_index_field` still return exactly one
index field — the ~75 single-index call sites and every single-index dataset are
untouched; the `MultiIndex` is encapsulated in this one field. The table index is
a **native `MultiIndex`**, built once by `partition`/`reduce` via a native
group-by (no per-operation regrouping).

- **Row identity is the composite tuple — unique, always** (the one inviolable
  invariant, §4), enforced unchanged by `DatasetState` (it is `MultiIndex`-safe).
- Sub-fields may be `ForeignIndexField` (a foreign reference, e.g. `creative_id`
  → `creatives`), `IndexColumnField` (a secondary index), or a plain value field.
- Index level names: lean **plain** (`creative_id`, `segment_id`) rather than
  dotted — one composite index means no cross-group collision, and plain names are
  more `MultiIndex`-native (open; §6).

**Stage B LANDED 2026-06-23** (632 passed; `tests/test_composite_index_field.py`,
12): the type, its fail-loud composition policy, the trio, and `MultiIndex`
plumbing (`make_from_dataframe.ensure_index_names` + `validate_table` via
`validate_in_table`); identity minted; column ops (`where`/`keep`/`drop`/
`map_fields`) preserve the composite index; the single-level-index audit (full
suite, with `CompositeIndexField` live as an `IndexField` subclass) is clean. The
scope, mapped against Stage A's hardening — the variant is *column-less*
(`table_columns()` → `()`), so most column hardening is moot; what transferred was
the *pattern* plus the **single-level-index** audit:

- **Transfers (reimplement, index-adapted):** the trio — `table_columns()` → `()`,
  `validate_in_table` → validate the `MultiIndex` *levels* (the real work),
  `rename_table_columns` → `{}` (level rename = a sub-structure operator, deferred);
  and **a distinct `CompositeIndexFieldCompositionPolicy` (fail-loud v1) —
  CRITICAL**, because the policy registry's MRO otherwise falls back to
  `IndexFieldCompositionPolicy` (single-level) and silently mishandles a
  `MultiIndex` (the index analog of the concat corruption).
- **New (index-specific):** `make_from_dataframe` + the `validate_table` index path
  accept/validate a native `MultiIndex`; and a **single-level-index audit** of the
  23 `isinstance(IndexField)` sites. Most are safe by design (identity helpers
  return "the one index field" — why subclassing works; `partition` rejects it as a
  key; `not isinstance(IndexField)` filters exclude it). Check: `resolution.py`
  (identity propagation), `concat._table_for_output_schema` (materializes the index
  as a column — behind the fail-loud policy), `explode`/`link`/`make_plan`/
  `set_index` (unlikely in v1; fail loud).
- **Does NOT transfer (column-only, moot here):** atomic `keep`/`drop`/`rename`
  (index preserved by pandas), the schema claimed-columns invariant (claims no
  columns), the single-target guards (`len(table_columns())>1` — index is 0;
  partition-by-index already rejected), the `map_fields` composite guard, the
  non-index guard (inverted — it *is* the index).

**Stage C LANDED 2026-06-24** (653 passed; `tests/test_partition_composite.py`,
`test_reset_index.py`). `partition`/`reduce` accept `by=[...]` → a composite base
(**the levels are the source key fields reused directly**, so a
`ForeignIndexField` key stays a level reference — polymorphic, no `isinstance`).
`null_keys=` is `error` (default) | `drop`; `"group"` and composite `domain=` are
deferred (fail-loud). **`reset_index`** (the decompose / way *out* of a composite —
the inverse of `set_index`) and the **`set_index` composite guard** are both
polymorphic via a field-owned **index↔column conversion**: `IndexField.to_data_fields()`
(index → data column(s); a single index → one `IndexColumnField` that keeps its
identity; a composite → its level fields) and `IndexField.level_names()`. The
`isinstance`-based demote in `set_index` is gone. **The rollup loop is closed and
tested**: composite `reduce` → `reset_index` (a `ForeignIndexField` level survives)
→ roll up by a level, re-aligning by the level's identity — the Stage D pattern,
end to end.

**Stage D LANDED 2026-06-24** (the forcing function). `audience_overfit`
(`examples/adtech_analysis.py`) is rewritten off the `creative|segment` string key
onto `reduce(by=[creative_id, segment_id])` → `reset_index` → `partition(by=
creative_id)` for the rollup. Output identical to the string-key version (same
flagged overfit creative + peak segment); the synthetic-key helpers
(`_cell_creative`/`_cell_segment`/`_count`) are deleted. Finding 3.5's composite
*grain* is resolved; the "nullable index / unattributed row" sub-part stays
deferred (`null_keys="group"`) — the example inner-joins, so it never arises.

## 3. Identity — OPEN (corner cases first)

Lean: the composite `IndexIdentity` is the **structured product** of the
sub-fields' foreign references when *all* components are foreign — so two
independently-built (creative × segment) datasets recognize a shared row-identity
namespace, and identity-aligned `concat`/align works at the composite grain;
**mint** when any component is a plain value (no namespace to compose).

**Not committed** (user, 2026-06-22) — corner cases to resolve first: mixed
foreign/value components; the same namespace in two levels; whether level *order*
is significant to identity; serialization + equality of a product identity; null
components vs identity. v1 may simply **mint** and add the product later — minting
is safe (it never falsely claims alignment), so this does not block the build.

## 4. Uniqueness and null (settled)

- **Uniqueness on the full index tuple is inviolable, at every stage** — the one
  invariant we do not give up. Enforced unchanged by `DatasetState.__post_init__`
  (`MultiIndex`-safe).
- **"Nullable index = a single NaN row"** falls *out of* uniqueness: pandas treats
  repeated nulls as non-unique (verified for float/object/string), so a nullable
  single index holds at most one null label by construction. "Nullable" is only
  "the dtype permits the one unattributed row."
- `partition`/`reduce` gain `null_keys=`: `"error"` (default, unchanged) | `"drop"`
  | `"group"` (the unattributed bucket as a null-labeled group).
- **Composite + null is deferred** (fail-loud): on a `MultiIndex`, uniqueness
  permits a null *component* across rows with differing tuples, so "single NaN row"
  is exact only for a single index / the all-null tuple. The forcing function
  (`audience_overfit`) inner-joins, dropping unattributed rows before the composite
  partition, so it never arises. Decide composite-null semantics at the workload.

## 5. Alternatives explored (do not re-litigate)

- **N top-level `IndexField`s** (one per level) — rejected: a `MultiIndex` level
  is non-unique, so a per-level *row*-identity is meaningless; it conflates row
  identity with foreign reference.
- **A thin descriptor** (lightweight `(name, dtype, ref)` specs, not Fields) —
  superseded: real Fields via the sub-schema make per-level types, identities,
  references, and validation first-class at no extra runtime cost (the columns and
  levels stay native).
- **A boxed "bundle of indices"** — rejected: an object-column of boxed index
  structures is unnatural in pandas and forces a repack after every group-by.
  `CompositeField` is the *un-boxed* form — the sub-schema is a schema-level
  pointer, not a per-row payload (and a field-holding-a-schema would otherwise just
  be a `BundleField`, the wrong tool here).

## 6. Open questions

- **Identity product-vs-mint** (§3) — corner cases first; v1 may mint.
- **Index level naming** — plain vs dotted; and the composite *index field's* name
  on a `partition`/`reduce` output (auto-derived, or a new parameter like `into`).
- **Projection / rollup** — projecting a composite to one component promotes that
  level to a real single index/foreign field at its grain (the recovery path for
  per-component foreign-key use; `audience_overfit`'s second step). Define the
  mechanism.
- **Composite `domain=`** (product totality) and **`MultiIndex` through
  `concat` / `merge` / `set_index`** — single-index-only, fail-loud, until a
  workload forces each.
- **Nested composites** — out of scope until a second consumer; v1 sub-schemas
  hold leaf fields only.
- **Landed (Stage A substrate, `tests/test_composite_field.py`):** the
  `table_columns` / `rename_table_columns` / `validate_in_table` trio (the field
  owns its physical-table relationship); `validate_table` / `keep` / `drop` /
  `rename` route through it (composite atomic, no `isinstance`); composition
  fail-loud via a distinct policy; coupling-opaque + non-index guards.
- **`concat`/`merge` composite composition** — fail-loud now (scope cut, not a
  design gap). Determined by atomicity (compose iff sub-schemas match as whole
  units); **row-stacking** same-sub-schema composites is the first to build.
- **Sub-structure operators** — renaming/dropping/retyping a *component* needs a
  dedicated composite operator (not generic-op behavior).
- **Formal field-capability ontology** — capabilities (`is_key` / `is_assignable`
  / spans-single-column / …) are the field's responsibility; today they're
  inferred from `table_columns()` (e.g. `len > 1` ⇒ not a single-column target).
  A formal ontology is the proper home but a **deliberate diversion** (user,
  2026-06-23), deferred until enough call sites justify it.
