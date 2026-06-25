# Field Authoring

Status: **reference** — *what* a `Field` is and the surface a new field type
implements, the load-bearing mechanics only. Peer to `docs/operator-authoring.md`
(operators/couplings) and CLAUDE.md "Source Authoring" (sources). For *why* a
choice was made, follow the `docs/design/` links; the worked example of the
trickiest shape (a 1→N field) is `docs/design/composite-field.md`. Read
`patchframe/dataset/fields.py` and the named exemplar for your shape — this page
is the map.

## The dataclass

A field is a **frozen, slotted dataclass** subclassing `Field`
(`fields.py`). To author one:

- Subclass `Field` with `@dataclass(frozen=True, slots=True)` and set the
  `logical_type: ClassVar[str]` tag (some code dispatches on `== "index"`; keep
  it distinct).
- Adding a *required* attribute after `Field`'s defaulted ones needs a
  keyword-only break: `_: KW_ONLY` then the attribute (see `DimensionField.dimension`,
  `CompositeField.sub_schema`). `slots=True` only slots the new attributes.
- `__post_init__` must call the parent **explicitly** — `Field.__post_init__(self)`,
  not `super()` (unreliable under `slots=True`) — then validate your invariants.
  Mutate frozen state with `object.__setattr__`.
- `dtype` accepts a builtin/numpy/pandas dtype and is stored as the pandas
  **nullable** equivalent (`to_nullable_dtype`); `None` disables dtype validation.
  Object-cell fields (`DataField`) keep `object` cells and use `dtype` only to
  *describe* the materialized payload.

## The field↔table-column relationship (the one law)

A field knows how it maps to the physical table. Operators **delegate to the
field; they never `isinstance` it** for column handling — this is what lets a
field be 1→1, 1→N, or 0-columns without every operator learning the new type
(`composite-field.md` §1a). The four methods:

- `table_columns() -> tuple[str, ...]` — the columns the field occupies. Default
  `(self.name,)`. Override for **0** (an index occupies the table index, not a
  column → `()`) or **N** (a `CompositeField` → its dotted `{name}.{sub}` columns).
- `validate_in_table(table)` — validate the field against the table. Default
  `self.validate_column(table[self.name])`; an index validates the index, a
  composite validates each of its columns. `Schema.validate_table` calls this.
- `validate_column(series)` — the single-column dtype check. Override to **relax**
  (object-cell fields `pass`) or **reject** (a multi-column field raises — single-
  series validation cannot apply).
- `rename_table_columns(new_name) -> dict[str, str]` — the column renames implied
  by renaming the field. Default `{name: new_name}`; index `{}` (its axis renames
  separately); composite re-prefixes.

If you skip these for a 1→1 scalar column, the defaults are correct. Override only
to change arity (index, composite) or validation (object cells).

## Identity, and schema rules

- `field_identity` (`FieldIdentity`) is the field's **lineage** token: minted in
  `__post_init__`, preserved by `dataclasses.replace`, excluded from structural
  equality (`compare=False`). Coupling derivation compares by it
  (`_derive_couplings`), so a renamed/retyped field that keeps its identity keeps
  its couplings. Don't mint a fresh one when transforming an existing field.
- A `Schema` enforces: **unique field names**; **≤1 `primary` field per concrete
  type** (`IndexField` is always `primary`); and **no two fields claim the same
  table column** (the union of `table_columns()` is unique — what stops a stray
  `location.lat` field from colliding with a composite's span).
- `metadata` is advisory and must hold no executable logic (design-constraints §12).

## Composition policy (concat/merge)

A field type **registers** how it composes, via
`register_field_policy(MyField, MyFieldCompositionPolicy())` at module load
(`field_composition.py`). `field_policy_for` resolves by **MRO**, so a type with no
policy falls back to its nearest registered base — usually the `Field` base policy
(matches concrete type + dtype). Two consequences:

- A field that should **not** compose yet registers a **fail-loud** policy (raise
  in `compose_rows`/`compose_column`/`compose_key`) — otherwise it silently
  inherits a base policy that mishandles it (the `CompositeField` /
  `CompositeIndexField` case; the latter would otherwise fall through to the
  single-level `IndexFieldCompositionPolicy`).
- **Distinct types register distinct policies.** If two field types must compose
  differently, make them separate types (not a subclass) so MRO picks each its own
  — the reason `CompositeIndexField` is a sibling of `CompositeField`, not a child.

## Row-exit

`exit_value(value)` converts a cell at the row-access exit boundary (`ds[item_id]`
returns plain Python). Default is identity; container fields override
(`BundleField` exports its fiber as a list of records). For a field type you do
**not** own, `register_field_exit(type, fn)` registers a conversion resolved by an
MRO walk that takes precedence over the method.

## Index and container fields

- **`IndexField`** (the row identity): `primary=True`, `nullable=False`, carries an
  `identity: IndexIdentity` (the row namespace, minted by `DatasetState` if
  absent). It owns the **index↔column conversion**, field-side:
  `level_names()` (single → `(name,)`), `to_data_fields()` (how the index demotes
  to data columns — used by `reset_index`/`set_index`, *not* an operator
  `isinstance`), and `ensure_index_names(table)` (names the axis).
- **`IndexColumnField`** / **`ForeignIndexField`** — table columns of index *labels*
  (a secondary or foreign reference); `ForeignIndexField` requires `index_identity`
  and exposes `target_identity`. Non-unique columns, not the row identity.
- **Container fields hold a sub-structure.** `BundleField` holds a `Dataset` per
  cell (one object column). `CompositeField` holds an index-less **`sub_schema`**
  describing N **native** columns (dotted) — a *field-of-fields realized as native
  columns, not boxed*; `CompositeIndexField` is the same over `MultiIndex` levels.
  Composites are **atomic for all purposes** (`composite-field.md`): generic ops
  treat them as a unit; reaching into the sub-structure needs a dedicated operator.

## Authoring shapes — copy the exemplar

| shape | copy | key surface |
|---|---|---|
| scalar/value column | `ValueField` | defaults (1 column, dtype check) |
| object-cell column (lazy/opaque) | `DataField` / `DimensionedSliceField` | `validate_column` = `pass` |
| per-row container of datasets | `BundleField` | `exit_value` → records; `validate_column` `pass` |
| **column group (1→N)** | `CompositeField` | `table_columns`→dotted, `validate_in_table`, index-less `sub_schema`, **fail-loud composition policy** |
| index (row identity) | `IndexField` | `identity`, `level_names`, `to_data_fields`, `ensure_index_names`, validate the index |
| **composite index** | `CompositeIndexField` | subclass `IndexField`, `sub_schema`=levels, native `MultiIndex`, **distinct** composition policy |
| index reference column | `IndexColumnField` / `ForeignIndexField` | `index_identity` / `target_identity` |

Export a new public field from `patchframe/dataset/__init__.py`.

## Capabilities are field-owned

When an operator needs to know what a field *can* do ("is this a single assignable
column?", "how many index levels?"), it **asks the field** — `len(table_columns())`,
`level_names()` — rather than `isinstance`-ing the type. A formal capability
ontology (`is_key`/`is_assignable`/…) is a deliberate future item, not built;
today capabilities are read off the structural methods above.

## Verifying

There is no `assert_field_contract` yet (a planned peer to
`assert_source_contract` / the planned `assert_operator_contract`). Until then the
checks are `Schema.validate_table` (the trio + the invariants), the registered
composition policy, and the test suite. A new field type should be exercised
through `make_from_dataframe` + the operators it must survive (`keep`/`drop`/
`rename`/`concat`) and the fail-loud paths.

## Pointers

- `patchframe/dataset/fields.py` — the hierarchy and the per-field methods.
- `patchframe/dataset/field_composition.py` — the policy registry + base policies.
- `patchframe/dataset/schema.py` — `validate_table` and the schema invariants.
- `patchframe/dataset/identity.py` — `FieldIdentity` / `IndexIdentity` and helpers.
- `docs/design/composite-field.md` — the worked example of a 1→N field and the
  index variant (the trickiest shape).
- `docs/operator-authoring.md`, CLAUDE.md "Source Authoring" — the peer surfaces.
