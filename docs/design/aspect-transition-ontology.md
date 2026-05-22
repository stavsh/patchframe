# Aspect Transition Ontology

Status: internal design rationale, reconciled with the Stage A implementation.
Not a public API spec. The transition vocabulary is implemented in
`patchframe/ops/transitions.py`; `aspect-transition-plan.md` records the Stage A
scope and what is deferred.

## Stage A implementation summary

Shipped:

- Per-aspect typed transition classes (`SchemaTransition`, `TableTransition`,
  `CouplingsTransition`, `SourcesTransition`, `IndexIdentityTransition`,
  `AccessorsTransition`), each with a small validated mode set and classmethod
  factories. `AspectTransition("derive")` is removed.
- `TransitionPlan` with worst-case-safe defaults: `schema=infer`,
  `table=mutate`, `couplings=derive`, `sources=inherit`,
  `index_identity=preserve`, `accessors=preserve`.
- A `resolve_transitions(*args, **kwargs) -> TransitionPlan` operator hook —
  default returns the class-level plan; `rename` overrides it to inject the
  rename mapping.
- `Cardinality` declared as an operator `ClassVar`.
- Default coupling derivation in `DatasetOperator`: couplings are computed from
  the schema transition (see Coupling Modes). Operators that add couplings
  override `new_couplings`.

Deferred:

- `infer` is a placeholder mode — it does not perform real observe-and-resolve
  inference; it is treated conservatively. Real schema inference arrives with
  the FieldIdentity stage (`field-identity.md`).
- Field-type policy dispatch stays operator-driven (libraries called from
  inside `apply_*`), not framework-dispatched from a transition mode.

## Purpose

Patchframe operators change more than table values. They can change schema
fields, coupling references, source provenance, row identity, and future
accessor caches. `TransitionPlan` exists so those effects are explicit and
eventually mechanically testable.

The transition ontology should support concrete upstream uses:

- Operator dispatch and defaults: decide which aspects can be passed through.
- Cache invalidation: decide which aspect-level caches remain valid.
- Coupling and `FieldRef` safety: decide whether references survive, need
  rewrite, or may become invalid.
- Operator contract tests: verify that implementations match declared
  structural effects.

It should not be a decorative taxonomy. A transition mode should exist only
when some current or plausible consumer can use it.

## Core Principles

- Modes are aspect-local. Schema modes describe field structure, not table
  movement or where fields came from.
- Modes describe structural relations between input aspects and output aspects,
  not implementation strategy.
- The table aspect should stay coarse until a concrete consumer needs more
  detail.
- The schema aspect needs more precision because field changes cascade into
  couplings, source mappings, `FieldRef` validity, and user-facing APIs.
- `AspectTransition("derive")` has been removed. Schema's "changed but
  uncharacterized" case is the explicit `infer` mode; couplings are computed
  from the schema transition via the `couplings=derive` mode (an unrelated
  meaning of the word).
- Field addition/removal is a schema concern. Row filtering or row expansion is
  a table/cardinality concern and does not by itself invalidate field refs.

## Base Aspect Modes

The implementation uses a separate typed transition class per aspect, each with
its own validated mode set (see the per-aspect sections below). The generic
modes here are the conceptual basis; not every aspect carries every one.

Most aspects can start with three generic modes:

- `preserve(input=0)`: the output aspect is the selected input aspect, unchanged.
- `mutate(input=0 | inputs=...)`: the output aspect is derived from input
  aspect(s), but changed. Existing caches for that aspect should be invalidated.
- `construct`: the output aspect is newly built. No preservation relation is
  assumed unless explicit lineage is declared.

`mutate` is intentionally different from `derive`: it is a real contract that
the aspect changes and must be recomputed or invalidated. It is just not a
fine-grained contract.

The modes below refine this base vocabulary only where there is a clear
structural consumer.

## Schema Modes

Schema modes should be organized around field-reference survivability.

### `preserve`

The output schema is identical to the selected input schema.

Implications:

- Existing `FieldRef`s remain valid.
- Couplings can be preserved.
- Field-level caches remain structurally valid.

Examples: `where`, `bind_slice`, `explode` preserving the source schema.

### `extend`

All existing fields from a selected input schema survive unchanged, and new
fields may be added.

Implications:

- Existing `FieldRef`s remain valid.
- Existing couplings can be preserved.
- New fields may introduce new couplings, but do not invalidate old ones.

Examples: `add_column` when adding a new field, `bind_dimensions` when creating
a new slice field, simple column composition with no collisions.

### `narrow`

Some fields may be removed. Surviving fields retain their identity unless a
separate rewrite is declared.

Implications:

- `FieldRef`s to removed fields become invalid.
- Couplings that reference removed fields must not be silently preserved.
- The operator must choose an explicit policy: fail, warn and drop invalid
  couplings, or consume coupling-dependent values before removal when that
  semantics is available.

Examples: `keep`, `drop`, future projection operators.

### `rewrite`

Existing field identities survive, but their representation changes. This
covers renaming, retyping, nullable/property changes, logical-role changes, and
reclassification such as moving a table-backed field into the primary index.

Implications:

- `FieldRef` identity should survive.
- Couplings should be rewritten through the field identity/name mapping.
- Source metadata that names fields may need the same rewrite.

Examples: `rename`, `set_index`, `consume` if materialization changes field
properties, `concat_rows` when compatible fields are widened.

Today many references are still name-based, so `rewrite` may initially be
implemented through explicit name mappings. A stronger `FieldRef` identity model
would make this mode more mechanically checkable.

### `construct`

The output schema is newly assembled. No field-reference preservation is assumed
by default.

Implications:

- Couplings should not be preserved unless the operator declares explicit
  lineage or constructs new couplings.
- Source metadata cannot assume field-name continuity.

Examples: `join`, `window_expansion_plan`, some future stack/unstack operators.

### `infer`

The operator changes the schema but does not characterize how. A placeholder
mode: in Stage A it is treated conservatively (couplings fall to the
retain-survivors path). Real inference — observing the input and output schemas
to resolve the precise mode — is deferred to the FieldIdentity stage, where
field lineage makes rename detection reliable. `infer` is the default `schema`
mode for an operator that declares nothing.

## Table Modes

The table transition should stay intentionally coarse for now:

- `preserve(input=0)`: exact table is preserved.
- `mutate(input=0 | inputs=...)`: table values, index, row count, order, or
  columns change.
- `construct`: table is newly built.

Finer categories such as relabel, stack, align, lookup, gather, and reshape are
not useful enough yet. They are not mutually exclusive, they overlap with schema
concerns, and no current consumer needs them.

If cache invalidation later needs field-scoped table effects, add optional
details such as changed field names. Do not promote operator-shaped table modes
without a concrete consumer.

## Cardinality Is Adjacent, Not Table Ontology

Lazy execution and future bundle dispatch may need a cardinality contract:

- `preserve`: one output row per input row.
- `filter`: output rows are a subset.
- `expand`: input rows can produce multiple output rows.
- `unknown`: no local cardinality guarantee.

This is useful, but it should be an operator capability or table detail consumed
by lazy dispatch, not the primary table aspect ontology. It answers "can this
operator run chunk-locally?", not "what table aspect can be preserved?"

## Coupling Modes

Couplings are a reference graph over schema fields, so coupling behavior is
largely a function of the schema transition. The implemented declared modes
collapse to four:

- `derive`: the framework computes the output couplings from the schema
  transition (the default). `preserve`, `rewrite_refs`, and `prune` are
  *derived outcomes* of this mode, not separate declarations.
- `union`: coupling sets from multiple inputs are combined (composition).
- `construct`: the operator builds the coupling set itself (`apply_couplings`).
- `clear`: the output intentionally has no couplings.

Under `derive`, given the schema transition:

- `schema` preserve / extend -> all couplings are kept.
- `schema` rewrite with a name mapping -> coupling field refs are rewritten
  through the mapping.
- `schema` narrow / infer / construct / rewrite-without-mapping -> couplings
  whose referenced fields all survive in the output schema are retained; the
  rest are dropped with a warning.

`append` is not a declared mode. An operator that adds couplings declares
`derive` and overrides `new_couplings`; the framework appends the returned
couplings to the derived set, skipping duplicates.

## Sources And Metadata

Sources can remain simple:

- `inherit(input=0)`: use one input's source records.
- `union(inputs=...)`: combine source records.
- `construct`: creation operator or source-producing operator.
- `clear`: intentionally no source records.

Metadata should remain advisory. It can have `preserve`, `mutate`, `construct`,
or `drop` behavior, but it should not be required for semantic correctness.
Operators should validate structural requirements from schema and table.

## Index Identity

Index identity is already a dedicated aspect because row identity has different
rules from table values:

- `preserve(input=0)`: output rows keep the selected input identity namespace.
- `inherit(input=n)`: output identity is selected from a non-primary input, such
  as a plan dataset.
- `mint`: output rows enter a new identity namespace.
- `coalesce(inputs=..., otherwise="mint")`: preserve only if compatible,
  otherwise mint.

The earlier descriptive names `row_stack` and `align_rows` were collapsed into
`coalesce`: both `concat_rows` and `concat_columns` declare
`index_identity=coalesce`, which preserves the identity when all inputs share
one namespace and mints otherwise.

## Operator Grounding

As implemented in Stage A:

| Operator | Schema | Table | Couplings | Sources | Index Identity |
| --- | --- | --- | --- | --- | --- |
| `where` | `preserve` | `mutate` | `derive` | `inherit` | `preserve` |
| `rename` | `rewrite` (mapping via `resolve_transitions`) | `mutate` | `derive` | `inherit` | `preserve` |
| `drop` / `keep` | `narrow` | `mutate` | `derive` | `inherit` | `preserve` |
| `add_column` | `extend` | `mutate` | `derive` + `new_couplings` | `inherit` | `preserve` |
| `set_index` | `rewrite` | `mutate` | `derive` | `inherit` | `mint` |
| `bind_dimensions` | `extend` | `mutate` | `derive` + `new_couplings` | `inherit` | `preserve` |
| `bind_slice` | `preserve` | `preserve` | `derive` + `new_couplings` | `inherit` | `preserve` |
| `bind_materialize` | `preserve` | `preserve` | `derive` + `new_couplings` | `inherit` | `preserve` |
| `consume` | `preserve` | `mutate` | `derive` | `inherit` | `preserve` |
| `concat_rows` | `construct` | `construct` | `union` | `union` | `coalesce` |
| `concat_columns` | `construct` | `construct` | `union` | `union` | `coalesce` |
| `join` | `construct` | `construct` | `clear` | `union` | `mint` |
| `merge` | `construct` | `construct` | `union` | `union` | `inherit(input=2)` |
| `window_expansion_plan` | `construct` | `construct` | `clear` | `inherit` | `mint` |
| `explode` | `construct` | `construct` | `construct` | `inherit` | `inherit(input="plan")` |

`concat_rows` and `concat_columns` carry no explicit declaration — they inherit
the `CompositionOperator` default, which is exactly their row above. `explode`
and `window_expansion_plan` override `__call__`, so their declarations are
descriptive rather than consumed by aspect dispatch.

Future examples:

- `assign_at`: schema is `preserve` when updating existing values, `extend`
  when adding fields, and `rewrite` when replacing field definitions. Table is
  `mutate`.
- Column stack/unstack: schema is likely `construct` unless explicit field
  lineage is declared. Table is `mutate`. Couplings should be pruned or rebuilt.

## Input- And Flag-Dependent Contracts

Many useful operators have contracts that depend on input schemas, field names,
or flags:

- `add_column` can extend or rewrite.
- `bind_dimensions` can create a slice field or replace an existing one.
- `consume` can preserve schema or rewrite field properties.
- `concat_columns` depends on collisions and collision policy.
- `assign_at` may update values, add fields, or replace fields.
- `explode` may remain schema-preserving today, but a future
  `keep_source_index=True` flag would make it schema-extending.

The current `TransitionPlan` is class-level and static, so it cannot precisely
represent these cases without becoming misleading.

Viable approaches:

1. **Static conservative contract.** Declare the broadest true mode, usually
   `mutate`, and avoid parameter-dependent precision.
2. **Resolved transition contract.** Add an optional
   `resolve_transitions(*states, **kwargs) -> TransitionPlan` hook that runs
   after parameters are normalized and before output validation.
3. **Post-hoc effect report.** Compare input and output aspects after execution
   and report actual effects for tests and cache invalidation.
4. **Split operators.** Expose separate operators for distinct contracts. This
   improves declarations but can harm ergonomics.

Near-term recommendation:

- Keep class-level transitions conservative.
- Replace `derive` with explicit coarse modes (`mutate` or `construct`) where
  possible.
- Add fine-grained schema modes only when they are true for all calls, or when a
  future resolved-transition hook can compute them from actual inputs.
- Treat input/flag-dependent precision as an open design problem until there is
  a concrete consumer that needs it.

## Mechanical Testing Direction

Initial generated tests should be modest:

- `preserve`: assert the output aspect equals the selected input aspect.
- `construct`: assert the operator did not accidentally pass through an input
  aspect by default, then validate normal invariants.
- `mutate`: validate normal invariants and invalidate caches; do not assert
  finer structure.
- `schema=extend`: assert input fields survive unchanged and additions are
  allowed.
- `schema=narrow`: assert removed-field coupling policy is explicit.
- `schema=rewrite`: assert field lineage or name mapping exists for rewritten
  refs.

This keeps the ontology useful without turning transition declarations into a
theorem prover.
