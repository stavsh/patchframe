# Aspect Transitions

Status: internal design rationale. Defines patchframe's operator transition
ontology — the typed vocabulary operators use to declare structural effects
across dataset aspects, the framework's collapse-and-dispatch mechanism, and
the per-mode structural assertions used by the operator contract suite.

Implemented in `patchframe/ops/transitions.py`,
`patchframe/ops/resolution.py`, and `patchframe/ops/dispatch.py`; consumed by
`patchframe/ops/base.py`. The operator contract suite described below is
planned follow-up work.

Cross-references:

- `field-identity.md` — `FieldIdentity` and `MergedField` mechanism.
- `design-constraints.md` §2 — transition declaration is the contract.
- `lazy-and-bundle.md` — cardinality and per-row independence as adjacent
  capabilities.

## Why transitions exist

`TransitionPlan` declarations serve four concrete purposes:

1. **Reduce boilerplate.** Aspects declared `preserve` route around hooks
   automatically; `derive` aspects collapse to concrete outcomes the framework
   applies without operator-side code.
2. **Stratify behavior, enable extension-friendly field policies.** Field
   types and coupling types can register policies keyed on transition modes;
   operators declare modes; the framework dispatches. The typed vocabulary
   keeps the surface stable for future extension-side registries.
3. **Mechanically checkable operator contracts.** Given a declared mode,
   `assert_operator_contract` verifies the structural promise without
   operator-specific test code. Extension authors verify their own operators
   against the same vocabulary.
4. **Compile-time inspection.** Cache invalidation and dispatch can read
   declared transitions before execution.

Lazy/greedy dispatch is **not** a consumer — that mechanism is type-based on
input (a `BundleField` triggers chunked execution), not capability-based.
Cardinality and per-row independence are operator capabilities adjacent to the
transition plan; they share the contract-test surface but not the dispatch
machinery.

## Core principles

- Modes are aspect-local. A schema mode describes field structure, not row
  movement or where data came from.
- Modes describe structural relations between input and output aspects, not
  implementation strategy.
- Field addition/removal is a schema concern; row filtering or expansion is a
  cardinality concern; they do not by themselves invalidate field references.
- Add a mode only when a current or near-term consumer needs it. When a
  behavior recurs across two operators, name it; when it doesn't, use
  `custom`.

## Structural vs identity equality

Throughout the ontology, "agree" / "match" / "equal" refer to **structural**
equality (name, dtype, concrete field type, declared parameters) — not
`FieldIdentity` or `IndexIdentity`. `Field.__eq__` and `CouplingSet.__eq__`
are identity-blind by design: identity tracks lineage for derivation, while
structural equality is what users observe and what composition operators
check.

Identity is used only where the ontology names it explicitly:

- `derive` rename detection (input vs output `FieldIdentity` comparison).
- `compose` `MergedField` lineage (parent identity propagation, fresh-on-
  divergence semantics for row unification).
- `index_identity` modes (`inherit(input=N)` / `coalesce` operate over
  `IndexIdentity`).
- `mint` and freshness checks (assert a new identity is not present in any
  input).

Where the ontology says "homogeneous" or "agree," substitute "structurally
equal." Where it says "preserve identity" or "mint," it means literal
`FieldIdentity` / `IndexIdentity` operations.

A practical consequence: concatenating two datasets with same-name fields
minted independently (different `FieldIdentity`s, same name and dtype) is a
valid row-stack. The unified output field receives a fresh `FieldIdentity`
because the input identities diverged, but the operation itself is not
rejected. Homogeneous coupling checks pass when the couplings are structurally
equal regardless of which side's field identities they refer to.

## Vocabulary

Each aspect has its own typed transition class with a small, validated mode
set and classmethod factories. Six aspects: schema, table, couplings, sources,
index_identity, accessors. Cardinality is an adjacent operator capability,
declared as a `ClassVar` on the operator class rather than inside
`TransitionPlan`.

### Schema (`SchemaTransition`)

Schema is always explicit. No `derive` mode.

- **`preserve`** — output schema equals the selected input schema. `_apply`
  skips `apply_schema`.
- **`extend`** — input fields survive unchanged (by `FieldIdentity`); new
  fields may be added with freshly minted identities.
- **`narrow`** — some input fields may be removed; survivors keep their
  `FieldIdentity`.
- **`rewrite`** — field identities survive but representation changes
  (rename, retype, role change). The rename map is derived structurally
  from `FieldIdentity` lineage; no explicit declaration required.
- **`compose`** — N-ary composition through `MergedField` parents. Same-name
  fields with shared `FieldIdentity` keep that identity; same-name fields
  with divergent identities are unified into an output field with a freshly
  minted identity. Collision resolution follows the declared
  `ColumnCollisionStrategy` and the operator's role (row-stack, column-add,
  key-coalesce).
- **`construct`** — no input lineage to derive from; output schema is built
  from nothing. The canonical case is `CreationOperator`: there is no input
  dataset, so the schema is whatever the operator's parameters produce.
  Plan operators that mint a fresh plan schema unrelated to any input also
  use `construct`.
- **`custom`** — input lineage exists but the operator's schema
  transformation is not one the structural vocabulary captures. Total
  escape hatch. The contract suite asserts nothing about schema and emits a
  warning that the aspect is unverified.

Default: `preserve`.

### Table (`TableTransition`)

Table is always explicit. No `derive` mode — table effects are not derivable
from schema alone.

- **`preserve`** — table is passed through unchanged. `_apply` skips
  `apply_table`.
- **`mutate`** — values, index, row count, order, or columns change.
- **`construct`** — table is newly built (composition assembles, planners
  emit).

Default: `preserve`.

There is no `custom` for table — table effects are observable as cardinality
plus column-set changes, and those are checked through the cardinality
declaration and the schema mode. An operator that wants to opt out entirely
would declare every other aspect `custom` instead.

### Couplings (`CouplingsTransition`)

- **`derive`** (default) — framework resolves couplings from the resolved
  schema mode via the resolution table below. Unary lineage uses
  `FieldIdentity`; composition lineage uses `MergedField` parents.
- **`inherit(input=N)`** — output couplings equal input N's couplings
  unchanged. Bare `inherit()` means `inherit(input=0)`.
- **`homogeneous`** — output couplings equal the inputs' couplings iff all
  input `CouplingSet`s are structurally equal; raise otherwise. Structural,
  identity-blind. The canonical use is `concat_rows` — row-stacking a
  coupling that applies to only some of the input rows is unsafe, so the
  operator requires all inputs to agree before propagating the coupling
  set.
- **`clear`** — output has no couplings.
- **`construct`** — no derivable lineage; operator builds couplings from
  scratch. Rare; used by operators that mint fresh couplings unrelated to
  any input.
- **`custom`** — lineage exists but the operator opts out. Suite warns.

The prior `union` mode is removed. Its uses split: composition operators
that did MergedField-aware pruning union (`concat_columns`, `merge`) become
`derive`; `concat_rows`'s strict-preserve becomes `homogeneous`.

### Sources (`SourcesTransition`)

- **`derive`** (default) — framework resolves from schema mode.
- **`inherit(input=N)`** — output sources = input N's sources.
- **`compose`** — deduped union of input sources. Used when source lineage
  composes independently of schema lineage.
- **`clear`** — output has no source records.
- **`construct`** — operator mints new source records (CreationOperator's
  natural mode).
- **`custom`** — opt out.

### Index identity (`IndexIdentityTransition`)

- **`derive`** (default) — framework resolves from schema mode.
- **`inherit(input=N)`** — output identity = input N's primary index
  identity. Bare `inherit()` means `inherit(input=0)`.
- **`mint`** — output rows enter a freshly minted identity namespace.
- **`coalesce`** — preserve when all inputs share one namespace, else mint.
- **`custom`** — opt out.

The prior `preserve(input=N)` mode collapses into `inherit(input=N)`. A
single name for a single operation.

### Accessors (`AccessorsTransition`)

Minimal — accessor caching has no real consumer yet.

- **`preserve`** (default) — pass through.
- **`mutate`** — invalidate.

### Cardinality

Adjacent to `TransitionPlan` on the operator (`cardinality: ClassVar`).
Not part of `TransitionPlan` because cardinality is an operator property,
not an input/output aspect relation.

- **`preserve`** — one output row per input row.
- **`filter`** — output rows are a subset of input rows.
- **`expand`** — one input row may yield several output rows.
- **`unknown`** — no local guarantee.

Default: `unknown`. Operators should narrow this where possible — lazy
dispatch and contract testing both consume it.

## Derive resolution

When `derive` is the declared mode for couplings, sources, or index_identity,
the framework collapses to a concrete outcome based on the resolved schema
mode. The resolution table:

| schema mode | couplings derive → | sources derive → | identity derive → |
|---|---|---|---|
| `preserve` | preserve couplings | `inherit(input=0)` | `inherit(input=0)` |
| `extend` | preserve + `new_couplings(...)` | `inherit(input=0)` | `inherit(input=0)` |
| `narrow` | prune dropped, identity-rename survivors | `inherit(input=0)` | `inherit(input=0)` |
| `rewrite` | rewrite refs via identity lineage | `inherit(input=0)` | `inherit(input=0)` |
| `compose` | compose-derive via `MergedField` | `compose` (deduped union by `source_id`) | `coalesce` |
| `construct` | undefined — operator must declare | `construct` (mint sources) | `mint` |
| `custom` | undefined — operator must declare | undefined — operator must declare | undefined — operator must declare |

Notes:

- **Rewrite vs index identity.** `rewrite` defaults identity to
  `inherit(input=0)`. `set_index` is the canonical exception that rewrites
  the `IndexField` itself; it explicitly declares `index_identity=mint`. The
  framework rejects `index_identity=derive` for operators that perform a
  rewrite affecting the primary `IndexField` (detected by comparing the
  input's `IndexField.field_identity` against the output schema and finding
  it no longer references an `IndexField`, or finding the primary
  `IndexField` came from a different input field's `FieldIdentity`).
- **Compose-derive specifics.** Compose-derive is the `MergedField`-aware
  pruning union: for each `MergedField` in the output, prune couplings
  whose referenced field name lost the collision (per
  `MergedField.winning_parent`); identity-rename survivors as needed;
  dedup-union the result.
- **Construct asymmetry.** Under schema=construct, sources and identity have
  natural defaults (mint fresh) so `derive` still collapses. Couplings has
  no natural default; the operator must declare `clear` / `construct` /
  `inherit` / `homogeneous` explicitly.
- **Custom propagates.** Under schema=custom, every other aspect must be
  explicit. `custom` schema means lineage is opaque to the framework, so
  no derive resolution is possible.

## The collapse mechanism

A single operator-independent helper:

```python
def resolve_derived_transitions(
    declared: TransitionPlan,
    *,
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> TransitionPlan
```

Returns a resolved `TransitionPlan`. Sources and index-identity `derive`
aspects collapse to concrete modes. Couplings stays `derive` but receives
lineage data (`rename_map`, `dropped`, `superseded_per_input`) computed from
the schemas.

The framework calls `resolve_derived_transitions` from `compute_output_state`
after `apply_schema` runs. The future contract suite should call the same
helper. Single code path; no duplicated lineage logic.

Resolved-plan provenance is deferred until the contract suite needs it.

## How the framework consumes transitions

### Defaults

```python
TransitionPlan(
    schema=SchemaTransition.preserve(),
    table=TableTransition.preserve(),
    couplings=CouplingsTransition.derive(),
    sources=SourcesTransition.derive(),
    index_identity=IndexIdentityTransition.derive(),
    accessors=AccessorsTransition.preserve(),
)
```

An operator that declares nothing gets pass-through identity for every aspect.
Useful baseline for trivial decorators or wrappers; rarely what a real
operator wants. The defaults are *safe-but-uninteresting*, not worst-case-
unsafe.

### `DatasetOperator._apply` flow

`_apply` delegates aspect computation to
`compute_output_state(self, (dataset.state,), args, kwargs)`, merges the four
returned aspects into `dataset.replace_state(...)`, then validates.

### `CompositionOperator._compose` flow

`_compose` delegates aspect computation to
`compute_output_state(self, states, (), kwargs)`, combines source managers,
and returns the assembled `Dataset`.

### Unified `compute_output_state` flow

1. Resolve the declared transition plan, including any call-time refinement.
2. Build the output schema, preserving the selected input schema when
   declared.
3. Call `resolve_derived_transitions(...)`.
4. Dispatch index identity, table, couplings, and sources through the
   per-aspect-per-mode registry.
5. Collapse any intermediate `MergedField`s before returning `DatasetState`.

This closes the gap from the prior `_compose` flow where transitions were
declared but not consumed — composition operators no longer hand-implement
inherit/homogeneous/clear/derive behavior in their `apply_couplings`. Custom
behavior remains operator-side via `construct` or `custom`.

## Mechanical contract testing

Each concrete mode has a structural assertion the framework will verify
mechanically against `(input_states, output_state)`. The planned contract suite
(`patchframe.testing.assert_operator_contract`) runs operators on author-
supplied scenarios and applies these checks.

### Per-aspect assertions

**Schema** (input is `state.schema` for the input selected by the mode's
`input` parameter, default 0; `compose` uses all inputs):

| Mode | Assertion |
|---|---|
| `preserve` | `out.schema == in.schema` (structural equality). |
| `extend` | Every input field's `FieldIdentity` appears in output; output may have additional fields with fresh identities (not present in input). |
| `narrow` | Every output field's `FieldIdentity` matches an input field's identity; no fresh identities introduced. |
| `rewrite` | Set equality of `FieldIdentity` sets between input and output; names/dtypes/types may differ. |
| `compose` | Output identities = composition of input identities through observed `MergedField` lineage (suite computes the expected composition via the same `MergedField.over` pipeline the operator uses). |
| `construct` | Output `FieldIdentity` set is disjoint from the union of input `FieldIdentity` sets. |
| `custom` | Warn; assert nothing. |

**Table:**

| Mode | Assertion |
|---|---|
| `preserve` | `out.table is in.table` (object identity — matches `_apply`'s skip). |
| `mutate` | Cardinality check applies (see below). |
| `construct` | Cardinality check applies. |

**Couplings:**

| Mode | Assertion |
|---|---|
| `derive` (resolved) | `out.couplings == derived_outcome(input, output_schema, new_couplings)` — suite re-derives independently and compares. |
| `inherit(input=N)` | `out.couplings == in[N].couplings`. |
| `homogeneous` | `all(in[i].couplings == in[0].couplings) → out.couplings == in[0].couplings`. The raise-on-disagreement branch is tested separately by the operator author. |
| `clear` | `len(out.couplings) == 0`. |
| `construct` | If operator declared `new_couplings`, assert non-empty. Otherwise warn. |
| `custom` | Warn; assert nothing. |

**Sources:**

| Mode | Assertion |
|---|---|
| `inherit(input=N)` | `out.sources == in[N].sources`. |
| `clear` | Empty. |
| `construct` | Non-empty; `source_id` set disjoint from union of input `source_id`s. |
| `custom` | Warn. |

**Index identity:**

| Mode | Assertion |
|---|---|
| `inherit(input=N)` | `primary_index_identity(out) == primary_index_identity(in[N])`. |
| `mint` | Output identity is not equal to any input's primary identity. |
| `coalesce` | If all inputs agree on primary identity, output preserves; else output is fresh (not equal to any input). |
| `custom` | Warn. |

**Cardinality:**

| Mode | Assertion |
|---|---|
| `preserve` | `len(out) == len(in[N])`. |
| `filter` | `len(out) <= len(in[N])`. |
| `expand` | `len(out) >= len(in[N])` for non-empty input. |
| `unknown` | No check. |

### Suite mechanics

- **Scenarios.** Authors supply one or more `OperatorScenario(inputs,
  kwargs)` covering representative call shapes. Live datasets, not
  factories — matches `assert_source_contract`'s convention. Property-based
  scenario generation is future work.
- **Aggregation.** All violations across all aspects and all scenarios
  collect into a single `AssertionError`. One report per `assert_operator_
  contract` call, with per-violation context (scenario name, aspect, mode,
  expected vs observed).
- **Warnings for `custom`.** Every `custom` aspect emits a one-line warning
  naming the operator and the unverified aspect. Visible opt-out beats
  silent escape.
- **Family dispatch.** `assert_operator_contract` dispatches internally on
  operator family. Unary `DatasetOperator`, N-ary `CompositionOperator`,
  `PlanOperator`. `CreationOperator` uses a sibling
  `assert_creation_contract` because it has no input dataset to compare
  against.

### Extension surface

Authors with custom field types or coupling types can register augmenting
checks:

```python
register_aspect_check(aspect="schema", mode="rewrite", checker=my_check)
```

Registered checkers run in addition to the framework defaults; all violations
aggregate. This is the analog of `register_field_policy(...)` for the
contract surface.

## Operator grounding

The full set of built-in operators under the new ontology:

| Operator | schema | table | couplings | sources | identity | cardinality |
|---|---|---|---|---|---|---|
| `where` | `preserve` | `mutate` | `derive` | `derive` | `derive` | `filter` |
| `rename` | `rewrite` | `mutate` | `derive` | `derive` | `derive` | `preserve` |
| `drop` / `keep` | `narrow` | `mutate` | `derive` | `derive` | `derive` | `preserve` |
| `add_column` | `extend` | `mutate` | `derive` (+`new_couplings`) | `derive` | `derive` | `preserve` |
| `set_index` | `rewrite` | `mutate` | `derive` | `derive` | **`mint`** | `preserve` |
| `bind_dimensions` | `extend` | `mutate` | `derive` (+`new_couplings`) | `derive` | `derive` | `preserve` |
| `bind_slice` | `preserve` | `preserve` | `derive` (+`new_couplings`) | `derive` | `derive` | `preserve` |
| `bind_materialize` | `preserve` | `preserve` | `derive` (+`new_couplings`) | `derive` | `derive` | `preserve` |
| `consume` | `preserve` | `mutate` | `derive` | `derive` | `derive` | `preserve` |
| `concat_rows` | `compose` | `construct` | **`homogeneous`** | `derive` | `derive` | `expand` |
| `concat_columns` | `compose` | `construct` | `derive` | `derive` | `derive` | `preserve` |
| `join` | `construct` | `construct` | `clear` | **`compose`** | **`mint`** | `unknown` |
| `merge` | `compose` | `construct` | `derive` | `derive` | **`inherit(input=2)`** | `unknown` |
| `make_from_dataframe` | `construct` | `construct` | `derive` (resolves to `clear`) | `derive` (resolves to `construct`) | `derive` (resolves to `mint`) | `unknown` |
| `window_expansion_plan` | `construct` | `construct` | `clear` | `inherit(input=0)` | `mint` | `expand` |
| `explode` | **`custom`** | `construct` | `inherit(input=0)` | `inherit(input=0)` | `inherit(input="plan")` | `expand` |
| `concat` (dispatcher) | `custom` | `custom` | `custom` | `custom` | `custom` | `unknown` |

Bolded entries are the explicit exceptions to derive — every other non-
construct/custom aspect collapses through the resolution table.

Operators that override `__call__` (`join`, `merge`, `concat_rows`,
`concat_columns`, `window_expansion_plan`, `explode`, `concat`) bypass the
framework dispatch above. Their declarations remain contractually binding —
the operator must honor them in its own implementation — and the contract
suite verifies that regardless of dispatch.

`explode` declares `schema=custom` honestly: its schema transformation
(inherit source fields, prepend plan index) is not one of the structural
modes. The other aspects use concrete modes, so most of explode's behavior is
still mechanically verified.

`concat` is the operator dispatcher — `custom` everywhere because it
delegates to `concat_rows` / `concat_columns`. The underlying operators get
tested directly.

## Open questions

- **Minimized mode vocabulary, richer contract attributes.** The current
  ontology uses many specific modes (preserve, extend, narrow, rewrite,
  compose, construct, custom, inherit, mint, coalesce, homogeneous,
  clear). A future formalism could collapse these to four primitive
  modes — `preserve`, `construct`, `mutate`, `derive` — and move the
  detailed transition-contract information into attributes on each
  aspect transition (akin to how Phase 2's
  `resolve_derived_transitions` populates `rename_map` / `dropped` /
  `superseded_per_input` on `CouplingsTransition` instead of inventing
  new modes per scenario). The result would be a more declarative,
  data-driven contract surface that the suite can read uniformly. Out
  of scope for the current ontology pass; revisit once the contract
  suite has shaken out which contract data is actually load-bearing.
- **Aspect-specific `compose` modes.** Sources now has one: `join` constructs
  a fresh plan schema while composing input source records. Add parallel
  aspect-local modes only when another concrete use case appears.
- **Coupling composition policy registry.** If coupling-specific modes
  proliferate (we add three more in the next year), the right structural
  move is a registry parallel to `FieldCompositionPolicy`. Premature now —
  current set is six modes for couplings, manageable.
- **`per_row_independence` capability declaration.** Adjacent to cardinality.
  Needed for Stage 1 lazy dispatch (`lazy-and-bundle.md`). Out of scope for
  this stage; the ontology accommodates it cleanly when the time comes.
- **Property-based scenario generation.** The contract suite v1 takes
  author-supplied scenarios. A future Hypothesis-style strategy generator
  for arbitrary schema/dataset shapes would expand coverage but requires
  coupling-aware generators; out of scope for v1.
