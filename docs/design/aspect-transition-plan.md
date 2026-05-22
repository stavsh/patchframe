# AspectTransition Tightening — Implementation Plan

Status: internal implementation plan. Companion to
`aspect-transition-ontology.md` (the vocabulary/rationale) and
`design-constraints.md` (the forward constraints). This document records the
agreed plan for the first concrete implementation stage and what it
deliberately leaves for later.

**Stage A is now implemented.** Two scope cuts were taken mid-implementation
and the as-built state is recorded in `aspect-transition-ontology.md` (Stage A
implementation summary): (1) `infer` is a placeholder mode treated
conservatively — the `inference.py` module and real observe-and-resolve
inference are deferred to the FieldIdentity stage; (2) the declared-vs-actual
"two-transition flow" was dropped — `_apply` runs on the single resolved
transition. The "Step 2 — Schema inference" and two-transition passages below
are superseded by that summary.

## Rationale

AspectTransition declarations exist for four purposes. Every mode and
mechanism below earns its place by serving at least one:

1. **Reduce boilerplate** for handling dataset state across operators —
   aspects an operator does not change should pass through with no code.
2. **Stratify the ontology of complex operations** and let special `Field`
   behavior be encoded separately from the operator, so extensions can plug
   field-type behavior in without rewriting operators.
3. **Automatic operator-correctness tests** — given a declared (or observed)
   transition, generate tests that verify the operator's structural promise.
4. **Compile-time / pre-execution inspection** — e.g. cache invalidation
   decisions that read transition declarations.

Lazy/greedy duality is explicitly **not** a consumer. That dispatch is
type-based (input carries a `BundleField` → operator registers as a coupling;
otherwise executes) and uses the `.instance(...)` infrastructure for call-time
state. It does not need transition declarations.

Current problem: `AspectTransition("derive")` is an untyped catch-all. `_apply`
branches only on `preserve` vs not-`preserve`, so every other mode behaves
identically — the vocabulary is documentation with no operational effect.

## Refined design (converged decisions)

### Per-aspect typed vocabulary

Each aspect gets a typed transition class with a small set of typed variants,
replacing the single stringly-typed `AspectTransition`:

```
Schema:         preserve, extend, narrow, rewrite, construct, infer
Couplings:      derive (default), union, construct, clear
Table:          preserve, mutate, construct
IndexIdentity:  preserve, inherit, mint, coalesce
Sources:        inherit, union, construct, clear
Accessors:      preserve            (minimal for now; revisit later)
```

Typed variants carry structured details (e.g. `Rewrite` carries the field-name
mapping, `Narrow` carries the drop policy, `Inherit` carries the input index).
Variants are pattern-matchable so future consumers do not string-compare.

### `infer` mode and the two-transition flow

`infer` exists on schema only. It means: run `apply_schema`, then observe the
input vs output schema and resolve the actual mode post-hoc. Schema is metadata
and cheap to run, so the lost skip-hook optimization is negligible.

Each operator call therefore has two transition values:

- **Declared** — from `resolve_transitions(...)` (or the class-level
  `transitions`). Drives *pre-execution* routing: whether to skip `apply_table`,
  which identity path to take.
- **Actual** — the declared plan with any `infer` mode replaced by the observed
  result. Used by *post-execution* consumers: tests, cache hints, future
  composed-transition inference.

`infer` cannot detect renames — `a` removed + `b` added is indistinguishable
from `a`→`b` without lineage. Renaming is therefore a declared act: operators
that rename fields declare `schema=rewrite(mapping=...)`.

### `resolve_transitions`

`resolve_transitions(self, *states, **kwargs) -> TransitionPlan` on `Operator`.
Default returns the class-level `self.transitions`. Operators with
flag-dependent contracts (collision strategy, optional selectors) override it
to refine the conservative class-level declaration into the precise per-call
plan. At its simplest it is a switch on inputs/kwargs; sugar comes later.

### Couplings derived from schema

Couplings are a reference graph over schema fields, so coupling behavior is a
function of the schema transition. A framework-default `apply_couplings`
derives output couplings from input couplings + the schema transition:

- schema `preserve`/`extend` → keep all couplings.
- schema `rewrite` → rewrite coupling field-refs through the declared mapping.
- schema `narrow` → prune couplings whose fields were dropped, per policy.
- schema `construct` → cannot derive; operator implements or default `clear`.

There is no circularity: `apply_schema` runs first, so the schema transition is
known before couplings are computed.

Operators that *add* couplings override a small hook
`new_couplings(state, **kwargs) -> tuple[Coupling, ...]`; the framework
combines derived + new. The declared coupling vocabulary collapses to
`derive` / `union` / `construct` / `clear` (`preserve`, `rewrite_refs`,
`prune` are derived outcomes, not choices; `append` is `derive` + a non-empty
`new_couplings`).

### Sources from family defaults

Source behavior is determined by operator family and already has correct
defaults: `DatasetOperator` → `inherit`, `CompositionOperator` → `union`,
`CreationOperator` → `construct`. No builtin operator needs a custom
`apply_sources`. Sources stay a strict, intentional declaration set at the
family base. No `infer`.

### Field policies remain operator-driven

Field-type policies (today: `compose_rows`, `compose_column`,
`compose_collision_field`) are libraries the operator calls from inside
`apply_X`, with operator-supplied context. They are **not** dispatched by the
framework from a transition mode. This avoids the circular dependency
(inferring a mode requires running the hook; dispatching the hook would
require the mode) and matches what merge/concat already do.

Reactive *post*-field-policies (e.g. re-bind geometry CRS after a rename,
validate dtype after `mutate`, refresh an `ExtensionArray`) are a separate,
future mechanism that would dispatch on the *observed* transition. Not built in
this stage.

### Cardinality

`Cardinality` (`preserve` / `filter` / `expand` / `unknown`) is declared as a
`ClassVar` on `Operator`, adjacent to `TransitionPlan` — not inside the table
aspect, not in a broader capabilities umbrella. It is populated during operator
migration. Per-row-independence is **dropped**: its only motivation was lazy
dispatch, which is type-based.

### Three-tier extension model

- **Tier 1** — compose existing operators; transitions inferred from the chain.
  Future; not in this stage.
- **Tier 2** — declare a new operator with `TransitionPlan` + optional
  `resolve_transitions`. The path this plan implements.
- **Tier 3** — override `__call__` directly. Existing escape hatch.

## Step-by-step plan (Stage A)

### Step 1 — Typed transition vocabulary (`patchframe/ops/transitions.py`)

- Define per-aspect transition classes with typed variants and structured
  details. Each variant pattern-matchable.
- Update `TransitionPlan` to type each field with its aspect class; set
  worst-case-safe defaults (`schema=infer`, `couplings=derive`, `table=mutate`,
  `index_identity=preserve`, `sources=inherit`, `accessors=preserve`).
- Add a `TransitionPlan._with(...)` helper for terse refinement in
  `resolve_transitions`.
- Define the `Cardinality` enum.
- Done when: vocabulary imports cleanly, defaults constructed, no operator
  migrated yet.

### Step 2 — Schema inference (`patchframe/ops/inference.py`)

- `infer_schema_mode(input_schema, output_schema) -> SchemaTransition`: set
  comparison → `preserve` / `extend` / `narrow` / `construct`. Never returns
  `rewrite` (not inferable).
- Done when: inference covers the four detectable modes with tests.

### Step 3 — Operator base changes (`patchframe/ops/base.py`)

- Add `resolve_transitions(*states, **kwargs) -> TransitionPlan`; default
  returns `self.transitions`.
- Add `cardinality: ClassVar[Cardinality]` to `Operator`.
- Rewrite `DatasetOperator._apply` to the two-transition flow: resolve, run
  hooks per declared mode, resolve `infer` to actual after `apply_schema`.
- Add default `apply_couplings` deriving couplings from the schema transition;
  add the `new_couplings` hook.
- Update `apply_index_identity` to consume typed `IndexIdentityTransition`
  variants (`preserve`/`inherit`/`mint`/`coalesce`).
- Update `CompositionOperator._compose` for the typed vocabulary; composition
  keeps its existing field-composition logic.
- Done when: base classes use typed transitions; default coupling derivation
  works.

### Step 4 — Migrate builtin operators (`patchframe/ops/builtin/*`)

- Replace every `AspectTransition("derive")` with the typed variant per the
  ontology grounding table. Schema/couplings → `infer`/`derive` unless a
  precise mode is known and useful. Identity/sources → explicit intentional
  variant.
- Flag-dependent operators (`concat_columns`, `bind_dimensions`, `consume`,
  `add_column`) get a `resolve_transitions` switch.
- `rename`/`set_index` declare `schema=rewrite(mapping=...)`.
- bind_* operators replace full `apply_couplings` with `new_couplings`.
- Populate `cardinality` per operator.
- Done when: no `"derive"` strings remain; behavior unchanged.

### Step 5 — Tests

- Update tests that assert mode strings (e.g. `test_index_identity.py`) to
  assert typed variants.
- Add coverage for `resolve_transitions`, `infer_schema_mode`, and default
  coupling derivation.
- Full suite green (baseline: 188 passed, 1 skipped).

### Step 6 — Reconcile `aspect-transition-ontology.md`

- Update the ontology doc to match the refined design: `infer` mode, the
  two-transition flow, the collapsed coupling vocabulary, sources as
  family-default, field policies as operator-driven libraries.
- Done when: ontology doc and implementation agree.

## What Stage A solves for

- **Purpose 1**: a typical non-trivial operator writes only `apply_schema` +
  `apply_table` (+ optional one-line `new_couplings`). Couplings, sources,
  identity come from defaults.
- **Purpose 3 (partial)**: `infer_schema_mode` and the actual-transition flow
  give tests a way to observe and assert structural effects.
- **Purpose 4 (partial)**: the actual transition is computable post-execution
  for cache-hint consumers.
- Phases out `AspectTransition("derive")` entirely.
- Migration is mechanical and small (~5-10 lines per operator).

## What is deferred

- **Purpose 2 fully** — reactive post-field-policies (geometry CRS, dtype
  validation, `ExtensionArray` refresh) dispatched on the observed transition.
  The typed vocabulary makes them addable; they are not built here.
- **Mode enforcement in `_apply`** — validating that an operator declaring
  `extend` actually extends, `narrow` actually narrows, etc. Stage A trusts
  declarations; enforcement is a later stage.
- **Tier 1 composition** — composing operators and inferring the composed
  transition. Needs a mode-level composition algebra (documented later) or
  fixture-run inference.
- **Generated operator-contract tests** — the full Purpose 3 payoff.
- **Cardinality consumers** — declared in Stage A, but tests/caching that read
  it land later.
- **Accessors aspect** — kept minimal; revisit when accessor caching is real.
- **Storing the actual transition on `DatasetState`** — not done; inference
  functions are called on-demand by consumers.

## What is given up

- **Compile-time transition resolution.** With `infer` and
  `resolve_transitions`, the precise transition of a call is known only at run
  time. Static, pre-execution reasoning about an operator's exact contract is
  no longer possible for `infer`/flag-dependent operators. Accepted in exchange
  for the ergonomic win — implementations are the source of truth, declarations
  are the optimization and the conservative contract.

## Open questions

- **Composition algebra.** Tier 1 needs rules for how typed transitions compose
  (`narrow ∘ extend`, `rewrite ∘ rewrite`, etc.). Some are field-lineage
  dependent. Document before Tier 1; not needed for Stage A.
- **Narrow coupling policy default.** When `narrow` drops a field that a
  coupling references — fail, drop+warn, or consume-before-drop as the default?
  Resolve during Step 3.
- **`resolve_transitions` for `CompositionOperator`.** Composition is N-ary and
  already overrides `_compose`; confirm the hook signature works for both
  families or whether composition needs a variant.
