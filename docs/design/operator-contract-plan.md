# Ontology and Dispatch — Implementation Plan

Status: implementation plan. Translates the structural pieces of
`aspect-transition.md` — the new transition vocabulary, the resolved-plan
collapse mechanism, and the unified aspect dispatch that consumes it — into
a phased, reviewable sequence of changes. Each phase ends green and can
merge independently.

Contract-suite work (`assert_operator_contract`, per-aspect checkers,
self-check regression net) is intentionally out of scope and tracked in a
follow-up plan; the structural foundation lands first because everything
else consumes it.

Implementation status: phases 1-5 have landed. The composition declaration
updates required to wire Phase 5 were pulled forward from Phase 6.

Cross-references:

- `aspect-transition.md` — the ontology and resolution table this plan
  implements.
- `field-identity.md` — `FieldIdentity` and `MergedField`, both load-bearing
  for compose-mode dispatch.
- `design-constraints.md` §2 — transition declaration is the contract;
  §11 — extension boundary.

## Goals

1. Update the typed transition vocabulary to match the design: add `compose`
   schema mode, add `custom` everywhere, remove `infer`, remove
   couplings `union`, add `homogeneous`, add `inherit(input=N)` for couplings
   and sources, collapse `IndexIdentityTransition.preserve` into `inherit`.
2. Land `resolve_derived_transitions` as the framework's collapse helper.
3. Replace the scattered `if t.X.mode == ...` checks in `_apply` and
   `_compose` with a single registry-driven dispatch helper that both flows
   call. Per-mode handlers are uniform-signature functions; the registry is
   the extension surface.
4. Re-declare every built-in operator under the new ontology and delete the
   `apply_couplings` / `apply_sources` overrides the dispatch now subsumes.
5. Small ergonomics pass on `MergedField` so the new dispatch's
   compose-mode handlers (and the future contract suite) read lineage
   through documented accessors instead of reaching into `parents` /
   `winning_parent()` from outside.

## Non-goals (this plan)

- `assert_operator_contract` and per-aspect checkers. Tracked in a follow-up
  plan; the registry-driven dispatch makes them straightforward, but
  shipping the structural change first keeps phases small.
- Property-based scenario generation.
- The "MergedField owns the per-collision resolve" tidy-up (one method
  returning the schema field + table column + coupling effects for a single
  MergedField). Useful once the dispatch is settled; not load-bearing.
- Cardinality enforcement at runtime — the declaration lands but no checker
  yet.

## Constraints

- No behavioral regressions. The existing test suite stays green at every
  phase boundary.
- The dispatch helper must not change what existing operators *do*; it only
  changes *who calls what*. Phase-by-phase tests verify the rewrite is
  pass-through.
- Public API stays stable except where the ontology explicitly renames a
  mode. `IndexIdentityTransition.preserve` gets a deprecation alias;
  `SchemaTransition.infer` gets one too. `CouplingsTransition.union` raises
  informatively (silent collapse would mask bugs — it meant two different
  things).

## The bold-dispatch shape

The structural choice driving the plan: `_apply` and `_compose` collapse
into a shared `compute_output_state(operator, states, args, kwargs) ->
DatasetState` helper. Aspect dispatch lives in a single per-aspect-per-mode
handler registry rather than scattered branches.

```python
# patchframe/ops/dispatch.py

Handler = Callable[
    [
        tuple[DatasetState, ...],  # input states
        TransitionPlan,            # post-collapse transition plan
        Schema,                    # output schema as built so far
        Operator,                  # for custom / construct callbacks
        tuple,                     # positional args
        dict,                      # kwargs
    ],
    Any,
]

_HANDLERS: dict[tuple[str, str], Handler] = {
    ("index_identity", "inherit"):  _identity_inherit,
    ("index_identity", "mint"):     _identity_mint,
    ("index_identity", "coalesce"): _identity_coalesce,
    ("index_identity", "custom"):   _call_operator_index_identity,
    ("table", "preserve"):  _table_preserve,
    ("table", "mutate"):    _call_operator_table,
    ("table", "construct"): _call_operator_table,
    ("couplings", "inherit"):     _couplings_inherit,
    ("couplings", "homogeneous"): _couplings_homogeneous,
    ("couplings", "clear"):       _couplings_clear,
    ("couplings", "derive"):      _couplings_derive,
    ("couplings", "construct"):   _call_operator_couplings,
    ("couplings", "custom"):      _call_operator_couplings,
    # sources: parallel set
    # schema: handled separately (must run before resolve_derived_transitions)
}

def compute_output_state(operator, states, args, kwargs) -> DatasetState:
    declared = operator.resolve_transitions(*states, *args, **kwargs)
    output_schema = _build_schema(operator, declared, states, args, kwargs)
    resolved = resolve_derived_transitions(
        declared,
        input_schemas=tuple(state.schema for state in states),
        output_schema=output_schema,
    )
    output_schema = _dispatch("index_identity", states, resolved,
                              output_schema, operator, args, kwargs)
    output_table  = _dispatch("table", states, resolved, output_schema,
                              operator, args, kwargs)
    output_couplings = _dispatch("couplings", states, resolved,
                                 output_schema, operator, args, kwargs)
    output_sources   = _dispatch("sources", states, resolved,
                                 output_schema, operator, args, kwargs)
    output_schema = resolve_merged_fields(output_schema)
    return DatasetState(
        schema=output_schema, table=output_table,
        couplings=output_couplings, sources=output_sources,
    )
```

Schema is the one aspect that can't go through the registry as-is: the
resolution table needs `output_schema` to collapse `derive` modes for the
other aspects. `_build_schema` is a small wrapper around `operator.
apply_schema` that respects the `preserve` short-circuit. Schema's
mode-specific assertions are a contract-suite concern; runtime dispatch
just needs the schema produced.

What this buys versus the conservative wiring:

- `_apply` and `_compose` shrink to thin family-specific wrappers (preserve
  source_manager / combine source managers; replace_state vs Dataset
  construction).
- New aspects or modes land by registering a handler — no edit to the
  framework's control flow.
- Extension authors can register operator-specific handlers if they ever
  need to override `derive` resolution for a custom field type.
- One execution path means one place to instrument for tracing,
  benchmarking, or cache-invalidation reasoning.

Cost: one indirection layer (registry lookup per aspect) plus a new module.
Manageable; the lookups are constant-time and the handlers are small.

## Phasing

Each phase ends green. Phases ship independently; later phases depend on
earlier ones.

### Phase 1 — Vocabulary updates

File: `patchframe/ops/transitions.py`. Pure additive/renaming changes. No
framework dispatch changes yet; operators continue declaring what they
declare today, but the new modes become available.

Changes:

- `SchemaMode`: add `compose`, add `custom`, remove `infer`.
- `CouplingsMode`: add `homogeneous`, add `custom`. Keep `derive`, `clear`,
  `construct`. Remove `union`.
- `SourcesMode`: add `derive`, `compose`, and `custom`. Keep `inherit`,
  `construct`, `clear`. Remove `union`.
- `IndexIdentityMode`: add `derive` and `custom`. Keep `inherit`, `mint`,
  `coalesce`. Remove `preserve` (deprecation alias).
- New factory classmethods to match: `SchemaTransition.compose()`,
  `SchemaTransition.custom()`, `CouplingsTransition.homogeneous()`,
  `CouplingsTransition.inherit(input=N)`, `SourcesTransition.derive()`,
  `SourcesTransition.compose()`, `SourcesTransition.inherit(input=N)`
  (already exists; keep),
  `IndexIdentityTransition.derive()`.
- **Defer** the `TransitionPlan` default-change to Phase 6. The doc-stated
  new defaults (`preserve` for schema/table, `derive` for couplings/sources/
  identity) would silently regress operators that rely on the current
  `table=mutate` default (`where`, `rename`, `drop`, `keep`, `consume`,
  `add_column`, `bind_dimensions`). Those operators get explicit
  declarations in Phase 6; the default change lands together.
- For Phase 1, the dataclass `default_factory` references on
  `TransitionPlan` switch from the now-deprecated classmethods
  (`SchemaTransition.infer`, `IndexIdentityTransition.preserve`) to the
  bare class constructors (`SchemaTransition`, `IndexIdentityTransition`)
  so bare `TransitionPlan()` construction doesn't emit deprecation
  warnings at module import. Class-level default modes are unchanged.
- Deprecation policy:
  - `SchemaTransition.infer()` → `DeprecationWarning`, returns
    `SchemaTransition.preserve()`. Note: this is a behavior-narrowing
    alias, fine because no built-in relies on the default and the warning
    surfaces the change for extension authors.
  - `IndexIdentityTransition.preserve(input=N)` → `DeprecationWarning`,
    returns `IndexIdentityTransition.inherit(input=N)`.
  - `CouplingsTransition.union()` → raises `ValueError` with migration
    guidance (compose-derive use case → `derive`; row-stack use case →
    `homogeneous`). No silent alias because the prior `union` meant two
    different things.
  - `SourcesTransition.union()` → raises `ValueError` pointing to
    `derive` (when sources follow schema lineage) or `compose` (when source
    records combine independently of schema lineage).
- `_MODES` literal frozensets updated; `__post_init__` validation updated.
- Update `patchframe/ops/__init__.py` and `patchframe/__init__.py` exports.

Three small operator-declaration updates are required so the `union()`
raises don't break module imports:

- `CompositionOperator.transitions` (`base.py`): `couplings=union()` /
  `sources=union()` → `derive()` / `derive()`.
- `merge.transitions`: same two replacements.
- `join.transitions`: `sources=union()` → `compose()`.

These are no-op for current `_compose` behavior — that flow calls
`apply_couplings` and `combine_sources` directly without dispatching on
the declared mode. The declarations become the real contract in Phase 6.

Tests (`tests/test_transitions.py`):

- New mode factories construct correctly.
- Removed modes raise (`CouplingsTransition(mode="union")` → `ValueError`).
- Deprecation aliases emit `DeprecationWarning` (`infer`, identity
  `preserve`).
- `CouplingsTransition.union()` / `SourcesTransition.union()` raise with
  migration messages.
- Bare `TransitionPlan()` emits no deprecation warnings.
- `pf.merge.transitions.couplings.mode == "derive"` (updated from `union`).

Exit criterion: existing test suite green; new vocabulary tests green; no
operator declarations changed yet.

### Phase 2 — `resolve_derived_transitions`

New module: `patchframe/ops/resolution.py`. Operator-independent pure
function over schemas — no operator instance required, no hooks called.

Signature and return type chosen to match the user's framing: plain
`TransitionPlan` in / plain `TransitionPlan` out, no new wrapper type.

```python
def resolve_derived_transitions(
    declared: TransitionPlan,
    *,
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> TransitionPlan
```

**Couplings is the centerpiece.** The function does not collapse
`couplings.derive` to a different mode — it stays `derive` — but it
populates three resolution-data fields on the returned
`CouplingsTransition` from input/output schema lineage:

- `rename_map: tuple[tuple[str, str], ...]` — `(input_name, output_name)`
  for non-MergedField output fields whose `FieldIdentity` matches an
  input field of a different name (rewrite lineage).
- `dropped: tuple[str, ...]` — input field names whose `FieldIdentity`
  is absent from the output and was not a winning compose-collision
  parent (narrow lineage).
- `superseded_per_input: tuple[tuple[int, tuple[str, ...]], ...]` — for
  compose, per-input losing-parent names walked from
  `MergedField.parents` + `winning_parent()`.

This makes couplings derive directly testable: given input/output
schemas, the function classifies the schema-lineage scenario, and tests
verify the classification without running an operator.

`CouplingsTransition` gains these three fields (default empty,
populated only by resolution). Future formalism (see open questions)
will revisit whether to fold this kind of resolution data into the
mode vocabulary itself.

**Sources** resolves more cleanly via mode-only collapse:

- `derive` under preserve / extend / narrow / rewrite / infer →
  `inherit(input=0)`.
- `derive` under `construct` → `construct`.
- `derive` under `compose` → stays `derive` (deduped-union handled by
  the compose dispatch handler later).
- `derive` under `custom` → raises.

**Index identity** also collapses to specific modes:

- `derive` under preserve / extend / narrow / infer → `inherit(input=0)`.
- `derive` under `construct` → `mint`.
- `derive` under `compose` → `coalesce`.
- `derive` under `rewrite` → `inherit(input=0)` if the rewrite preserves
  the primary `IndexField` (same `FieldIdentity`, still an
  `IndexField` in output); raises if the primary was demoted or removed
  (`set_index`-style — must declare `mint` explicitly).
- `derive` under `custom` → raises.

**Hard errors** — declared combinations that have no defined outcome:

- `schema=construct` + `couplings=derive` (no input lineage).
- `schema=custom` + any aspect `derive` (vocabulary cannot describe the
  effect; operator must declare).

Schema, table, and accessors aspects are passed through unchanged (no
`derive` mode for those).

Tests (`tests/test_resolution.py`) — 27 cases:

- Couplings derive yields the expected `rename_map` / `dropped` /
  `superseded_per_input` per schema mode (preserve, extend, narrow,
  rewrite single + multi-rename, compose collision, compose row-unify).
- Couplings derive non-derive modes pass through unchanged.
- Couplings derive under construct / custom raises.
- Sources derive resolves per the table (unary modes → inherit; compose
  → compose; construct → construct; custom raises).
- Identity derive resolves per the table (unary → inherit; rewrite
  preserving primary → inherit; rewrite demoting primary → raises;
  compose → coalesce; construct → mint; custom raises).
- Non-derive aspects pass through unchanged.
- Resolution is idempotent.

Exports `resolve_derived_transitions` from `patchframe.ops` and
`patchframe`.

Exit criterion: helper covered by tests; not yet wired into framework
flows.

### Phase 3 — Dispatch helper and registry

New module: `patchframe/ops/dispatch.py`. Built and tested in isolation
against synthetic operators; framework flows are not touched in this phase.

Components:

- `Handler` type alias as described in "The bold-dispatch shape" above.
- `_HANDLERS` registry covering every `(aspect, concrete_mode)` pair.
- `_dispatch(aspect, states, resolved, output_schema, operator, args,
  kwargs)` — looks up the handler, calls it.
- `_build_schema(operator, declared, states, args, kwargs)` — calls
  `operator.apply_schema(...)` unless declared schema mode is `preserve`,
  in which case returns `states[declared.schema.input].schema`.
- `compute_output_state(operator, states, args, kwargs) -> DatasetState`
  — the unified flow shown above.
- `register_aspect_handler(aspect, mode, handler)` — extension hook
  (mirrors `register_field_policy`).
- Concrete per-mode handlers:
  - **identity**: `_identity_inherit` (use
    `primary_index_identity(states[N])`), `_identity_mint` (call
    `mint_primary_index_identity(schema)`), `_identity_coalesce`
    (preserve when all inputs share one namespace, else mint),
    `_call_operator_index_identity` (delegate to
    `operator.apply_index_identity` for `custom`).
  - **table**: `_table_preserve` (returns `states[0].table` for the
    single-input case; the helper is unary-only — composition never
    preserves table), `_call_operator_table` (calls
    `operator.apply_table` for `mutate` / `construct`).
  - **couplings**: `_couplings_inherit` (return `states[N].couplings`),
    `_couplings_homogeneous` (verify all input `CouplingSet`s are
    structurally equal, return shared or raise), `_couplings_clear`
    (return empty `CouplingSet`), `_couplings_derive` (delegate to
    `derive_composed_couplings` when `schema=compose`, including
    collision-free composition schemas; otherwise delegate to
    `derive_unary`), `_call_operator_couplings` (operator-side for
    `construct` and `custom`).
  - **sources**: parallel set (`inherit`, `clear`, `compose` →
    deduped-union helper, `construct` / `custom` → operator-side).
- The `derive_unary` helper is the renamed version of
  `DatasetOperator._derive_couplings`, moved out of the class.

Tests (`tests/test_dispatch.py`):

- Each handler called in isolation with synthetic states produces the
  expected aspect-value.
- `_dispatch` routes correctly per `(aspect, mode)`.
- `compute_output_state` against a stub operator (with deterministic
  `apply_schema`/`apply_table`) returns the expected `DatasetState`.
- The full suite stays green (still nothing wired into `_apply` /
  `_compose`).

Exit criterion: dispatch helper covered; not yet wired.

### Phase 4 — Wire dispatch into `_apply` (landed)

File: `patchframe/ops/base.py`. `DatasetOperator._apply` collapses to:

```python
def _apply(self, dataset, *args, **kwargs):
    new_state = compute_output_state(
        self, (dataset.state,), args, kwargs,
    )
    result = dataset.replace_state(
        schema=new_state.schema,
        table=new_state.table,
        couplings=new_state.couplings,
        sources=new_state.sources,
    )
    self._validate_output(result)
    return result
```

`apply_index_identity` becomes a no-op default that handler functions
call into for the `custom` path only (default raises informatively). The
identity dispatch for `inherit` / `mint` / `coalesce` is registry-side.

`_resolve_couplings`, `_derive_couplings`, `new_couplings`: kept on
`DatasetOperator` as the operator-side knobs. The registry `derive`
handler reads `new_couplings` from the operator and appends to the
derived set, matching today's behavior.

Tests:

- The full existing suite stays green. Operator behavior is identical;
  only the call path changed.
- New `tests/test_apply_dispatch.py` exercises a few unary operators
  end-to-end and asserts the resolved-plan provenance is what the
  resolution table says.

Exit criterion: `_apply` runs through the dispatch helper; unary
operators behave identically.

### Phase 5 — Wire dispatch into `_compose` (landed)

File: `patchframe/ops/base.py`. `CompositionOperator._compose` collapses
to:

```python
def _compose(self, *datasets, **kwargs):
    states = tuple(d.state for d in datasets)
    new_state = compute_output_state(self, states, (), kwargs)
    source_manager = self.combine_source_managers(
        *datasets, composed_schema=new_state.schema,
    )
    return Dataset(state=new_state, source_manager=source_manager)
```

`apply_couplings` / `apply_sources` / `combine_sources` on
`CompositionOperator` are kept on the class but only invoked by the
registry handlers for `custom` / `construct` modes. The `derive` /
`inherit` / `homogeneous` / `clear` paths route through the registry.
Sources `derive` resolves to the deduped-union helper (the body of the
current `combine_sources`).

Tests:

- The full existing suite stays green — composition operators behave
  identically through the new dispatch.
- New `tests/test_compose_dispatch.py` covers the dispatch paths for
  each composition mode (inherit, homogeneous, clear, compose-derive).
- Verify `MergedField`s in the intermediate schema flow through every
  aspect handler before `resolve_merged_fields` collapses them.

Exit criterion: `_compose` runs through the dispatch helper; composition
operator boilerplate shrinks; existing tests stay green.

### Phase 6 — Re-declare built-in operators

Per the operator grounding table in `aspect-transition.md`. Mechanical
changes; every change is from "inherited default" or "wrong declaration"
to "explicit correct declaration."

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
| `make_from_dataframe` | `construct` | `construct` | `clear` | `construct` | `mint` | `unknown` |
| `window_expansion_plan` | `construct` | `construct` | `clear` | `inherit(input=0)` | `mint` | `expand` |
| `explode` | **`custom`** | `construct` | `inherit(input=0)` | `inherit(input=0)` | `inherit(input="plan")` | `expand` |
| `concat` (dispatcher) | `custom` | `custom` | `custom` | `custom` | `custom` | `unknown` |

Cleanup:

- Delete `apply_couplings` overrides on `concat_columns`, `merge`, `join`
  where the registry `derive` / `clear` handler now covers them.
- Delete `apply_sources` / `combine_sources` overrides where the
  registry `derive` (deduped-union) handler covers them. Keep
  `combine_source_managers` — that's a manager-level concern, not a
  sources-aspect concern.
- `concat_rows.preserve_row_couplings` is replaced by the registry's
  `homogeneous` handler. The helper in `_composition.py` can be deleted
  once nothing else calls it.
- `Operator`-class-level `transitions = TransitionPlan()` (the
  inherit-everything default) still works because every aspect resolves
  to a sensible `derive` outcome. `CompositionOperator.transitions`
  needs updating because `couplings=union` / `sources=union` are gone —
  set them to `derive`, `derive` (still composition-friendly).

Tests:

- All existing tests stay green.
- Update `tests/test_transitions.py::test_operator_transition_
  declarations` (if it exists) to the new expected values.

Exit criterion: every built-in operator declares the modes from the
grounding table; behavior is identical.

### Phase 7 — `MergedField` ergonomics

File: `patchframe/dataset/field_composition.py`,
`patchframe/ops/builtin/_composition.py`. Three small additions so the
new dispatch's compose handlers (and the future contract suite) read
lineage through documented accessors. Not load-bearing — could ship at
any phase boundary — but cleanest if landed after the dispatch is in
place and before the contract-suite work begins.

Changes:

- `MergedField.is_row_unify -> bool` property (returns `self.collision is
  None`). Replace `isinstance(..., MergedField) and field.collision is
  None` checks with this.
- `MergedField.expected_identity() -> FieldIdentity` — public accessor
  exposing `self.resolve_field().field_identity`. The future contract
  suite needs this to compute expected output identity sets.
- `MergedField.losing_parents() -> tuple[FieldParent, ...]` — moves the
  walk currently in `_composition.py:65-77` (`superseded_names_by_input`)
  onto the class. Returns an empty tuple for row unifications.
- Refactor `derive_composed_couplings` to read `MergedField.
  losing_parents()` instead of computing `superseded_names_by_input`
  inline. The helper still lives in `_composition.py` (or its renamed
  dispatch-helper home) because it walks all states and all
  MergedFields — that's an orchestration concern, not a per-collision
  concern.

What this *isn't*: MergedField does not take over the composition
schema walk, table assembly, or coupling pruning loop. Its ownership
stays exactly per-collision: one `MergedField` answers
`resolve_field()`, `resolve_column(parent_columns)`, and
`losing_parents()` about its own parents. The composition operator (or
the dispatch handler) owns the loop around them.

Tests:

- Update `tests/test_merged_field.py` to cover the new accessors.
- Existing composition tests stay green.

Exit criterion: external callers stop reaching into `collision` /
`parents` / `winning_parent()`; lineage queries go through the new
accessors.

## Migration concerns

- **`SchemaTransition.infer` removal.** No built-in declared `infer`
  explicitly today (it was the default); the redeclaration in Phase 6
  makes every built-in explicit. With the new default `preserve`,
  extension operators that relied on the `infer` default now get
  `preserve` semantics (no schema change). For operators that *do*
  modify the schema and relied on `infer`, the deprecation alias warns
  and routes to `preserve`, which short-circuits `apply_schema` — those
  authors must explicitly pick the right mode. Audit the test suite at
  Phase 6 to confirm no built-in regresses.
- **`CouplingsTransition.union` removal.** All current uses are via the
  `CompositionOperator` default. Phase 6 replaces them explicitly. No
  alias because the prior `union` meant compose-derive in some places
  and homogeneous in others.
- **`IndexIdentityTransition.preserve` rename.** Phase 1 alias maintains
  source compatibility for one release. Schedule removal in the release
  notes.
- **Composition operator `apply_couplings`/`apply_sources` deletions.**
  Audit per-operator during Phase 6. Delete only where the registry
  handler fully replaces them; keep where the operator still wants the
  hook for `construct` / `custom` paths.

## Risks

- **Dispatch helper hides errors.** A misregistered handler or wrong
  mode mapping could silently produce wrong aspect values. Mitigation:
  Phase 3 tests cover every `(aspect, mode)` pair in isolation; Phase 4
  / 5 verify the full suite stays green through the rewrite.
- **Custom operator `run` paths.** Operators such as `explode` or dispatchers
  such as `concat` may normalize through `OperatorCall` and still bypass the
  aspect dispatch helper for part of execution. The grounding-table
  declarations for those operators still bind contractually (the future
  contract suite verifies them). Worth noting that aspect dispatch helps only
  operators whose `run` path calls `compute_output_state`.
- **One-extra-layer overhead.** Registry lookup per aspect adds a
  constant cost. Negligible per call, but worth measuring once on the
  concat / merge / consume benchmarks at Phase 5 boundary to confirm
  no regression.
- **`derive_unary` extraction.** Moving `_derive_couplings` out of the
  class is a small risk surface (different `self` context, different
  warning stacklevel). Keep `_derive_couplings` as a thin shim on the
  class that forwards to the module function if any external code
  depends on it.

## Test coverage summary

After Phase 6:

- `tests/test_transitions.py` — vocabulary, new modes, removed modes,
  deprecation behavior, defaults.
- `tests/test_resolution.py` — every row of the resolution table.
- `tests/test_dispatch.py` — every handler in isolation; full
  `compute_output_state` against stub operators.
- `tests/test_apply_dispatch.py` — unary end-to-end through the new
  flow.
- `tests/test_compose_dispatch.py` — N-ary end-to-end through the new
  flow.
- `tests/test_field_identity.py`, `tests/test_index_identity.py`,
  `tests/test_merged_field.py`, all existing operator tests — stay
  green throughout.

After Phase 7:

- `tests/test_merged_field.py` — new accessors covered.

## Open questions to resolve during implementation

- **Resolved-plan provenance.** Add it when the contract suite needs to
  distinguish declared from derived guarantees. Keep it out of runtime
  dispatch until a concrete consumer exists.
- **Handler registration scope.** Global registry only, or per-operator
  override? Global is simpler and matches `register_field_policy`'s
  pattern. Per-operator may matter once extension authors want custom
  `derive` resolution for a custom field type — defer until a real use
  case appears.
- **Whether `_build_schema` belongs in `dispatch.py` or stays on
  `Operator`.** It calls `apply_schema`, which is operator-side. Argue
  for keeping it module-level (every aspect goes through the dispatch
  module; schema being special is local to that module). Decide during
  Phase 3.
- **`apply_index_identity` deprecation.** The current method on
  `DatasetOperator` is shadowed by the registry. Keep as a thin
  forwarder for the `custom` mode or delete entirely. Decide during
  Phase 4.
