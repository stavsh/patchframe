# Operator Authoring

Status: **reference** — *how* operators and couplings work and how to author them,
the load-bearing mechanics only. For *why* a choice was made, follow the
`docs/design/` links; the peer surfaces are `docs/field-authoring.md` (field
types) and CLAUDE.md "Source Authoring" (sources). Read `patchframe/ops/base.py`
and the named exemplar for your shape — this page is the map, not a substitute.

## Execution model (the metaclass)

Operators are classes (`OperatorMeta`).

- **`MyOp(args)` executes** — builds a temporary default instance and calls it.
- **`MyOp.instance(**params)` configures** — a *non-executing* instance carrying
  bound params. A configured operator passed *as a value* (e.g. a reduction in
  `reduce(aggs=...)`) must be `.instance(...)` or a sugar over it (`Sum.on("col")
  == Sum.instance(column="col")`) — **never `MyOp(...)`**, which would run.
- `x = Parameter(default=...)` class attrs are instance config (`copy`, a column),
  bound via `.instance(...)`, read via `bound_params()` / `resolve_param(name, arg)`.
  Not per-call operands.

## Operand-type dispatch (the one law)

The operand *type* selects the arm (`lazy-and-bundle.md`, `lazy-duality-plan.md`):

- **`Dataset` operand → eager**: runs now, returns a `Dataset`.
- **`FieldHandle` operand → lazy**: records, returns a handle. *Handles always
  select the lazy arm* — there is no eager handle resolution, anywhere.
- A *dual* op opts in by declaring `returns = FieldReturn`/`SelectionReturn`. Given
  a handle it routes (`_dispatch_lazy`):
  - **`coupling_able()`** (`schema ∈ {preserve, extend}` ∧ `cardinality PRESERVE` ∧
    `per_row_independent INDEPENDENT`) → **same-level coupling**: records via
    `new_couplings`, returns a handle (e.g. `map_fields`).
  - else → **bundle carrier**: deferred as an `ApplyOperator` on a `BundleField`.
  - **The arm is not inferable from cardinality alone** — `coupling_able()` gates on
    schema mode *and* cardinality *and* per-row-independence, and is overridable.
    `set_index` (cardinality `PRESERVE`, but `schema=rewrite` + global uniqueness
    [`per_row_independent=DEPENDENT`] + `index_identity=mint`) is *not* coupling-able
    and bundle-lifts; `ReducingOperator` (`FieldInput(BundleField)` + `REDUCE`)
    overrides `coupling_able() → True` and stays same-level. Read the gate.
- Terminals (`consume`/`collect`/`extract`/`flatten`) and creation ops are the only
  non-lazy handle consumers.

## The bundle / lift model (the lift precedes the op)

- The **lift** (`partition`, `bundle`) happens *before* the operator. An op applied
  to a **`BundleField` field handle** (the fibers) records an *in-level* coupling on
  the carrier, per fiber — so a per-fiber op declares `FieldInput(field_type=
  BundleField)` + `returns = FieldReturn`; the carrier already exists.
- A **coupling is a deferred operator application**, run by the engine at
  `consume`/`collect`: `MapCoupling` (a fn over column values), `ApplyOperator` (an
  operator over `BundleField` cells → a Dataset cell), `ReduceCoupling` (a reducing
  op over a fiber → a scalar). Carrying the *operator* (typed) lets the engine
  introspect it for future fusion; an opaque fn does not.

## Declaring an operator

Class attrs the metaclass collects into a `signature`:

- Operands (accept handles): `DatasetInput`, `FieldInput(field_type=…, output=…)`,
  `SelectionInput`. **`FieldInput` strictly means "this slot accepts a handle".**
- `ParamInput` — per-call data, *not* an operand (`where.predicate`, `partition.by`).
  Field-*naming* args on bundle-arm ops are `ParamInput`, never `FieldInput`.
- `FieldOutput(field_type=…)` — the caller-named produced field (`out=`).
- `returns = FieldReturn() | SelectionReturn() | DatasetReturn()` — the eager↔lazy
  seam. `DatasetReturn` = always a Dataset (eager-only ops, terminals, creation).

Capabilities are ClassVars + a `TransitionPlan`: `cardinality`,
`per_row_independent`, and `transitions = TransitionPlan(schema=, table=,
couplings=, sources=, index_identity=)`. `derive` modes collapse from the schema
mode (`aspect-transition.md`). Declarations are **contractually binding** and
**declared, never detected** — the op must honor them; nothing inspects bytecode.

## Authoring shapes — copy the exemplar

| shape | copy | mechanism |
|---|---|---|
| coupling-able add (per-row, schema `extend`) | `map_fields` | `apply_schema` adds the field, `apply_table` inits null, `new_couplings` returns the coupling; the lazy arm is free |
| anonymous table escape | `pipe` | inline `table -> table` fn plus per-call `TransitionPlan`; use only for one-off transforms and promote recurring shapes to named operators |
| composition wrapper (delegates to operators) | `partition` / `match` / `implode` / `reduce` | custom `normalize_call` + `run` composing existing operators |
| reducing op (fiber → scalar) | `reduce`'s `ReducingOperator` | `FieldInput(BundleField)` + column `Parameter` + `FieldReturn`; implement `apply(series)`; `coupling_able() → True`; `new_couplings` → a typed reduce-coupling |
| creation (no input dataset) | `make_from_dataframe` / a `make_*` source maker | `build` (+ `make_source` / `generate_source_info` when registering a source) |

`DatasetOperator` hooks (`apply_schema`/`apply_table`/`new_couplings`) run through
`compute_output_state`; ops with bespoke shape (`partition`, `concat`) override
`run` and normalize first.

## Authoring a coupling

A frozen `Coupling` subclass (`dataset/couplings.py`): `input_fields()`,
`output_field()`, `compute(state) -> Series` (bulk), `apply_row(row, state)`
(per-row). Name every column reference `FieldRef` so `rename`/`drop` rewrite it.
Carry the deferred call in a `CallSpec` and call `warn_if_unpicklable` at record
time — a lambda or script-local won't pickle, which blocks worker/persist
(design-constraints §7).

## Verifying

The transition declaration *is* the contract. `assert_source_contract` /
`assert_predicate_contract` verify sources/predicates today;
`assert_operator_contract` (planned) verifies operators against their declared
modes. Eligibility is always declaration, never detection.

## Pointers

- `patchframe/ops/base.py` — the model (metaclass, dispatch, `coupling_able`,
  `_run_eager`, `_dispatch_lazy`).
- `docs/design/lazy-and-bundle.md`, `lazy-duality-plan.md` — the dual arm, the
  bundle substrate, the routing (the *why*).
- `docs/design/aspect-transition.md` — the transition vocabulary + contracts.
- `docs/design/design-constraints.md` — invariants any change must respect.
