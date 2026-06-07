# Lazy ↔ Eager Duality — Implementation Plan

Status: implementation plan. Turns the design line of `lazy-and-bundle.md` into a
phased, reviewable sequence. Each phase ends green and can merge independently.
The eager wide-record `BundleField` mechanism (phase 0 below) has already landed.

Cross-references:

- `lazy-and-bundle.md` — the substrate rationale; operand-type dispatch; the
  three-way routing; the staged-commitment posture.
- `aspect-transition.md` — `cardinality` and the transition vocabulary;
  `per_row_independence` is the adjacent capability this plan adds.
- `design-constraints.md` §3 (coupling = deferred operator application), §4
  (no parallel `LazyDataset`; call-site dispatch), §7 (pickle-friendliness),
  §9 (authoring surface).

## The spine

Everything below serves one law:

> **Every transform operator is a coupling-producer.** A `Dataset` operand →
> produce *and* apply now (eager, returns `Dataset`). A handle operand →
> produce only (lazy, returns a handle); `consume`/`collect` applies later. The
> produced coupling lands **same-level** if the op passes the 3-part test,
> otherwise on a **`BundleField` carrier** — in-level if the operand is already
> a bundle handle, else lift.

One producer model, one executor (`consume`/`collect`), and the Bundle is simply
where the couplings that cannot be same-level live. This is `lazy-and-bundle.md`
§6 made concrete.

## The converged model

- **Deferred operand = a `FieldHandle` to a `BundleField` column.** Not a new
  dataset-scoped handle. Scope is carried by the operand (`lazy-and-bundle.md`
  §10): `op(bundle.field("col"))` → in-level / per-fiber; `op(bundle)` → base.
  The user bundles first; operators never auto-lift on a bundle they were handed.
- **Context propagation follows the arm.** A lazy operation always *propagates*
  the `DatasetContext`: the chain threads one context forward, so the result
  handle shares it with the operands. An eager operation *forks*: it returns a
  new `Dataset` facade with its own fresh context. The explicit `with ctx:`
  cursor is the opt-in that additionally advances an eager op's ambient context.
  `Dataset.field()`/`fields()` therefore bind to a cached context that threads
  forward (never re-pinned to the snapshot).
- **The deferred op is an `ApplyOperator` coupling over `BundleField` columns.**
  It reads input cells (each a `Dataset`), runs the eager operator per row
  (= per fiber), and writes the result cell. `ApplyOperator` already exists and
  its `compute` already loops over rows, so the N-fiber case is supported; the
  one-row "wide record" is just N=1.
- **The coupling-able gate** decides same-level vs bundle. A same-level lazy op
  records a *coupling*, and couplings are the add/fill subset (design-constraints
  §3), so the gate is stricter than the §4 "3-part test":

  ```
  coupling-able = schema ∈ {preserve, extend}   # add / fill, not rewrite/narrow
               ∧ cardinality == preserve
               ∧ per_row_independent             # (no-mint is subsumed)
  ```

  It derives from existing declarations — no new capability. Coupling-able ops
  (`bind_*`, `add_column`, `assign`) go same-level; everything else needs a
  `BundleField` cell. This refines §4's "same-level" label: `rename`/`drop`/
  `keep` pass cardinality + per-row-independence but rewrite/narrow the schema,
  so they are **not** coupling-able and route to the bundle arm, like `where`.

  | op capability | operands | routing |
  |---|---|---|
  | coupling-able | field handle(s) | same-level coupling — no bundle |
  | needs-bundle | already-**bundle** `FieldHandle`(s) | **in-level**: add `ApplyOperator` on the carrier (the 99%) |
  | needs-bundle | ≥1 **non-bundle** operand | **lift**: mint a `BundleField` carrier, then in-level apply (the 1%) |
  | — | rule not airtight | **`custom`** escape hatch |

  **Creation/plan ops** (`make_from_dataframe`, `make_plan`) are lazy-exempt:
  their lazy form is *deferred creation* (a `DatasetAccessor`/`DatasetSource`
  cell), the workload-gated tier. The duality is universal over transforms;
  creation is the eager entry point.

- **The terminal** (`consume`/`collect`) resolves any pending deferred form —
  same-level couplings, bundle-couplings, and `join`'s plan/apply — uniformly,
  via `CouplingEngine`'s existing topo-sort. Chained in-level ops accrete a flat
  coupling graph on one carrier; the engine orders them. No new executor.

## Fork decisions (provisional; revisit if a workload pushes back)

- **`bind_*` stay declare-only.** They are coupling-able, so they never need a
  bundle; they are the coupling-authoring layer *beneath* the duality, not
  instances of the lift/in-level machinery. Whether to rename `bind_slice` →
  `slice` (eager-applies / handle-defers) is a **naming** question deferred
  until the verb convention is exercised by the bundle ops first.
- **`ApplyOperator` carries a serializable call-spec** (phase 4), not today's
  loose `operator + params`. The persisted spec is operator + normalized
  args/kwargs + field bindings + `variant`; the live `OperatorCall`
  (datasets/states/contexts/effects) stays runtime-only (§7).

## Phasing

### Phase 0 — eager wide-record Bundle (landed)

`BundleField` (eager `Dataset` cell); `ApplyOperator(Coupling)`;
`patchframe/ops/bundle.py` `build_apply_bundle` + `collect`. Tests in
`tests/test_bundle.py`. `collect(deferred_merge) == eager merge` verified.

### Phase 1 — `per_row_independent` capability

`PerRowIndependence` enum in `transitions.py` (`INDEPENDENT` / `DEPENDENT` /
`UNKNOWN`, mirroring `Cardinality`); `per_row_independent` `ClassVar` on
`Operator` (default `UNKNOWN`). Declare on builtins per the `lazy-and-bundle.md`
§4 inventory. The only missing input to the 3-part test; purely additive.
Exit: declared on all transform builtins; declaration test green; suite green.

### Phase 2 — entry bridge + selection

`Dataset.field()` / `Dataset.fields([...])` (today only on `DatasetContext`),
returning context-bound `FieldHandle`s; a *selection* type for multi-field
operands (`lazy-and-bundle.md` §1). Without this there is no ergonomic way to
get the handles the lazy arm consumes.

### Phase 3 — `bundle` / `flatten` / `extract` + unify the terminal (landed)

The flat<->bundle morphisms are **operators**, not loose functions — they pick
up transition declarations, the lifecycle, FieldHandle handling, and context
propagation like every other transform:

- `bundle(left=…, right=…)` — `CompositionOperator` (kwargs-as-cells via a custom
  `normalize_call`; positional auto-names `cell_0…`). Constructs the carrier;
  `schema`/`table=construct`, `couplings=clear`, `identity=mint`; eager sibling
  (no cursor advance).
- `extract(b.field("left"))` / `extract(b, "left")` — `DatasetOperator`,
  `field_handle_inputs=("field",)`, custom `run` returning the cell. The **first
  real `FieldHandle`→`BundleField` operand** — gets eager/lazy dispatch and
  propagation for free.
- `flatten(b)` — `DatasetOperator`, custom `run` → `concat_rows` of every cell.

The terminal is internal `_collect` (`consume`, then `extract` **only when the
field is a `BundleField`** — extraction is bundle-specific; a regular field
returns the materialized container); the user-facing exit bridge is the nullary
**`FieldHandle.collect()`** method that dispatches to it (the §1 "handles don't
execute" carve-out — `collect` is not an operator).
`consume` is now idempotent on an already-materialized field (no-coupling +
column-exists → return unchanged instead of raise). `build_apply_bundle` stays an
internal fused-leaf helper. **Out of scope:** `partition`/`chunk`/`section` (tall
substrate).

### Phase 4 — `OperatorSignature` + typed operands + carried call-spec

`OperatorSignature` declares, per operand slot, the accepted operand kinds:

- a **dataset slot** accepts a `Dataset` (eager) **or** a `FieldHandle`→
  `BundleField` (lazy / per-fiber — the audit's gap #2, the trigger for
  whole-dataset ops like `where`/`merge`);
- a **field slot** accepts a typed `FieldHandle` (`bind_slice.slice_field :
  DimensionedSliceField`) or a name;
- an **output slot** — `FieldOutput`, the dual of `FieldInput` (option 2): the
  caller supplies the *name* of the produced field (`merge(…, out="merged")`),
  the op produces a `BundleField` of that name, and the lazy arm returns a handle
  to it — the chaining point, **inherent to every lifting op** (it's what makes a
  lazy chain expressible). In-place ops (`bind_slice`) declare no output slot:
  their output is an input, resolved from the recorded coupling via `returns`;
- plus `returns` (the eager-vs-lazy seam for the non-`FieldOutput` cases),
  same/cross-dataset validation, ambient-context behavior, and cursor
  advancement.

**The return rule (consistent + honest).** A coupling-producing lazy op's output
*is* a field (regular or bundle) or a field selection, so its lazy arm returns a
handle to it — chainable, and honest because it points at the recorded op's
actual `output_field`(s), never a statically-named slot:

- `FieldReturn` — dual: eager → `Dataset`; lazy → a `FieldHandle` to the
  coupling output (single field). Most transforms.
- `SelectionReturn` — dual: eager → `Dataset`; lazy → a `Selection` (multi-output
  ops, e.g. `assign`).
- `DatasetReturn` — always a `Dataset`. Only for ops with **no** coupling output:
  eager-only ops, the `bundle` **entry** constructor, and the `extract`/
  `flatten`/`collect` **exit** bridges (which leave the bundle world). A handle
  operand still selects the lazy arm; what that arm *returns* is the signature's
  `returns` — so the dispatch law is "handle ⇒ lazy arm," not "handle ⇒ handle
  out" unconditionally.

Interpreted by a **shared `normalize_call`, not codegen** (`lazy-and-bundle.md`
§7), with a `custom` escape hatch. `ApplyOperator` switches to carrying the
serializable call-spec. Subsumes today's `field_handle_inputs` and the
per-operator `normalize_call` boilerplate. The same-level-vs-bundle routing
reads the **coupling-able** derivation; no separate declaration.

*Landed:* the data model (`signature.py`); **dataclass-style operand
declarations** — operators declare `slice_field = FieldInput(...)` etc. as class
attributes, which `OperatorMeta` collects (in definition order, exactly like it
already collects `Parameter`) into a metaclass-built `signature`; and the
interpreter **seam** — `_field_input_slots()` sources the field-slot tuple from
`signature.field_slots()` (else `field_handle_inputs`), with **no
`normalize_call` rewrite** (the minimal/generic win). `bind_slice` migrated as
the proof, behavior unchanged. **`FieldOutput`** (option 2) landed on the
declaration side — collected by the metaclass into `signature.outputs`, the
caller-named produced-field slot (`out`) inherent to lifting ops. *Remaining:*
acting on `FieldOutput` (bind the `out` param → produced column → returned
handle) lands with `merge`'s lazy arm (Phase 6, against a real consumer);
migrate the other `field_handle_inputs` ops (note `bind_dimensions`' nested
`bindings` operand); the `ApplyOperator` serializable call-spec; the
`DatasetInput` bundle-handle dispatch (Phase 5).

### Phase 5 — routing predicate + handle-returning arm

`needs_bundle(op) = not coupling_able(op)` (coupling-able derived as above),
plus the per-call operand bundle-check and the `custom` override. The lazy arm
returns a handle and advances the `DatasetContext` cursor — `handle --op-->
handle` made real — for **both** arms (audit gap #3): a handle to the affected
regular field (same-level / coupling-able) and a handle to the new
`BundleField` column (bundle). Wire the **coupling-able** ops (`bind_*`,
`add_column`, `assign`) here as the first consumers — their lazy arm records the
coupling and returns a handle.

### Phase 6 — wire the needs-bundle set

Roll the lazy arm out across every needs-bundle operator. Per-row-independent
(streaming lift, finest factorization): `where`, **`rename`, `drop`, `keep`**,
`explode`, `window_expansion_plan`, `concat_rows`. Not per-row-independent
(blocking, single-fiber): `merge`, `join`, unaligned `concat_columns`,
`set_index`. `rename`/`drop`/`keep` are here — not in the coupling arm — because
they rewrite/narrow the schema (audit gap #1). `merge` first. Verify chained
in-level graph execution (multiple `ApplyOperator`s on one carrier). Audit
bundle validity / identity (shape-generic, §1); enforce the one-level nesting
cap.

**Rollout — LANDED 2026-06-07 (via the interpreter, no manual branches).** All
needs-bundle transforms wired by declaring a signature only: `rename`, `drop`,
`keep`, `set_index` (single `DatasetInput` + `ParamInput` + `FieldOutput`),
`concat`/`concat_rows`/`concat_columns` (variadic `DatasetInput`), `join` and
`explode` (fixed `DatasetInput`s; their custom `__call__`/`normalize_call` now
forward `out`). `concat` stays a dispatcher — the bundle arm captures it
as-called and re-dispatches the row/column variant at `collect`.

**Taxonomy correction (2026-06-07).** `add_column`/`assign` are *named-output*
ops (the output names are `field_def.name` / the column keys), so their deferred
form is a handle/`Selection` to those columns with **no `out`** — they belong with
the `bind_*` same-level family, not the bundle-lifters. Their blocker was the
*trigger*: they build columns from values, not from a field-reference handle.
**Resolved 2026-06-07 via `new_field`** (not a dataset-scoped `FieldArrayHandle`,
which was considered and dropped as the wrong abstraction): `Dataset.new_field(
field_def)` / `DatasetContext.new_field(...)` adds a null-filled field and returns
a `FieldHandle`. It must be a *cursor* operation (advances the shared context),
**not** a pure `Dataset` function — otherwise `[ds.new_field(a), ds.new_field(b)]`
forks two snapshots and the handles don't co-resolve. With the targets in hand,
`SelectionInput`/`SelectionReturn` fit: `assign([h_a, h_b], values)` (handle form,
values keyed by name) fills them and returns a `FieldSelection`. `assign` keeps a
small `@overload`-style `__call__` split (`Dataset + **cols` vs `selection +
values`; `target`/`values` positional-only). `add_column` will follow.
`window_expansion_plan` is **not** a creation op — it is a
transform (source → plan), a bundle-lifter like `explode`. Its lazy arm needs a
slot-type-aware gate (only `DatasetInput`-slot handles trigger deferral) so its
eager `field`/`bindings` references don't misfire. Its old `normalize_call` was
pre-interpreter boilerplate; **modernized 2026-06-07** — `window_expansion_plan`
is now fully declarative (`dataset=DatasetInput`, `field=FieldInput`,
`bindings=SelectionInput`, `returns=DatasetReturn`; no `field_handle_inputs`, no
`normalize_call`). The source-dataset normalization moved to
`PlanOperator._normalize_source_plan_call` (signature-driven, activates on a
`DatasetInput` slot; resolves source + field-handles via the shared
`Operator._resolve_field_handles_for_dataset`), inherited by any source-dataset
plan op. `make_plan` keeps its own `normalize_call` (its `target` is a
dataset-*level* handle, `_resolve_target`/IndexField — a different pattern). Only
`make_from_dataframe`/`make_plan` are true eager creation entry points.

**Chained in-level graph execution verified**: `bundle` → deferred
`merge` → `where` → `drop`, three `ApplyOperator`s on one carrier, topo-sorted
and materialized in one `collect()`, equals the eager pipeline
(`examples/lazy_eager_duality_usage.py` + `tests/test_lazy_arm.py` +
`tests/test_lazy_duality_example.py`). 404 passed. The lift case (mixed
`Dataset`/handle operands) and the one-level nesting cap remain deferred.

*Landed (both arms proven against real ops):*

- **Bundle arm** — `defer_in_level(operator, *handles, out, params)`: record an
  `ApplyOperator` on the carrier (direct carrier-extension), advance the cursor,
  return the `out` chaining handle. Wired on `merge` (blocking) and `where`
  (streaming-lift) via a manual `__call__` branch (bundle-handle operands →
  `defer_in_level`; eager `Dataset` operands unchanged). `defer_in_level` is
  operator-generic, so the remaining lifting ops reuse it.
- **Same-level arm** — for coupling-able ops, run the op (records its coupling on
  the dataset, no bundle, no `out`) and return a handle to **`coupling.output_
  field`** — one rule covering in-place (`bind_materialize`'s `field`) and fresh
  (`bind_dimensions`'s `slice_field`) outputs. Wired on `bind_materialize` (in-
  place) and `bind_dimensions` (nested `bindings` handles + fresh output) via
  manual `__call__` branches.

Both lazy arms return a chaining handle and propagate the context;
`collect(merge(b.field("left"), b.field("right"), b.field("plan"), out="merged"))
== eager merge`, verified.

**Phase 4 revisit (the learning) — LANDED 2026-06-07.** The four hand-written
`__call__` branches are deleted; one signature-driven interpreter at the top of
`Operator.__call__` drives the routing. Shape:

- **Gate** — `_is_dual_lazy_call`: the op has a signature, is not `custom`, its
  `returns` is `FieldReturn`/`SelectionReturn` (a *handle return* = a
  coupling-producer; this exempts the `DatasetReturn` bridges and eager-only
  ops), and a `FieldHandle` operand is present. Else → `_run_eager` (today's
  lifecycle, factored out, unchanged). `>1` distinct contexts raises here.
- **Route** — `coupling_able()` (derived: `schema ∈ {preserve,extend} ∧
  cardinality PRESERVE ∧ per_row_independent INDEPENDENT`). Same-level arm:
  `_run_eager` (records the coupling, resolves the ambient dataset from the
  handle's context, advances the cursor) then return a handle to the output —
  `output = coupling.output_field`, read from the declaration. Bundle arm:
  `_bind_bundle` → `defer_in_level(type(self), *handles, out, params)`. Both
  mechanisms unchanged; only routing moved in.
- **The binding** (`_bind_slots`/`_bind_bundle`) walks the *ordered*
  `signature.inputs`, fills slots positionally (a `variadic` slot consumes the
  rest), binds kwargs by name, pulls the `FieldOutput` value as `out`, then
  classifies: operand slots → `*handles` (declaration order); `ParamInput` +
  undeclared kwargs → `params` (named, so `ApplyOperator` replays them as kw).
- **`ParamInput`** (new slot, decision **(A)**): a declared per-call
  positional-or-keyword param (`where.predicate`) — *not* an operand, *not* an
  instance `Parameter`. It names a positional argument so eager/lazy calls stay
  positionally symmetric and the deferred call is self-documenting.
- **Output resolution** (`_lazy_output_names`): caller-supplied `FieldOutput`
  value(s) for fresh outputs (`bind_dimensions.slice_field`,
  `merge`/`where.out`), else the `FieldInput` marked `output=True`
  (`bind_slice.data_field`), else the sole `FieldInput` (`bind_materialize`).
  Equivalent to `coupling.output_field` by construction.

Migrations (each ran green): `where` (DatasetInput + ParamInput + FieldOutput,
bundle arm) → `merge` (variadic DatasetInput, bundle arm) → `bind_materialize`
(single in-place FieldInput, same-level) → `bind_dimensions` (FieldOutput
`slice_field` + SelectionInput `bindings`, same-level/nested). **`bind_slice`
gained a lazy arm for free** — it had no manual branch; the interpreter routes
it from its existing signature (the generalization proof). 397 passed.

The binding's normalized call is what `ApplyOperator` will carry; its
**serializability** (#2) stays a separate, deferrable persistence concern
(a `where` lambda replays in-memory but won't pickle).

**Long-term direction (user, 2026-06-07):** make distinct operator call
structures *explicit overloads* (à la PyTorch / `typing.overload`) rather than
one signature + a binding interpreter. The binding is deliberately shaped as
`(signature, args, kwargs) → bound`, which generalizes to "try each overload";
the current single `OperatorSignature` is overload-of-one. Deferred until the
single-signature interpreter is exercised across more ops.

### Phase 7 — `bind_*` convention + contract integration

Resolve the `bind_*` naming convention once the verb pattern is proven. Wire
`per_row_independent` into `assert_operator_contract` so the routing is
mechanically verified, not just declared. Update `lazy-and-bundle.md` with the
converged model.

## Out of scope (workload-gated)

`DatasetAccessor` / lazy `BundleField` cell; the tall/collection substrate
(`over_fibers`, pullback/pushforward, `partition`/`chunk`, `FiberSpec`); the
streaming/fusion executor. `collect` runs the op now — it is not the executor.

## Risks

- **Bundle datasets through existing machinery.** Schema validation,
  unique-index, and the coupling engine must accept `BundleField` carriers.
  Audited lightly in phase 0; re-audit as touched in phases 3/6.
- **Carried call-spec serialization.** The persisted spec must round-trip
  (§7); live state must not leak into it. Phase 4 tests pickle round-trips.
- **`UNKNOWN` capability → conservative routing.** An undeclared/dynamic op
  (`consume`, unaligned `concat_columns`) fails the 3-part test and routes to a
  bundle — safe, never a silent same-level misclassification.
