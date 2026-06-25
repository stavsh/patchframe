# The Constrained `.table` Escape (`pipe`)

Status: **built**, agreed 2026-06-21. Records the model behind `pipe`
(`patchframe/ops/builtin/pipe.py`) — the sanctioned, re-validated alternative to
dropping to `.table` + `make_from_dataframe` for a dataset→dataset transform.
Resolves finding 3.2 in `adtech-findings.md` (the fork "constrained escape vs
growing operators" → *both, as one continuum*; declaration surface = the full
`TransitionPlan` vocabulary).

Cross-references:

- `adtech-findings.md` §3.2 (the forcing function), §3.1/§3.3 (the related
  `reduce`/return-honesty work), §5 (the work order).
- `design-constraints.md` §2 (operators declare effects through `TransitionPlan`;
  flag-dependent contracts refine in `resolve_transitions`), §9 (the three-tier
  authoring budget).
- `operator-authoring.md` — the lifecycle `pipe` reuses; the "declared, never
  detected" law that `pipe` turns into a runtime check.
- `aspect-transition.md` — the transition vocabulary; `SchemaTransition.custom`
  is the escape `pipe` makes callable.
- `schema-specs.md` — the must-contain validation track this rhymes with.

## 1. The smell

`Dataset.table` returns the raw pandas frame. Three uses, two sanctioned:

- **Inspection** (`ds.table[col].mean()`) — read-only. Sanctioned.
- **Fiber-reduce inside a `map_fields` fn** (`fiber.table[col].sum()`) — read-only
  over a fiber. Sanctioned.
- **Transform** — filter / reshape / resolve → rebuild a dataset (typically with
  `make_from_dataframe`). **The smell.**

The sharp case is `attribute()` in `examples/adtech_analysis.py`: a
dataset→dataset transform (lookback-narrow + last-touch resolution) done entirely
through `.table` + pandas, then rebuilt with `make_from_dataframe`. The cost is
not "it uses pandas" — it is that `make_from_dataframe` mints a **parentless**
dataset: source provenance dropped, couplings gone, row identity silently
re-minted, and the result validated against nothing but a hand-written schema.
Every contract patchframe exists to carry is discarded at that boundary,
invisibly.

## 2. The model — an *anonymous operator*

The framework's own vocabulary already names this. `SchemaTransition.custom` is
documented as *"input lineage exists but the transformation is not one the
structural vocabulary captures. Total escape hatch."* The smell is just `custom`
that nobody made callable **at the use site**.

So `pipe(ds, fn, *, schema=, transitions=)` is an operator whose `apply_table`
**is the caller's `fn`** — `fn` receives the input `Dataset` (consistent with the
wrapped surface; its `.table` is an isolated copy) and returns the new *table* (a
DataFrame), which `pipe` re-wraps. The structural effect the caller **declares**
through the ordinary `TransitionPlan` vocabulary, or lets `pipe` **infer from
whether a `schema=` is supplied** (§4). It runs the standard lifecycle
(`normalize → resolve_transitions → run → validate`) and **re-validates the
returned table against the declaration**, failing loud on a lie.

This collapses "constrained escape **vs** growing operators" into one continuum:

> A built-in operator is a *named, tested, reusable* transition declaration with
> a fixed `apply_table`. `pipe` is an *inline, anonymous, per-call* one with a
> user-supplied `apply_table`. Same lifecycle, different commitment level.

- `pipe` is the **floor**: always strictly safer than `.table` +
  `make_from_dataframe` (it cannot silently drop lineage; the return is checked).
- A named operator is the **ceiling**: the home for a recurring, vectorizable
  pattern.
- The **promotion path** is the existing one — when a `pipe` body recurs, copy an
  exemplar into `patchframe/ops/builtin/` and give it the transition declaration
  the `pipe` call already wrote. `examples/` → `patchframe/`, unchanged.

Declaration surface = the real `TransitionPlan`, not a curated mini-enum (the fork
the user resolved 2026-06-21): the escape *is* the contract system made callable;
inventing a parallel vocabulary would re-fight it.

## 3. What "constrained" buys — the per-mode re-validation

The declaration is not decorative; it is what `pipe` checks, and the check is the
whole difference from `make_from_dataframe`:

| declared | `pipe` re-validates the returned table |
|---|---|
| `schema=preserve` | output fields == input fields (else fail loud) |
| `schema=construct` + `schema=` | `schema.validate_table(out)` — the **return-honesty** check (subsumes finding 3.3 here) |
| `index_identity=inherit` | output index ⊆ input index (rows stay in the input namespace) |
| `index_identity=mint` | fresh row identity; the `IndexField` must validate (unique, non-null) |
| `couplings=derive` | keep couplings whose fields survived by identity; invalid with `schema=construct` |
| `couplings=clear` / `inherit` | none / explicitly carry and validate references |
| `sources=inherit` (default) | **source records carried forward** — the concrete win `make_from_dataframe` throws away |

The class-level `transitions` stays maximally conservative (`schema=custom`,
`index_identity=mint`, `couplings=clear`, `sources=inherit`; `cardinality=UNKNOWN`)
per design-constraints §2 — an escape promises nothing structurally until the
call declares it. `resolve_transitions` adopts the call's `transitions=` (the
flag-dependent-contract refinement §2 anticipates), or **infers** one when
omitted (§4). The inference is safe *because* validation is fail-loud: a fn that
violates it raises, it never silently corrupts.

## 4. v1 supported shapes — and the schema-presence shortcut

Two coherent shapes. There is no `cardinality=` argument (a `pipe`'s cardinality
genuinely varies per call, so the *index-identity* mode carries the row check
instead). For the common cases the shape is **inferred from whether a `schema=`
is supplied**, so most calls need no `transitions=` at all:

- **No `schema=` → `preserve` / `inherit`** — a row/value transform (filter,
  sort, recompute). The output schema is **copied from the input**. The
  no-argument default: `pf.pipe(ds, fn)`.
- **A `schema=` → `construct` / `mint`** — a rebuild (the `attribute()` case).
  The output schema is the one supplied (validated against the returned frame),
  a fresh row identity is minted, couplings are cleared because there is no
  trustworthy field-identity lineage through a fresh schema, and sources still
  `inherit` as provenance:
  `pf.pipe(ds, fn, schema=out_schema)`.

`transitions=` is passed explicitly only to override a default (e.g.
`sources=clear`) or to assert `preserve` *while* supplying a `schema=`. Coherence
rule: `schema=construct` *requires* `index_identity=mint` (a construct has no
input lineage, so its rows are a new namespace — matching how `CreationOperator`
pairs `construct` schema with `mint` identity). `preserve` also accepts `mint`
(same columns, renumbered rows). `schema=construct` also rejects
`couplings=derive`; explicit `couplings=inherit` is allowed only after immediate
reference validation.

Everything else **fails loud with a pointer to the right operator**: `extend` →
`map_fields`/`assign`; `narrow` → `keep`/`drop`; `rewrite` → `rename`/`set_index`;
`compose` → `concat`/`merge`. This is deliberate v1 scope — the escape's sweet
spot is the *preserve* and *construct* shapes the smell actually needs;
`narrow`/`extend` support is a clean later addition (it needs identity-carrying
for the kept/added fields) gated on a workload, not built speculatively.

## 5. Division of labor — `pipe` does not excuse the gaps

`pipe` is the guard rail, **not** a reason to leave a recurring transform
unbuilt. `attribute()` rides `pipe` today because its two operations are still
genuinely outside the vocabulary:

- **Lookback = an interval predicate** — `match`/`overlap` landed, so this is
  *nearly* expressible (it needs the timestamps as a temporal-dimension slice,
  the fusion example's pattern).
- **Last-touch = argmax over an ordered fiber** — `reduce` ships
  Sum/Count/Mean/Min/Max/Distinct but **no `ArgMax`/`Last`**, and it needs
  ordered fibers (finding 3.4). Genuinely not expressible yet.

So `attribute()` selects the rebuild by supplying `schema=ATTRIBUTED_SCHEMA`
(which infers construct/mint/inherit-sources) and carries the seven source
records forward + validates the return — strictly better than the old silent
drop — and keeps a `MARK` pointing at `match(overlap)` + an `ArgMax` reducer as
the proper fix. When those land, the body promotes to operators and disappears.
That is the intended lifecycle, not a permanent home.

This is also the precise answer to the findings-doc open question "are 3.1 and 3.2
related": **declared operators (incl. `reduce`) are the safe path; `pipe` is the
escape's guard rail for everything still outside the vocabulary.** Two ends of one
declaration spectrum.

## 6. Considered and not done

- **A curated `kind=` enum** (`pipe(ds, fn, kind="filter")`) instead of the full
  `TransitionPlan`. Rejected (user, 2026-06-21): the escape should *be* the
  contract vocabulary, not a parallel surface. Revisit only if the three-tier
  budget (§9) proves the vocabulary too heavy in conventional usage; the friendly
  no-arg default already covers the common case.
- **Read-only `.table`** (copy-on-access or a writeable=False view) to force all
  transform intent through `pipe`. Rejected: copy-on-access breaks the zero-copy
  fiber-reduce fast path and a writeable flag fights pandas. The inspection/
  transform boundary is kept social + greppable (look for a `.table` feeding a
  `make_from_dataframe`), not enforced by crippling `.table`.
- **`ds.pipe(...)` facade sugar.** Deferred. The top-level `pf.pipe` is primary —
  it keeps the escape *visible* (greppable in review), which a method would blur.

## 7. Named form — the `table_transform` decorator

`pipe` is the inline, anonymous form. Its named sibling is a decorator that
**binds the contract once**: `@table_transform(schema=, transitions=)` over a
`Dataset -> DataFrame` fn yields a reusable `g(dataset, *args, **kwargs)` where
`g(ds, **kw) == pipe(ds, partial(fn, **kw), schema=, transitions=)` (extra args
forward to the fn after the dataset). Supplying `schema=` selects the rebuild
(construct); bare `@table_transform` is the preserve default — a reusable
row/value transform that copies the input schema.

It is a `pipe` *wrapper* (a `functools.partial` for the contract), **not** a
generated operator class — the `schema`/`transitions` are **static**. This is
the deliberate scope (user, 2026-06-21): the decorator carries the common
cases — *no schema change* and a *static output schema* — and a transform whose
schema *varies per call* or is *data-dependent* routes to full operator
authoring, which is a small jump once you are already declaring transitions.

The schema-as-a-function generalization (exposing `apply_schema` as a
caller-supplied derivation, so the fn could "return a new schema") was
considered and **bounded out**: an operator is structurally
`(apply_schema, apply_table)`, and `pipe` deliberately freezes `apply_schema`
to a static value because *a schema independent of the data is what keeps the
return-honesty check meaningful* — if the fn computed its schema from the data,
`validate_table` would pass by construction and the check that justifies the
escape over `make_from_dataframe` goes vacuous. Derived/dynamic schemas earn a
real, tested operator instead.

`attribute()` (`examples/adtech_analysis.py`) is the worked example:
`resolve_last_touch(dataset, *, lookback_days)` is `@table_transform`-decorated
with just `schema=ATTRIBUTED_SCHEMA` (the schema-presence shortcut infers the
construct/mint/inherit-sources contract), and `attribute` just calls
`resolve_last_touch(pairs, lookback_days=...)`.

## 8. Open

- **A lazy, column-adding table escape — the deferred `extend` arm.** `pipe` is
  eager-only and supports only `preserve`/`construct`; there is no path for a user
  who wants to write a *table-level* function that **adds a column** and **thread
  it into the lazy computation graph**. The envisioned case (user, 2026-06-22): a
  custom table-accessing fn that appends a column with **whole-column context** —
  a rank, cumulative sum, groupby-transform — which is *not* the per-row
  `map_fields` shape and *not* a full `construct` rebuild, recorded as a deferred
  node and evaluated at `consume`/`collect`/row-access. Two pieces, neither built:
  1. **A `FieldReturn` lazy arm for `pipe`/`table_transform`** — handed a
     `FieldHandle`, *record* the table-fn (returning a chaining handle) instead of
     running it now, so it joins the coupling/bundle graph. This is the dual arm
     every real operator has (`map_fields`, `where`); `pipe` deliberately ships
     eager-only in v1, so making it dual is what completes the "anonymous operator"
     story (§2).
  2. **A whole-table / whole-chunk compute coupling** — distinct from the per-row
     `MapCoupling` (`map_fields`), the bundle-cell `ApplyOperator`, and the fiber
     `ReduceCoupling`: a fn over whole input columns → an output column. The
     coupling model *can* hold it (`design-constraints.md` §3: a coupling is a
     deferred operator application, evaluable eagerly or chunk-wise), but it
     surfaces the **per-row-independence** question — a column-add with cross-row
     dependence is not `coupling_able()` (the gate requires `per_row_independent
     INDEPENDENT`), so today it would lift onto the bundle carrier, awkward for
     "just add a column." The honest axis is *partition strategy*, not
     row-vs-table (`design-constraints.md` §3).

  No example forces this yet — recorded as a deliberate deferral, not an oversight
  (build it at the workload, per the staging discipline). The eager `extend` mode
  (a column-adding table-fn run now, today rejected with a pointer to
  `map_fields`/`assign`) is the degenerate case of the same gap.
- `narrow` schema mode for `pipe` (identity-carrying for kept fields) — when a
  workload wants it.
- An `assert_operator_contract` that exercises a `pipe` call against its declared
  transition the same way it will exercise built-ins (the planned contract suite;
  `pipe` is a natural first client since its contract is per-call).
- Whether the return-honesty check should reuse the must-contain spec machinery
  (`schema-specs.md`) once that lands — a declared output spec is exactly what
  `construct` validates against.
