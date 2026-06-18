# Schema Specs and Plan Typing

Status: design rationale, agreed 2026-06-17. **Not built** — gated on a
forcing function (the factored-plan work, or a concrete authoring-ergonomics
pull). Records the model so it can be built without re-deriving it.

Cross-references:

- `join-dimensions-identity.md` §1.6 — the two plan kinds (single-FK
  one-to-many vs two-FK many-to-many) this types.
- `dimension-join-execution.md` §2 — the factored/block representation (a
  multi-dataset constellation) that motivates richer plan typing.
- `lazy-and-bundle.md` §3 — the bundle substrate a factored plan rides on.
- `aspect-transition.md` — the `TransitionPlan` schema modes that *derive* the
  propagation rules (§2).

## Purpose

An **optional** layer of explicit, typed schema specifications, with two pulls:

- **Stricter plan protocols.** Today a dataset is treated as a plan
  structurally — it carries the `ForeignIndexField`(s) a consumer needs. That
  duck-typing is fine for the single-dataset expanded plan but does not extend
  to the **factored** plan (a multi-dataset constellation; §5). Naming plan
  *shapes* as types lets consumers dispatch and validate explicitly, and lets
  plan structure be **richer** than "exactly these FK columns" — the type is
  the contract; everything else is free.
- **Authoring / IDE ergonomics.** A readable, typed declaration of what a
  dataset's schema must contain.

The whole layer stays optional: datasets without a spec are dynamic, exactly
as today.

## 1. Must-contain specs (not exact-match) — at boundaries, not threaded

The defining choice (user, 2026-06-17): a spec declares **what the schema must
contain** (fields by name/type), and **tolerates additional fields added
dynamically**. This is *not* SQLAlchemy-style exact-match — and that deviation
is what makes it fit a transformation-based framework. An exact-match class
breaks the instant an operator adds a payload column; a must-contain spec
survives `extend`/`concat_columns` (the required fields persist) and stops
matching only when a required field is genuinely *removed* or *retyped* — which
is the correct contract behavior, not rigidity. Must-contain is also
order-independent (a `Schema` is ordered; specs are a set check).

Three uses, all at **boundaries**:

1. **Validation** — `CorrespondencePlan.matches(ds)`, or a consumer declaring
   "I require a `CorrespondencePlan`." Formalizes and *names*
   `PlanConsumerMixin` / `required_plan_fields`.
2. **Declaration** — define a dataset/plan's required schema as a readable,
   typed class (the IDE win at the definition site).
3. **Construction** — a factory that builds a `Schema`/dataset with the
   declared fields, then lets more be added.

**Not** a threaded typed *wrapper.* A `Dataset` carries its schema as data
(`.schema`, `.table`), not as class attributes, and operators return bare
`Dataset`s — so a wrapper that preserved a spec-type through every operator
would require operators to become spec-aware (re-fighting the transformation
model). The win lives at the endpoints (declare, validate); §2 makes it thread
*advisorily* without that cost.

## 2. Transition-derived propagation (best-effort, safe)

A `Dataset` may carry an **advisory** spec-type annotation. Operators
propagate it best-effort, and the rules are *derived from the existing
`TransitionPlan` schema modes* — little per-operator work:

- **`preserve` / `extend` → propagate, free.** Required fields survive both, so
  if the spec held on input it holds on output. No check.
- **`narrow` → propagate iff required ∩ `dropped` = ∅** (the transition already
  carries `dropped`).
- **`rewrite` → check `rename_map`** (the subtlety; see below).
- **`construct` → the operator's *declared output spec*.** A fresh schema has no
  input lineage → dynamic by default, *unless the producer declares it emits a
  spec* (`match`/`join` → `CorrespondencePlan`). This is the highest-value,
  cheapest piece — one declaration per plan-producing operator, at the origin.
- **`compose` → declared or dynamic.**

**Safe by construction — no false positives.** Each propagation is *justified
by the transition info* (the required fields provably survived), so the type is
attached only when it genuinely still holds; when unsure, dynamic. "Best-effort"
means "type when provable, dynamic otherwise," never "guess." A dataset is never
mislabeled.

**Where the annotation lives:** advisory, alongside `metadata` — it must not be
part of semantic state. (See §3 for the one way it is allowed to be
load-bearing.)

**The `rewrite` subtlety — match by name or by identity.** A must-contain-by-
*name* spec (`left_index`, `right_index`) breaks under `rename` unless re-mapped
through `rename_map`; a by-`FieldIdentity` spec survives rename but is harder to
express declaratively. For plans, **by name** is the call (the mapping columns
are conventional), which makes `rename` → dynamic — acceptable (renaming a
plan's index columns is rare; the fallback is safe). This is the one rule that
is not purely mechanical; decide it explicitly when building.

## 3. Load-bearing at consumption — fail-loud, with structural fallback

The guardrail, refined (user, 2026-06-17): **load-bearing is fine; silent
fallback is not.** A plan-consuming operator *may require* a spec — but the
failure mode must be a **loud reject** ("needs a `CorrespondencePlan`; got an
untyped dataset"), never a quiet branch into different behavior. The danger was
never "the type is used in execution"; it was "a best-effort annotation that
falls back to dynamic silently changes what an operator does."

To make required-typing robust against the propagation's best-effort gaps,
**dispatch by spec-*match*, with the propagated type as a fast-path hint:**
structural must-contain is the ground truth; the annotation short-circuits it.
A dataset that *is* structurally a `CorrespondencePlan` but lost its annotation
through some transform is still accepted (the structural check catches it) — no
false-negative rejections. So the type is genuinely load-bearing without being
brittle, and §2's best-effort gaps are harmless.

This is the major sell point: plans defined by type → richer plan structures +
reliable dispatch, without the duck-typing's blind spots.

## 4. A plan taxonomy that types the existing operators (not a mega-operator)

Name the plan shapes — `ExpansionPlan` (single-FK, one-to-many),
`CorrespondencePlan` (two-FK, many-to-many), `FactoredCorrespondence` (§5) — and
have `explode`/`implode`/`partition`/`merge` each **declare which plan type they
consume and require**. That yields the dispatch, validation, and richer-
structure payoff while keeping each operator's contract fixed and its call site
explicit.

**Unifying the operators themselves into one `apply_plan` was considered and
rejected** (2026-06-17), for two concrete reasons:

1. **The plan type underdetermines the operation.** A two-FK correspondence
   supports *both* `explode` (expand one side → flat pairs) *and* `implode`
   (collapse → grouped fibers) — the cardinality *dual* — and which one is a
   *user choice the type cannot make*. So dispatch-by-type-alone can't unify
   them; you'd pass a direction, at which point "one operator + direction" is no
   simpler than two named operators and *loses* call-site clarity (`explode`
   tells you the output shape; `apply_plan` doesn't). It is unambiguous only at
   the degenerate end (a one-FK plan can only expand).
2. **Type-dependent output shape is a flag-dependent contract.** The output
   *shape* would vary by plan type (flat rows vs fibers) — the exact pattern the
   project rejects (`explode`'s `keep_source_index` flag) — requiring the
   deferred explicit-overloads mechanism (per-type `TransitionPlan`) to be
   honest.

So: unify the **type layer**, keep the operators separate and explicit. The
user picks expand-vs-collapse by *choosing the operator*, which is the honest
place for that choice.

## 5. Factored plan = a `FactoredCorrespondence` bundle type

The factored correspondence is a multi-dataset constellation (`blocks`,
`left→block`, `right→block`; different lengths — not one plan dataset;
`dimension-join-execution.md` §2). patchframe's multi-dataset container already
exists — the **bundle** (a dataset whose `BundleField` cells hold sub-datasets).
So a factored plan is a `FactoredCorrespondence` **bundle type**, and the same
spec/propagation/dispatch machinery types it. This is the original motivation:
the "different shape" a future plan needs is a bundle, and typing is how
consumers tell `CorrespondencePlan` (expanded, one dataset) from
`FactoredCorrespondence` (factored, a bundle) and dispatch accordingly.

## 6. Open questions

- **The reshape-family intuition** (user, 2026-06-17): `explode`, `implode`,
  `partition`, and `merge` all "consume a relation (plan/key) and produce a
  reshaped dataset," differing only in output shape (expanded rows / fibers /
  wide pairs). There is a *tingling* sense they form one family — maybe
  unifiable at a deeper level than plan-type dispatch (which §4 ruled out). Not
  enough to push; recorded so the thread is not lost. The honest reading so far
  is *a family* (relational reshape), not *one operator*.
- **Declarative schemas for general datasets** (beyond plans) — a `pydantic`/
  `attrs`-style ergonomic layer for defining any dataset's schema — is a
  separate, broader track with the same must-contain-not-exact-match
  consideration. Decide whether the spec layer covers general datasets from the
  start or plans first.
- **The `rewrite` match criterion** (§2) — name vs identity; lean name.
- **Where the annotation lives** — advisory channel parallel to `metadata`; must
  not enter semantic state.

## Timing

Build at the forcing function — the factored-plan work (which *needs* the
bundle plan type and the dispatch) or a concrete authoring-ergonomics pull — not
speculatively. Propagation (§2) is not a bolt-on: `construct`-producers
declaring their output spec is the spine, so design it in from the start. v1 of
the join is all-expanded and does not need any of this yet.
