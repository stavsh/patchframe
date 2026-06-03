# Field Identity and Composition Lineage

Status: internal design rationale. Not a public API spec. Records the model
for `FieldIdentity` and the `MergedField` composition-lineage mechanism. This
is the stage *after* the AspectTransition tightening (see
`aspect-transition.md`); it is not required for that stage to ship.

Cross-references:

- `aspect-transition.md` — transition vocabulary and the mechanisms that
  consume it.
- `design-constraints.md` §1 — identity must be serializable and stable
  across save/load.

## The problem

Reliable coupling derivation needs **field lineage**: which input field became
which output field, which were dropped, which were constructed, and — in
composition — which input field won a collision. Today that lineage lives in
one of two unsatisfactory places:

- **Re-inferred** — comparing input/output schemas by name/type. Unreliable:
  a renamed field (`a`→`b`) is indistinguishable from a drop of `a` plus an
  add of `b`.
- **Declared** — the `schema=rewrite(mapping=...)` agreed for Stage A. Correct,
  but it asks operators to hand-maintain a mapping the framework should be able
  to know structurally.

Patchframe's semantic-state model (README) says identity is *carried state*,
minted at creation and propagated by operators — not re-inferred from content.
Rows already follow this: `IndexIdentity` is minted and propagated, and
`ForeignIndexField` carries cross-namespace row references. Fields do not. This
asymmetry is the gap.

## The FieldIdentity model

Every `Field` carries a `FieldIdentity` — a minted uuid token, the column-level
analog of `IndexIdentity`.

- **Minted** in `Field.__post_init__` when not supplied. Every field carries an
  identity from construction; there is no anonymous/`None` state. Minted, not
  content-derived: a field's semantic identity must survive value changes,
  retyping, and renaming.
- **Preserved** by `dataclasses.replace` — it copies the existing non-`None`
  `field_identity`, so `__post_init__` does not re-mint. Rename and retype
  (which go through `replace`) keep identity for free; passing a field through
  unchanged keeps it.
- **Fresh** for genuinely new fields — any field constructed from scratch
  (`add_column`'s field, plan fields, `join` output) mints a new identity.
  Operators that rebuild a *logically-same* field from scratch (`set_index`
  promoting a column) must thread `field_identity` explicitly.

Minting site — resolved. Index identity is `DatasetState`-minted because a row
namespace is a dataset-instance property and a schema is a reusable template.
A field's identity is a property of the field itself, so it is minted at field
construction. This also gives the "never `None`" invariant for free.

### Structural equality

`field_identity` is declared `compare=False` on the `Field` dataclass, so it is
excluded from `__eq__`. Structural equality stays structural — two fields with
the same name/dtype/type are `==` regardless of lineage. This is required:
schema equality is used pervasively (operator output checks, tests), and an
identity-sensitive `==` would break it wholesale. Lineage is compared
explicitly via `field_identity` where it matters (e.g. `concat_rows` taking the
trivial path when all inputs for a name share one identity).

(`Field` is already unhashable due to its `metadata` dict; `compare=False` does
not change that.)

### Beyond lineage

`FieldIdentity` is also the stable token the context-bound `FieldHandle` layer
resolves against. `DatasetContext` handles support imperative authoring while
stored couplings remain local and name-based. A future serializable
`DatasetRef` for `BundleField` values is still separate work. The uuid-token
design supports that extension without further change.

The `IndexIdentity` work is the playbook for the propagation shape; the minting
site differs (field construction vs dataset construction) for the reason above.

### What it solves

With every field carrying identity, lineage for unary operators becomes
structural — compare output field identities against input field identities:

- identity in both → same field (possibly renamed/retyped).
- identity in input, absent from output → dropped.
- identity in output, absent from input → constructed.

Consequences:

- **`schema=infer` becomes reliable.** The caveat accepted in
  `aspect-transition.md` — "infer cannot detect renames" — disappears.
- **`schema=rewrite(mapping=...)` declarations become unnecessary.** `rename`
  and `set_index` stop hand-declaring the mapping; the framework reads it from
  identities. The `rewrite` mode stays in the vocabulary; only its hand-written
  `mapping` detail goes away.
- **Coupling rewrite churn shrinks** (see sub-question 3).

## Sub-question 1: collision identity in composition — `MergedField`

Composition collisions are the hard case. When `concat_columns` merges two
datasets that both have a field named `label`:

- Left has coupling `label ↔ audio`.
- Right has coupling `label ↔ spectrogram`.

Both couplings reference the *name* `label`. After the collision resolves (say
left wins), the output `label` field **is left's field**. The right-side
coupling `label ↔ spectrogram` then silently re-targets to left's `label` —
semantically wrong. Right's `label` did not survive; its coupling should be
pruned or strategy-handled, not quietly re-pointed.

Name-based couplings cannot catch this: the colliding fields share a name, so
the name is ambiguous *precisely across the collision*. Identity is required to
say "this coupling was about right's `label`, and right's `label` lost."

`merge` partly avoids this already — its `ColumnCollisionStrategy` `left`/
`right` paradigm declares which side wins, so the surviving identity is known.
`concat` has no such parameter in its natural signature, and requiring one
would feel unnatural to users who expect `concat(a, b)` to just work.

### The mechanism

`MergedField` is a first-rate `Field` subclass produced during composition
when fields share a name. It holds:

- `parents: tuple[FieldParent, ...]` — ordered `(input_index, field)` pairs so
  both coupling work and table work can identify which input each parent came
  from.
- the collision strategy (`None` for a row unification).
- the composition context.

It rides in the intermediate schema returned by `apply_schema` and is consumed
by every aspect hook that `_compose` orchestrates. `MergedField` owns the
collision resolution itself through two methods over the same
parents/strategy:

- `resolve_field() -> Field` — the schema-side answer (used by
  `resolve_merged_fields` at the end of `_compose`).
- `resolve_column(parent_columns) -> pd.Series` — the table-side answer
  (called by `apply_table` with the parent columns gathered by
  `input_index`).

Because both methods read the same lineage and the same `winning_parent()`,
schema winner and table winner cannot diverge — the collision decision is made
once. `apply_couplings` auto-derives: for each `MergedField`, the losing
input's couplings on the superseded name are pruned (shared helper
`derive_composed_couplings` in `ops/builtin/_composition.py`). `concat_rows`
is the carve-out — its couplings stay preserved by field name.

`MergedField` complements `FieldIdentity`; it does not replace it. Identity
makes the parents distinguishable (`left.label` ≠ `right.label` despite the
shared name); `MergedField` carries those distinguishable parents through the
apply pipeline.

This is deliberately lighter than persistent lineage edges
(`FieldIdentity.derived_from`). Coupling derivation needs lineage only *during*
the operation that derives couplings; afterward the winner has a clean identity
and couplings are settled. `MergedField` scopes the lineage to exactly that
window — transient scaffolding, not permanent infrastructure that accumulates
and must be serialized.

### Design points

1. **First-rate `Field` subclass, not a proxy.** `MergedField` mirrors the
   would-be-resolved field's attributes (`name`, `dtype`, `nullable`,
   `primary`) so generic field-reading code (`Schema` validation,
   `normalize_column`, etc.) treats it like any other field. Build through
   `MergedField.over(parents, collision=..., context=...)` so the mirror stays
   consistent with `resolve_field()`.
2. **`_compose` resolves before assembly.** `apply_schema` returns the
   intermediate (possibly MergedField-bearing) schema; every aspect hook reads
   it; `resolve_merged_fields` collapses it just before the final
   `DatasetState`. No MergedField appears in a returned dataset's schema.
3. **Flat N-ary parents.** A three-way collision produces
   `MergedField(parents=(p0, p1, p2), ...)`, not nested wrappers.
4. **One decision, two answers.** `resolve_field` and `resolve_column` read the
   same `parents`/`collision`/`context` and the same `winning_parent()`, so
   schema winner and table winner are consistent by construction.
5. **No bucket info on `MergedField`.** Bucket policy (primary-field downgrade
   across same-type fields) is orthogonal to collision (same-name fields). It
   stays in `compose_column` for non-colliding fields; a resolved MergedField
   produces an ordinary field that goes through the same bucket check as every
   other output field.
6. **Sensible default resolution.** `MergedField` accepts a strategy but
   defaults to first/left-wins, so `concat(a, b)` works with no strategy
   argument and `collision=...` stays optional. This is what keeps the
   `concat` signature natural without a `FieldIdentityLineage` mechanism.

### Rejected alternative

`apply_schema` could resolve collisions immediately and return a clean schema
plus a side-channel `lineage` map for `apply_couplings`. This keeps the schema
always-valid (no transient pseudo-field) but scatters the lineage into a
side-channel every consumer must be handed separately. `MergedField` keeps the
lineage *in* the structure it describes, so `apply_couplings` just reads the
schema. Self-describing wins, despite the "never escape" sharp edge.

## Sub-question 2: row-stack identity

`concat_rows` is not a "winner" case. When `label` appears in both row-stacked
datasets, the output field is a genuine unification (left's rows then right's
rows). The rule, parallel to `IndexIdentity`'s composition behavior:

- **preserve** the field identity if all parents share one identity,
- **mint** a fresh identity otherwise.

`MergedField` can carry this (`strategy="row_unify"`), but its resolution for
row-unify does not strictly "pick a parent" — it returns the shared identity or
a freshly minted field. The honest contract is therefore "`MergedField`
resolves to a concrete field" — usually a parent for column collision, possibly
minted for row-unify.

## Sub-question 3: do couplings reference identity or name?

- **Option A — identity for lineage only.** Couplings stay name-based. The
  framework computes the rename map from field identities and still runs
  `CouplingSet.rewrite_field_names`. Smaller change; couplings stay readable.
- **Option B — identity-based coupling refs.** Couplings reference
  `FieldIdentity` directly; rename needs no coupling rewrite at all because the
  identity is stable. Larger change; couplings become opaque to a user reading
  them (need name resolution for display).

**Recommendation: Option A first.** It gets reliable lineage without migrating
the coupling reference model. Migrate to Option B later only if name-based
references prove fragile in practice.

## Sub-question 4: serialization

`FieldIdentity` is a minted token and inherits the `design-constraints.md` §1
rule: it must round-trip through save/load and be stable across sessions, so
datasets persisted separately reconnect correctly. Not content-derived.

## Sub-question 5: `IndexField` carries two identities

After this stage, an `IndexField` carries both:

- its `IndexIdentity` — the row namespace the column defines,
- its `FieldIdentity` — the column itself as a schema entity.

These are different things and must not be conflated. Both serialize. Naming
and accessors should keep them visibly distinct.

## Future: `MergedField` owning composition control flow

Once the basics above work, `MergedField` is a natural place to absorb
composition control flow and cut operator boilerplate. Today each composition
operator hand-writes the loop: detect collisions by position, call
`compose_collision_field`, resolve schema/table/couplings separately.

A later refactor could give `MergedField` responsibility for resolving all
three concerns consistently — schema field, table column, coupling
implications — since it already holds the parents, strategy, and context. The
composition operator would shrink to: pair fields, wrap collisions in
`MergedField`, hand off to a generic resolver. `concat_rows`, `concat_columns`,
and `merge` would then share resolution machinery and differ only in how they
*pair* fields — which also cleans up the row/column decomposition noted in the
operator-family discussion.

This is deliberately postponed. Implement `MergedField` as the collision
lineage carrier first, validate it against real composition behavior, then
expand its responsibility.

## Staging

- **Stage A (AspectTransition tightening) shipped**, using declared
  `schema=rewrite(mapping=...)` as the interim lineage mechanism. Composition
  keeps its current `union_couplings` / `preserve_row_couplings`.
- **B1 — FieldIdentity foundation (shipped).** `FieldIdentity` +
  `new_field_identity`; `Field.field_identity` (`compare=False`) minted in
  `Field.__post_init__`; `set_index` threads identity through its rewrite;
  other unary operators propagate via `replace`/passthrough. Tests in
  `tests/test_field_identity.py`. No behavior change to coupling derivation.
- **B2 — MergedField (shipped).** `MergedField` as a first-rate `Field`
  subclass with `resolve_field` / `resolve_column` / `winning_parent`;
  `FieldParent` (input-indexed parents); `CompositionOperator._compose`
  threads the intermediate schema to every aspect hook and resolves last;
  `concat_columns`, `merge`, and `concat_rows` all produce `MergedField`s
  (collision for the first two, row unification for the last);
  `apply_couplings` auto-derives via the shared `derive_composed_couplings`
  helper (`concat_rows` keeps `preserve_row_couplings` as the name-based
  carve-out). Composition's previous helpers `compose_collision_field` /
  `resolve_collision_column` / `union_couplings` were removed as dead code.
  Tests in `tests/test_merged_field.py`.
- **B3 — wire identity into derivation (shipped).** `_derive_couplings` is
  identity-based and mode-agnostic: it compares input/output schemas by
  `field_identity` to derive the rename map and the surviving set, replacing
  the declared `rewrite(mapping=...)` mechanism for unary operators.
  `schema=infer` is now operationally as precise as any other mode for
  coupling work. `rename`'s `resolve_transitions` mapping injection was
  removed; the `mapping` field on `SchemaTransition` was removed as dead
  state. Tests in `tests/test_field_identity.py`.

The stages compose without wasted work: Stage A's `rewrite(mapping=...)` was
the manual version of what B3 later automated.

## Open questions

- **Minting sites — resolved.** Field identity is minted in
  `Field.__post_init__` (every field carries one from construction; `replace`
  preserves it). See "The FieldIdentity model".
- **Permanent lineage edges.** Whether `FieldIdentity` ever needs a persistent
  `derived_from` — current position is no; `MergedField`'s transient lineage
  covers coupling derivation, and dataset-level provenance belongs to sources.
  Revisit only if a concrete consumer for permanent field provenance appears.
- **Bundle interaction.** `BundleField` (a field type) carries a
  `FieldIdentity` like any field; this is orthogonal to the base/fiber/total
  *row* identities a Bundle tracks. Confirm no conflation when Bundle work
  begins.
- **Option B trigger.** What concretely would signal that name-based couplings
  are fragile enough to justify migrating to identity-based references.
