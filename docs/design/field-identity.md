# Field Identity and Composition Lineage

Status: internal design rationale. Not a public API spec. Records the model
for `FieldIdentity` and the `MergedField` composition-lineage mechanism. This
is the stage *after* the AspectTransition tightening (see
`aspect-transition-plan.md`); it is not required for that stage to ship.

Cross-references:

- `aspect-transition-plan.md` — Stage A; ships with declared
  `schema=rewrite(mapping=...)` as the interim lineage mechanism.
- `aspect-transition-ontology.md` — transition vocabulary.
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

`FieldIdentity` is also intended as the stable token a future field-reference
layer resolves against — binding fields by reference instead of by name, and a
`BundleField` reference signalling an operator to be treated as a lazy
coupling. That "field reference manager" is future work, kin to the
`FieldHandle` / `DatasetRef` layer in `design-constraints.md`; it is not built
in this stage. The uuid-token design supports it without further change.

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
  `aspect-transition-plan.md` — "infer cannot detect renames" — disappears.
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

`MergedField` is a thin, transient wrapper produced during composition when
fields collide. It holds:

- references to the 2+ parent `Field`s,
- the collision strategy,
- the composition context.

It lives only inside `_compose`. It travels through `apply_schema` →
`apply_couplings`, carrying the distinguishable parents so coupling derivation
can ask it, per parent, "does your resolution keep this parent's couplings?"
Just before the final `DatasetState` is assembled, it **resolves** to a
concrete field.

This is deliberately lighter than persistent lineage edges
(`FieldIdentity.derived_from`). Coupling derivation needs lineage only *during*
the operation that derives couplings; afterward the winner has a clean identity
and couplings are settled. `MergedField` scopes the lineage to exactly that
window — transient scaffolding, not permanent infrastructure that accumulates
and must be serialized.

`MergedField` complements `FieldIdentity`; it does not replace it. Identity
makes the parents distinguishable (`left.label` ≠ `right.label` despite the
shared name); `MergedField` carries those distinguishable parents through the
apply pipeline.

### Design points

1. **Field-duck-typed.** `MergedField` subclasses `Field` and proxies attribute
   reads (`.name`, `.dtype`, `.logical_type`) to its default-resolved parent,
   so most of the apply pipeline does not special-case it. Only
   collision-aware code (`apply_couplings`, the resolver) inspects it.
2. **It must never escape.** Because it subclasses `Field`, it could survive
   into a returned `DatasetState` and pass schema validation. Enforce an
   invariant: a final schema must contain no `MergedField`. Assert in
   `_compose` (or schema validation).
3. **Flat N-ary parents.** Three-way `concat` produces `MergedField(a, b, c)`,
   not nested wrappers.
4. **Resolution agrees with table-column resolution.** The strategy that picks
   the winning *field* must be the same one `resolve_collision_column` uses for
   the winning *column values*. Schema winner and table winner cannot diverge.
5. **Resolution timing.** In `_compose`, after `apply_schema` and
   `apply_couplings`, before final state assembly. Composition's `apply_schema`
   therefore returns an *intermediate* schema that may contain `MergedField`s —
   a real but contained contract change.
6. **Sensible default resolution.** `MergedField` accepts a strategy but
   defaults to first/left-wins, so `concat(a, b)` works with no strategy
   argument and `collision=...` stays optional (already true of
   `concat_columns` today). This is what keeps the `concat` signature natural
   without a `FieldIdentityLineage` mechanism.

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
- **B2 — MergedField.** The composition-collision carrier and composition's
  field-identity rules (row-stack preserve-or-mint). Not yet started.
- **B3 — wire identity into derivation.** Make `_derive_couplings` use
  identity-based lineage and `schema=infer` reliable; drop `rename`'s
  `resolve_transitions` mapping injection. Deferred until B1/B2 are stable.

The stages compose without wasted work: Stage A's `rewrite(mapping=...)` is the
manual version of what B3 later automates.

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
