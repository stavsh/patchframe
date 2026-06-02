"""
patchframe.ops.resolution

Collapse declared transition plans into concrete-mode plans.

``resolve_derived_transitions(declared, *, input_schemas, output_schema)``
is the operator-independent helper that walks the resolution table from
``docs/design/aspect-transition.md``: every aspect declared ``derive`` is
replaced by its concrete outcome mode (or by concrete resolution data, in
the couplings case) based on the resolved schema mode plus structural
reads of the input/output schemas.

Pure function: no operator instance required, no operator hooks called,
no side effects. Consumers (the framework dispatch in Phase 3, the
contract suite later) call it with the schemas they already have.

Couplings is the centerpiece. ``derive`` couplings stay declared as
``derive`` after resolution, but the function populates the resolution
data fields on ``CouplingsTransition`` so the lineage analysis is
inspectable and testable:

- ``rename_map`` — survivors whose ``FieldIdentity`` reappears in output
  under a different name (rewrite lineage).
- ``dropped`` — input field names whose ``FieldIdentity`` is absent from
  output (narrow lineage).
- ``superseded_per_input`` — for compose, per-input losing-parent names
  walked from ``MergedField.parents`` and ``winning_parent()``.
"""

from __future__ import annotations

from patchframe.dataset.field_composition import MergedField
from patchframe.dataset.fields import IndexField
from patchframe.dataset.schema import Schema
from patchframe.ops.transitions import (
    CouplingsTransition,
    IndexIdentityTransition,
    SourcesTransition,
    TransitionPlan,
)


def resolve_derived_transitions(
    declared: TransitionPlan,
    *,
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> TransitionPlan:
    """Return ``declared`` with every resolvable ``derive`` aspect collapsed.

    Sources ``derive`` resolves to ``inherit(input=0)`` for the unary-shaped
    schema modes (preserve/extend/narrow/rewrite/infer), to ``construct``
    under ``construct`` schema, and to ``compose`` under ``compose`` (the
    dispatch handler produces a deduped union of input sources).

    Index identity ``derive`` resolves to ``inherit(input=0)`` for the
    unary-shaped schema modes, ``mint`` under ``construct``, and
    ``coalesce`` under ``compose``. Under ``rewrite`` schema, raises if the
    rewrite touched the primary ``IndexField`` (detected by comparing
    ``FieldIdentity`` between input and output); the operator must declare
    ``mint`` or ``inherit`` explicitly in that case.

    Couplings ``derive`` stays ``derive`` but the returned
    ``CouplingsTransition`` carries ``rename_map`` / ``dropped`` /
    ``superseded_per_input`` populated from schema lineage.

    Under ``custom`` schema, any aspect declared ``derive`` raises — the
    structural vocabulary cannot describe the operator's effect, so the
    operator must declare every other aspect explicitly too.

    Raises ``ValueError`` for ``construct`` + ``derive`` couplings (no
    input lineage to derive from); the operator must declare ``clear``,
    ``construct``, ``inherit``, or ``homogeneous``.
    """
    return TransitionPlan(
        schema=declared.schema,
        table=declared.table,
        couplings=_resolve_couplings(declared, input_schemas, output_schema),
        sources=_resolve_sources(declared),
        index_identity=_resolve_index_identity(
            declared, input_schemas, output_schema
        ),
        accessors=declared.accessors,
    )


def _resolve_couplings(
    declared: TransitionPlan,
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> CouplingsTransition:
    transition = declared.couplings
    if transition.mode != "derive":
        return transition
    schema_mode = declared.schema.mode
    if schema_mode == "construct":
        raise ValueError(
            "resolve_derived_transitions: schema=construct with "
            "couplings=derive has no defined outcome (no input lineage). "
            "Declare clear / construct / inherit / homogeneous explicitly."
        )
    if schema_mode == "custom":
        raise ValueError(
            "resolve_derived_transitions: schema=custom with "
            "couplings=derive has no defined outcome. Declare a concrete "
            "couplings mode explicitly."
        )
    rename_map, dropped, superseded = _compute_couplings_lineage(
        input_schemas, output_schema
    )
    return CouplingsTransition(
        mode="derive",
        input=transition.input,
        rename_map=rename_map,
        dropped=dropped,
        superseded_per_input=superseded,
    )


def _resolve_sources(declared: TransitionPlan) -> SourcesTransition:
    transition = declared.sources
    if transition.mode != "derive":
        return transition
    schema_mode = declared.schema.mode
    if schema_mode == "custom":
        raise ValueError(
            "resolve_derived_transitions: schema=custom with sources=derive "
            "has no defined outcome. Declare a concrete sources mode."
        )
    if schema_mode == "construct":
        return SourcesTransition.construct()
    if schema_mode == "compose":
        return SourcesTransition.compose()
    return SourcesTransition.inherit(input=0)


def _resolve_index_identity(
    declared: TransitionPlan,
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> IndexIdentityTransition:
    transition = declared.index_identity
    if transition.mode != "derive":
        return transition
    schema_mode = declared.schema.mode
    if schema_mode == "custom":
        raise ValueError(
            "resolve_derived_transitions: schema=custom with "
            "index_identity=derive has no defined outcome. Declare a "
            "concrete index_identity mode."
        )
    if schema_mode == "construct":
        return IndexIdentityTransition.mint()
    if schema_mode == "compose":
        return IndexIdentityTransition.coalesce()
    if schema_mode == "rewrite":
        _ensure_rewrite_preserves_primary_index(input_schemas, output_schema)
    return IndexIdentityTransition.inherit(input=0)


def _compute_couplings_lineage(
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    tuple[tuple[int, tuple[str, ...]], ...],
]:
    """Walk input/output schemas to compute the couplings resolution data.

    Returns ``(rename_map, dropped, superseded_per_input)``:

    - ``rename_map`` lists ``(input_name, output_name)`` for non-MergedField
      output fields whose ``FieldIdentity`` matches an input field of a
      different name.
    - ``dropped`` lists input field names whose ``FieldIdentity`` does not
      appear in any output field (including MergedField parents) and was
      not the winning parent of a column-collision MergedField.
    - ``superseded_per_input`` lists ``(input_index, names)`` pairs for
      column-collision losing parents — the names whose couplings the
      compose-derive helper should prune on the loser's side.
    """
    rename_pairs: list[tuple[str, str]] = []
    superseded: dict[int, list[str]] = {}
    surviving_identities: set = set()

    for output_field in output_schema:
        if isinstance(output_field, MergedField):
            if output_field.collision is None:
                # Row unification: every parent identity is preserved.
                for parent in output_field.parents:
                    surviving_identities.add(parent.field.field_identity)
            else:
                winner = output_field.winning_parent()
                surviving_identities.add(winner.field.field_identity)
                for parent in output_field.parents:
                    if parent.input_index == winner.input_index:
                        continue
                    superseded.setdefault(parent.input_index, []).append(
                        parent.field.name
                    )
        else:
            for input_schema in input_schemas:
                matched = _find_by_identity(
                    input_schema, output_field.field_identity
                )
                if matched is None:
                    continue
                surviving_identities.add(matched.field_identity)
                if matched.name != output_field.name:
                    rename_pairs.append((matched.name, output_field.name))
                break

    dropped_names: list[str] = []
    seen_dropped: set[str] = set()
    for input_schema in input_schemas:
        for input_field in input_schema:
            if input_field.field_identity in surviving_identities:
                continue
            if input_field.name in seen_dropped:
                continue
            dropped_names.append(input_field.name)
            seen_dropped.add(input_field.name)

    return (
        tuple(rename_pairs),
        tuple(dropped_names),
        tuple(
            (idx, tuple(names))
            for idx, names in sorted(superseded.items())
        ),
    )


def _find_by_identity(schema: Schema, identity) -> object | None:
    for f in schema:
        if f.field_identity == identity:
            return f
    return None


def _ensure_rewrite_preserves_primary_index(
    input_schemas: tuple[Schema, ...],
    output_schema: Schema,
) -> None:
    """Reject derive identity when rewrite touched the primary IndexField.

    A rewrite that leaves the input's primary ``IndexField`` still an
    ``IndexField`` in the output (same ``FieldIdentity``, possibly
    renamed/retyped) is a safe identity-preserving rewrite — derive
    resolves to ``inherit(input=0)``. A rewrite that demotes or removes
    the primary ``IndexField`` (set_index-style) needs ``mint`` declared
    explicitly so the new identity namespace is intentional.
    """
    if not input_schemas:
        return
    primary = _primary_index_field(input_schemas[0])
    if primary is None:
        return
    for output_field in output_schema:
        if output_field.field_identity == primary.field_identity:
            if isinstance(output_field, IndexField):
                return
            break
    raise ValueError(
        "resolve_derived_transitions: schema=rewrite with "
        "index_identity=derive but the rewrite touched the primary "
        "IndexField (it is no longer an IndexField in the output schema). "
        "Declare index_identity=mint or inherit explicitly."
    )


def _primary_index_field(schema: Schema) -> IndexField | None:
    for f in schema:
        if isinstance(f, IndexField):
            return f
    return None
