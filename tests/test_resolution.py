"""Tests for ``resolve_derived_transitions``.

Couplings resolution is the centerpiece — verifying that schema lineage
analysis correctly classifies preserve / rename / drop / compose
scenarios from input/output schemas alone.
"""

from dataclasses import replace

import pytest

from patchframe.dataset.field_composition import (
    ColumnCollisionStrategy,
    FieldParent,
    MergedField,
)
from patchframe.dataset.fields import IndexField, ValueField
from patchframe.dataset.schema import Schema
from patchframe.ops.resolution import resolve_derived_transitions
from patchframe.ops.transitions import (
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)


def _index(name: str = "id") -> IndexField:
    return IndexField(name=name, dtype="int64")


def _value(name: str, dtype: str = "int64") -> ValueField:
    return ValueField(name=name, dtype=dtype)


def _plan(
    *,
    schema=None,
    couplings=None,
    sources=None,
    identity=None,
) -> TransitionPlan:
    return TransitionPlan(
        schema=schema or SchemaTransition.preserve(),
        table=TableTransition.preserve(),
        couplings=couplings or CouplingsTransition.derive(),
        sources=sources or SourcesTransition.derive(),
        index_identity=identity or IndexIdentityTransition.derive(),
    )


# ---------------------------------------------------------------------------
# Couplings — the centerpiece
# ---------------------------------------------------------------------------


def test_couplings_derive_under_preserve_yields_empty_resolution():
    idx, a = _index(), _value("a")
    schema = Schema(fields=(idx, a))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.preserve()),
        input_schemas=(schema,),
        output_schema=schema,
    )
    assert resolved.couplings.mode == "derive"
    assert resolved.couplings.rename_map == ()
    assert resolved.couplings.dropped == ()
    assert resolved.couplings.superseded_per_input == ()


def test_couplings_derive_under_extend_yields_empty_resolution():
    idx, a = _index(), _value("a")
    b = _value("b")  # newly added with fresh identity
    input_schema = Schema(fields=(idx, a))
    output_schema = Schema(fields=(idx, a, b))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.extend()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    # No renames, no drops — surviving identities (idx, a) are unchanged.
    # The new b is irrelevant to coupling derivation.
    assert resolved.couplings.rename_map == ()
    assert resolved.couplings.dropped == ()
    assert resolved.couplings.superseded_per_input == ()


def test_couplings_derive_under_narrow_computes_dropped():
    idx, a = _index(), _value("a")
    b = _value("b")
    input_schema = Schema(fields=(idx, a, b))
    output_schema = Schema(fields=(idx, a))  # b is dropped
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.narrow()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    assert resolved.couplings.dropped == ("b",)
    assert resolved.couplings.rename_map == ()


def test_couplings_derive_under_rewrite_computes_rename_map():
    idx, a = _index(), _value("a")
    a_renamed = replace(a, name="b")  # same FieldIdentity, new name
    input_schema = Schema(fields=(idx, a))
    output_schema = Schema(fields=(idx, a_renamed))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.rewrite()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    assert resolved.couplings.rename_map == (("a", "b"),)
    assert resolved.couplings.dropped == ()


def test_couplings_derive_under_rewrite_handles_multiple_renames():
    idx = _index()
    a = _value("a")
    b = _value("b")
    a_renamed = replace(a, name="aa")
    b_renamed = replace(b, name="bb")
    input_schema = Schema(fields=(idx, a, b))
    output_schema = Schema(fields=(idx, a_renamed, b_renamed))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.rewrite()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    assert set(resolved.couplings.rename_map) == {("a", "aa"), ("b", "bb")}


def test_couplings_derive_under_compose_computes_superseded_for_collision():
    idx0, idx1 = _index(), _index()
    left_label = _value("label")
    right_label = _value("label")  # independent FieldIdentity
    parents = (FieldParent(0, left_label), FieldParent(1, right_label))
    merged = MergedField.over(
        parents, collision=ColumnCollisionStrategy(mode="keep", side="left")
    )
    input_schemas = (
        Schema(fields=(idx0, left_label)),
        Schema(fields=(idx1, right_label)),
    )
    # Output schema carries the MergedField unresolved (compose intermediate).
    output_schema = Schema(fields=(idx0, merged))

    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.compose()),
        input_schemas=input_schemas,
        output_schema=output_schema,
    )
    # Left wins; right.label is superseded on input 1.
    assert resolved.couplings.superseded_per_input == ((1, ("label",)),)


def test_couplings_derive_under_compose_row_unification_preserves_all():
    """A row unification (collision=None) preserves every parent identity."""
    idx0, idx1 = _index(), _index()
    a_left, a_right = _value("a"), _value("a")  # different identities
    a_merged = MergedField.over(
        (FieldParent(0, a_left), FieldParent(1, a_right))
    )
    idx_merged = MergedField.over(
        (FieldParent(0, idx0), FieldParent(1, idx1))
    )
    input_schemas = (
        Schema(fields=(idx0, a_left)),
        Schema(fields=(idx1, a_right)),
    )
    output_schema = Schema(fields=(idx_merged, a_merged))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.compose()),
        input_schemas=input_schemas,
        output_schema=output_schema,
    )
    # Every input identity survives through a row-unify MergedField.
    assert resolved.couplings.superseded_per_input == ()
    assert resolved.couplings.dropped == ()


def test_couplings_derive_non_derive_mode_is_passthrough():
    schema = Schema(fields=(_index(), _value("a")))
    resolved = resolve_derived_transitions(
        _plan(couplings=CouplingsTransition.inherit(input=0)),
        input_schemas=(schema,),
        output_schema=schema,
    )
    assert resolved.couplings.mode == "inherit"
    assert resolved.couplings.input == 0
    # Resolution data is only computed for derive.
    assert resolved.couplings.rename_map == ()


def test_couplings_derive_under_construct_raises():
    schema = Schema(fields=(_index(),))
    with pytest.raises(ValueError, match="construct.*couplings=derive"):
        resolve_derived_transitions(
            _plan(schema=SchemaTransition.construct()),
            input_schemas=(),
            output_schema=schema,
        )


def test_couplings_derive_under_custom_raises():
    schema = Schema(fields=(_index(),))
    with pytest.raises(ValueError, match="custom.*couplings=derive"):
        resolve_derived_transitions(
            _plan(schema=SchemaTransition.custom()),
            input_schemas=(schema,),
            output_schema=schema,
        )


# ---------------------------------------------------------------------------
# Sources resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_mode_factory",
    [
        SchemaTransition.preserve,
        SchemaTransition.extend,
        SchemaTransition.narrow,
        SchemaTransition.rewrite,
    ],
)
def test_sources_derive_resolves_to_inherit_under_unary_modes(schema_mode_factory):
    schema = Schema(fields=(_index(),))
    resolved = resolve_derived_transitions(
        _plan(schema=schema_mode_factory()),
        input_schemas=(schema,),
        output_schema=schema,
    )
    assert resolved.sources.mode == "inherit"
    assert resolved.sources.input == 0


def test_sources_derive_under_compose_resolves_to_compose():
    schema = Schema(fields=(_index(),))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.compose()),
        input_schemas=(schema, schema),
        output_schema=schema,
    )
    assert resolved.sources.mode == "compose"


def test_sources_derive_under_construct_resolves_to_construct():
    schema = Schema(fields=(_index(),))
    # Construct also raises on couplings=derive, so use clear() to isolate.
    resolved = resolve_derived_transitions(
        _plan(
            schema=SchemaTransition.construct(),
            couplings=CouplingsTransition.clear(),
        ),
        input_schemas=(),
        output_schema=schema,
    )
    assert resolved.sources.mode == "construct"


def test_sources_derive_under_custom_raises():
    schema = Schema(fields=(_index(),))
    with pytest.raises(ValueError, match="custom.*sources=derive"):
        resolve_derived_transitions(
            _plan(
                schema=SchemaTransition.custom(),
                couplings=CouplingsTransition.clear(),
            ),
            input_schemas=(schema,),
            output_schema=schema,
        )


# ---------------------------------------------------------------------------
# Index identity resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_mode_factory",
    [
        SchemaTransition.preserve,
        SchemaTransition.extend,
        SchemaTransition.narrow,
    ],
)
def test_identity_derive_resolves_to_inherit_under_unary_modes(schema_mode_factory):
    schema = Schema(fields=(_index(),))
    resolved = resolve_derived_transitions(
        _plan(schema=schema_mode_factory()),
        input_schemas=(schema,),
        output_schema=schema,
    )
    assert resolved.index_identity.mode == "inherit"
    assert resolved.index_identity.input == 0


def test_identity_derive_under_rewrite_preserving_primary_resolves_to_inherit():
    idx = _index()
    idx_renamed = replace(idx, name="row")  # same FieldIdentity, new name
    input_schema = Schema(fields=(idx,))
    output_schema = Schema(fields=(idx_renamed,))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.rewrite()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    assert resolved.index_identity.mode == "inherit"


def test_identity_derive_under_rewrite_demoting_primary_raises():
    """set_index-style: primary IndexField's identity no longer an IndexField."""
    idx = _index()
    idx_demoted = ValueField(
        name="id", dtype="int64", field_identity=idx.field_identity
    )
    new_primary = _index(name="new_id")
    input_schema = Schema(fields=(idx,))
    output_schema = Schema(fields=(new_primary, idx_demoted))
    with pytest.raises(ValueError, match="rewrite.*primary IndexField"):
        resolve_derived_transitions(
            _plan(schema=SchemaTransition.rewrite()),
            input_schemas=(input_schema,),
            output_schema=output_schema,
        )


def test_identity_derive_under_compose_resolves_to_coalesce():
    schema = Schema(fields=(_index(),))
    resolved = resolve_derived_transitions(
        _plan(schema=SchemaTransition.compose()),
        input_schemas=(schema, schema),
        output_schema=schema,
    )
    assert resolved.index_identity.mode == "coalesce"


def test_identity_derive_under_construct_resolves_to_mint():
    schema = Schema(fields=(_index(),))
    resolved = resolve_derived_transitions(
        _plan(
            schema=SchemaTransition.construct(),
            couplings=CouplingsTransition.clear(),
        ),
        input_schemas=(),
        output_schema=schema,
    )
    assert resolved.index_identity.mode == "mint"


def test_identity_derive_under_custom_raises():
    schema = Schema(fields=(_index(),))
    with pytest.raises(ValueError, match="custom.*index_identity=derive"):
        resolve_derived_transitions(
            _plan(
                schema=SchemaTransition.custom(),
                couplings=CouplingsTransition.clear(),
                sources=SourcesTransition.clear(),
            ),
            input_schemas=(schema,),
            output_schema=schema,
        )


# ---------------------------------------------------------------------------
# Pass-through aspects
# ---------------------------------------------------------------------------


def test_non_derive_aspects_are_passed_through_unchanged():
    schema = Schema(fields=(_index(),))
    declared = TransitionPlan(
        schema=SchemaTransition.preserve(),
        table=TableTransition.mutate(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.inherit(input=1),
        index_identity=IndexIdentityTransition.mint(),
    )
    resolved = resolve_derived_transitions(
        declared, input_schemas=(schema,), output_schema=schema
    )
    assert resolved.schema is declared.schema
    assert resolved.table is declared.table
    assert resolved.couplings is declared.couplings
    assert resolved.sources is declared.sources
    assert resolved.index_identity is declared.index_identity


def test_resolve_is_idempotent():
    """Resolving an already-resolved plan returns an equivalent plan."""
    idx, a, b = _index(), _value("a"), _value("b")
    input_schema = Schema(fields=(idx, a, b))
    output_schema = Schema(fields=(idx, a))
    once = resolve_derived_transitions(
        _plan(schema=SchemaTransition.narrow()),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    twice = resolve_derived_transitions(
        TransitionPlan(
            schema=SchemaTransition.narrow(),
            table=TableTransition.preserve(),
            couplings=once.couplings,
            sources=once.sources,
            index_identity=once.index_identity,
        ),
        input_schemas=(input_schema,),
        output_schema=output_schema,
    )
    # Couplings resolution data recomputed on second pass — same answer.
    assert twice.couplings.dropped == once.couplings.dropped
    assert twice.couplings.rename_map == once.couplings.rename_map
    assert twice.sources.mode == once.sources.mode
    assert twice.index_identity.mode == once.index_identity.mode
