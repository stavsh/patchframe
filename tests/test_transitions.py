"""Tests for the operator transition vocabulary and resolution."""

import warnings

import pytest

import patchframe as pf
from patchframe.ops.transitions import (
    AccessorsTransition,
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)


def test_transition_plan_defaults_are_worst_case():
    plan = TransitionPlan()
    assert plan.schema.mode == "infer"
    assert plan.table.mode == "mutate"
    assert plan.couplings.mode == "derive"
    assert plan.sources.mode == "inherit"
    assert plan.index_identity.mode == "preserve"
    assert plan.accessors.mode == "preserve"


def test_transition_plan_defaults_emit_no_warnings():
    """Bare ``TransitionPlan()`` must not emit deprecation warnings even though
    its default schema/identity modes are deprecated factories.

    The class-level defaults still use the deprecated modes for Phase 1
    compatibility, so the dataclass field defaults route through the
    constructor (which does not warn) rather than the deprecated factory
    classmethods (which do).
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        TransitionPlan()
    assert [w for w in record if issubclass(w.category, DeprecationWarning)] == []


def test_transition_factories_produce_expected_modes():
    assert SchemaTransition.preserve().mode == "preserve"
    assert SchemaTransition.extend().mode == "extend"
    assert SchemaTransition.narrow().mode == "narrow"
    assert SchemaTransition.rewrite().mode == "rewrite"
    assert SchemaTransition.compose().mode == "compose"
    assert SchemaTransition.construct().mode == "construct"
    assert SchemaTransition.custom().mode == "custom"
    assert TableTransition.preserve().mode == "preserve"
    assert TableTransition.mutate().mode == "mutate"
    assert TableTransition.construct().mode == "construct"
    assert CouplingsTransition.derive().mode == "derive"
    assert CouplingsTransition.inherit(input=2).input == 2
    assert CouplingsTransition.inherit(input=2).mode == "inherit"
    assert CouplingsTransition.homogeneous().mode == "homogeneous"
    assert CouplingsTransition.construct().mode == "construct"
    assert CouplingsTransition.clear().mode == "clear"
    assert CouplingsTransition.custom().mode == "custom"
    assert SourcesTransition.inherit().mode == "inherit"
    assert SourcesTransition.derive().mode == "derive"
    assert SourcesTransition.compose().mode == "compose"
    assert SourcesTransition.construct().mode == "construct"
    assert SourcesTransition.clear().mode == "clear"
    assert SourcesTransition.custom().mode == "custom"
    assert IndexIdentityTransition.inherit(input="plan").input == "plan"
    assert IndexIdentityTransition.inherit(input=2).input == 2
    assert IndexIdentityTransition.mint().mode == "mint"
    assert IndexIdentityTransition.coalesce().mode == "coalesce"
    assert IndexIdentityTransition.derive().mode == "derive"
    assert IndexIdentityTransition.custom().mode == "custom"
    assert AccessorsTransition.preserve().mode == "preserve"
    assert AccessorsTransition.mutate().mode == "mutate"


def test_transition_rejects_unknown_mode():
    with pytest.raises(ValueError):
        SchemaTransition(mode="bogus")
    with pytest.raises(ValueError):
        # 'derive' is a couplings mode, not a table mode
        TableTransition(mode="derive")
    with pytest.raises(ValueError):
        # 'preserve' is no longer a couplings mode
        CouplingsTransition(mode="preserve")
    with pytest.raises(ValueError):
        # 'union' has been removed from couplings modes
        CouplingsTransition(mode="union")
    with pytest.raises(ValueError):
        # 'union' has been removed from sources modes
        SourcesTransition(mode="union")


def test_schema_infer_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="SchemaTransition.infer"):
        transition = SchemaTransition.infer()
    assert transition.mode == "infer"


def test_index_identity_preserve_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="IndexIdentityTransition.preserve"):
        transition = IndexIdentityTransition.preserve(input=1)
    assert transition.mode == "preserve"
    assert transition.input == 1


def test_couplings_union_raises_with_migration_guidance():
    with pytest.raises(ValueError, match="derive\\(\\).*homogeneous\\(\\)"):
        CouplingsTransition.union()


def test_sources_union_raises_with_migration_guidance():
    with pytest.raises(ValueError, match="derive\\(\\)"):
        SourcesTransition.union()


def test_transition_plan_with_replaces_aspects():
    plan = TransitionPlan()
    refined = plan._with(schema=SchemaTransition.rewrite())
    assert refined.schema.mode == "rewrite"
    # the original plan is unchanged
    assert plan.schema.mode == "infer"
    # untouched aspects carry over
    assert refined.table.mode == "mutate"
    assert refined.couplings.mode == "derive"


def test_resolve_transitions_default_returns_class_plan():
    op = pf.where.instance()
    assert op.resolve_transitions() is op.transitions


def test_operator_cardinality_declarations():
    assert pf.where.cardinality is Cardinality.FILTER
    assert pf.drop.cardinality is Cardinality.PRESERVE
    assert pf.keep.cardinality is Cardinality.PRESERVE
    assert pf.rename.cardinality is Cardinality.PRESERVE
    assert pf.explode.cardinality is Cardinality.EXPAND


def test_operator_per_row_independence_declarations():
    independent = (
        pf.where,
        pf.rename,
        pf.drop,
        pf.keep,
        pf.add_column,
        pf.assign,
        pf.slice_data,
        pf.materialize,
        pf.compose_slice,
        pf.explode,
        pf.concat_rows,
        pf.window_expansion_plan,
    )
    for op in independent:
        assert op.per_row_independent is PerRowIndependence.INDEPENDENT, op.__name__

    # Global / cross-row operators fail the 3-part test on per-row independence.
    for op in (pf.set_index, pf.join, pf.merge):
        assert op.per_row_independent is PerRowIndependence.DEPENDENT, op.__name__

    # Dynamic (consume, from its couplings) / conditional (unaligned
    # concat_columns) operators stay UNKNOWN — conservative routing.
    for op in (pf.consume, pf.concat_columns):
        assert op.per_row_independent is PerRowIndependence.UNKNOWN, op.__name__

    # Default for an undeclared operator is UNKNOWN.
    assert pf.Operator.per_row_independent is PerRowIndependence.UNKNOWN


def test_operator_transition_declarations():
    assert pf.where.transitions.schema.mode == "preserve"
    assert pf.drop.transitions.schema.mode == "narrow"
    assert pf.keep.transitions.schema.mode == "narrow"
    assert pf.rename.transitions.schema.mode == "rewrite"
    assert pf.add_column.transitions.schema.mode == "extend"
    assert pf.compose_slice.transitions.schema.mode == "extend"
    assert pf.set_index.transitions.index_identity.mode == "mint"
    assert pf.join.transitions.couplings.mode == "clear"
    assert pf.join.transitions.sources.mode == "compose"
    assert pf.merge.transitions.couplings.mode == "derive"
    assert pf.merge.transitions.sources.mode == "derive"
