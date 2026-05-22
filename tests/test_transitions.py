"""Tests for the operator transition vocabulary and resolution."""

import pytest

import patchframe as pf
from patchframe.ops.transitions import (
    AccessorsTransition,
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
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


def test_transition_factories_produce_expected_modes():
    assert SchemaTransition.preserve().mode == "preserve"
    assert SchemaTransition.extend().mode == "extend"
    assert SchemaTransition.narrow().mode == "narrow"
    assert SchemaTransition.construct().mode == "construct"
    assert SchemaTransition.infer().mode == "infer"
    assert SchemaTransition.rewrite(mapping={"a": "b"}).mapping == {"a": "b"}
    assert TableTransition.construct().mode == "construct"
    assert CouplingsTransition.clear().mode == "clear"
    assert SourcesTransition.union().mode == "union"
    assert IndexIdentityTransition.inherit(input="plan").input == "plan"
    assert IndexIdentityTransition.inherit(input=2).input == 2
    assert AccessorsTransition.preserve().mode == "preserve"


def test_transition_rejects_unknown_mode():
    with pytest.raises(ValueError):
        SchemaTransition(mode="bogus")
    with pytest.raises(ValueError):
        # 'derive' is a couplings mode, not a table mode
        TableTransition(mode="derive")
    with pytest.raises(ValueError):
        # 'preserve' is no longer a couplings mode
        CouplingsTransition(mode="preserve")


def test_transition_plan_with_replaces_aspects():
    plan = TransitionPlan()
    refined = plan._with(schema=SchemaTransition.rewrite(mapping={"x": "y"}))
    assert refined.schema.mode == "rewrite"
    assert refined.schema.mapping == {"x": "y"}
    # the original plan is unchanged
    assert plan.schema.mode == "infer"
    # untouched aspects carry over
    assert refined.table.mode == "mutate"
    assert refined.couplings.mode == "derive"


def test_resolve_transitions_default_returns_class_plan():
    op = pf.where.instance()
    assert op.resolve_transitions() is op.transitions


def test_rename_resolve_transitions_injects_mapping():
    op = pf.rename.instance()
    resolved = op.resolve_transitions(None, {"old": "new"})
    assert resolved.schema.mode == "rewrite"
    assert resolved.schema.mapping == {"old": "new"}


def test_operator_cardinality_declarations():
    assert pf.where.cardinality is Cardinality.FILTER
    assert pf.drop.cardinality is Cardinality.PRESERVE
    assert pf.keep.cardinality is Cardinality.PRESERVE
    assert pf.rename.cardinality is Cardinality.PRESERVE
    assert pf.explode.cardinality is Cardinality.EXPAND


def test_operator_transition_declarations():
    assert pf.where.transitions.schema.mode == "preserve"
    assert pf.drop.transitions.schema.mode == "narrow"
    assert pf.keep.transitions.schema.mode == "narrow"
    assert pf.rename.transitions.schema.mode == "rewrite"
    assert pf.add_column.transitions.schema.mode == "extend"
    assert pf.bind_dimensions.transitions.schema.mode == "extend"
    assert pf.set_index.transitions.index_identity.mode == "mint"
    assert pf.join.transitions.couplings.mode == "clear"
    assert pf.merge.transitions.couplings.mode == "union"
