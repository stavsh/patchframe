"""Tests for the normalized operator-call lifecycle scaffold."""

from __future__ import annotations

import pandas as pd
import pytest

import patchframe as pf


def _dataset(values: list[int]) -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame({"value": pd.array(values, dtype="Int64")}),
        pf.Schema(fields=(pf.ValueField(name="value", dtype=int),)),
    )


def test_base_operator_lifecycle_runs_hooks_in_order():
    events: list[str] = []

    class _LifecycleOperator(pf.Operator):
        def normalize_call(self, *args, **kwargs):
            events.append("normalize")
            return pf.OperatorCall(
                operator=self,
                args=args,
                kwargs=kwargs,
                variant="toy",
            )

        def resolve_call_transitions(self, call):
            events.append("resolve")
            assert call.variant == "toy"
            return self.transitions

        def run(self, call, transitions):
            events.append("run")
            assert transitions is self.transitions
            assert call.args == ("input",)
            assert call.kwargs["flag"] is True
            return "result"

        def validate_result(self, call, result):
            events.append("validate")
            assert result == "result"

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    result = _LifecycleOperator.instance()("input", flag=True)

    assert result == "result"
    assert events == ["normalize", "resolve", "run", "validate", "effects"]


def test_default_operator_rejects_field_handles_without_custom_normalization():
    left = _dataset([1]).context()
    right = _dataset([2]).context()
    left_value = left.field("value")
    right_value = right.field("value")

    with pytest.raises(TypeError, match="override normalize_call"):
        pf.Operator.instance().normalize_call(left_value, target=right_value)


def test_operator_call_can_record_multiple_reference_contexts():
    left = _dataset([1]).context()
    right = _dataset([2]).context()

    call = pf.OperatorCall(
        operator=pf.Operator.instance(),
        args=("left",),
        kwargs={"target": "right"},
        reference_contexts=(left, right),
    )

    assert call.args == ("left",)
    assert call.kwargs["target"] == "right"
    assert call.reference_contexts == (left, right)


def test_context_effect_advances_context_after_validation():
    initial = _dataset([1])
    updated = _dataset([2])
    ctx = initial.context()
    events: list[str] = []

    class _AdvanceOperator(pf.Operator):
        def normalize_call(self):
            events.append("normalize")
            return pf.OperatorCall(
                operator=self,
                context_effects=(
                    pf.ContextEffect(
                        context=ctx,
                        anchor=initial,
                        effect="advance",
                    ),
                ),
            )

        def run(self, call, transitions):
            events.append("run")
            return updated

        def validate_result(self, call, result):
            events.append("validate")
            assert ctx.dataset is initial
            assert result is updated

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    result = _AdvanceOperator.instance()()

    assert result is updated
    assert ctx.dataset is updated
    assert events == ["normalize", "run", "validate", "effects"]


def test_context_effect_does_not_run_when_validation_fails():
    initial = _dataset([1])
    updated = _dataset([2])
    ctx = initial.context()

    class _FailingValidationOperator(pf.Operator):
        def normalize_call(self):
            return pf.OperatorCall(
                operator=self,
                context_effects=(
                    pf.ContextEffect(
                        context=ctx,
                        anchor=initial,
                        effect="advance",
                    ),
                ),
            )

        def run(self, call, transitions):
            return updated

        def validate_result(self, call, result):
            raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        _FailingValidationOperator.instance()()

    assert ctx.dataset is initial


def test_context_effect_rejects_unknown_effect():
    ctx = _dataset([1]).context()

    with pytest.raises(ValueError, match="advance.*sibling.*none"):
        pf.ContextEffect(context=ctx, effect="replace")


def test_dataset_level_concat_rejects_field_handles():
    left = _dataset([1]).context()
    right = _dataset([2]).context()

    with pytest.raises(TypeError, match="concat: FieldHandle inputs"):
        pf.concat(left.field("value"), right.field("value"))


def test_merge_field_handles_route_to_the_lazy_arm_and_require_bundle_cells():
    # merge now routes field-handle operands to its lazy in-level arm (the
    # duality); that arm requires the handles to point at BundleField cells, so a
    # non-bundle field handle is rejected there rather than wholesale. (Plain
    # Dataset operands still take the eager arm — bundle first for the lazy one.)
    ds = _dataset([1]).context()

    with pytest.raises(TypeError, match="not a BundleField"):
        pf.merge(ds.field("value"), ds.field("value"), ds.field("value"), out="merged")


def test_dataset_operator_runs_through_normalized_lifecycle():
    events: list[str] = []

    class _UnaryLifecycleOperator(pf.DatasetOperator):
        transitions = pf.TransitionPlan(schema=pf.SchemaTransition.preserve())

        def normalize_call(self, *args, **kwargs):
            events.append("normalize")
            return super().normalize_call(*args, **kwargs)

        def resolve_call_transitions(self, call):
            events.append("resolve")
            return super().resolve_call_transitions(call)

        def run(self, call, transitions):
            events.append("run")
            return super().run(call, transitions)

        def apply_table(self, state, **_):
            events.append("apply_table")
            return state.table.copy()

        def validate_result(self, call, result):
            events.append("validate")
            super().validate_result(call, result)

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    ctx = _dataset([1]).context()

    with ctx:
        result = _UnaryLifecycleOperator.instance()()

    assert ctx.dataset is result
    assert events == [
        "normalize",
        "resolve",
        "run",
        "apply_table",
        "validate",
        "effects",
    ]


def test_dataset_operator_validation_failure_does_not_advance_context():
    class _InvalidUnaryOperator(pf.DatasetOperator):
        transitions = pf.TransitionPlan(schema=pf.SchemaTransition.preserve())

        def apply_table(self, state, **_):
            return pd.DataFrame(index=state.table.index)

    initial = _dataset([1])
    ctx = initial.context()

    with ctx:
        with pytest.raises(ValueError, match="missing schema fields"):
            _InvalidUnaryOperator.instance()()

    assert ctx.dataset is initial


def test_dataset_operator_normalize_call_resolves_field_handles_before_run():
    ctx = _dataset([1]).context()
    value = ctx.field("value")

    bind_call = pf.bind_slice.instance().normalize_call(value, value)
    consume_call = pf.consume.instance().normalize_call(value)
    dimensions_call = pf.bind_dimensions.instance(dataset_context=ctx).normalize_call(
        slice_field="clip",
        bindings={"x": (value,)},
    )

    assert bind_call.args == ("value", "value")
    assert bind_call.reference_contexts == (ctx,)
    assert consume_call.args == ("value",)
    assert consume_call.reference_contexts == (ctx,)
    assert dimensions_call.kwargs["bindings"] == {"x": ("value",)}
    assert dimensions_call.reference_contexts == (ctx,)


def test_field_handles_outside_declared_field_inputs_are_rejected():
    ctx = _dataset([1]).context()
    value = ctx.field("value")

    with pytest.raises(TypeError, match="declared field-scoped parameters"):
        pf.bind_slice.instance().normalize_call(value, unexpected=value)


def test_creation_operator_runs_through_normalized_lifecycle_without_advancing_context():
    events: list[str] = []

    class _CreationLifecycleOperator(pf.CreationOperator):
        def normalize_call(self, *args, **kwargs):
            events.append("normalize")
            return super().normalize_call(*args, **kwargs)

        def run(self, call, transitions):
            events.append("run")
            return super().run(call, transitions)

        def generate_source_info(self, *args, **kwargs):
            events.append("source_info")
            return pf.DatasetSourceInfo(
                source_uri="memory://lifecycle",
                source_type="lifecycle",
            )

        def build(self, *args, **kwargs):
            events.append("build")
            table = pd.DataFrame({"value": [1]})
            schema = pf.Schema(fields=(pf.ValueField(name="value", dtype=int),))
            return pf.DatasetState(schema=schema, table=table)

        def validate_result(self, call, result):
            events.append("validate")
            super().validate_result(call, result)

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    ctx = _dataset([0]).context()

    with ctx:
        result = _CreationLifecycleOperator.instance()()

    assert ctx.dataset.table["value"].tolist() == [0]
    assert result.table["value"].tolist() == [1]
    assert events == ["normalize", "run", "source_info", "build", "validate", "effects"]


def test_composition_operator_runs_through_normalized_lifecycle_and_advances():
    events: list[str] = []

    class _CompositionLifecycleOperator(pf.CompositionOperator):
        def normalize_call(self, *args, **kwargs):
            events.append("normalize")
            return super().normalize_call(*args, **kwargs)

        def run(self, call, transitions):
            events.append("run")
            return super().run(call, transitions)

        def apply_schema(self, *states, **kwargs):
            events.append("apply_schema")
            return states[0].schema

        def apply_table(self, *states, **kwargs):
            events.append("apply_table")
            return states[0].table.copy()

        def apply_couplings(self, *states, **kwargs):
            events.append("apply_couplings")
            return states[0].couplings

        def validate_result(self, call, result):
            events.append("validate")
            super().validate_result(call, result)

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    initial = _dataset([1])
    ctx = initial.context()

    with ctx:
        result = _CompositionLifecycleOperator.instance()(ctx.dataset)

    assert ctx.dataset is result
    assert result.table["value"].tolist() == [1]
    assert events == [
        "normalize",
        "run",
        "apply_schema",
        "apply_table",
        "validate",
        "effects",
    ]


def test_plan_operator_runs_through_lifecycle_without_advancing_context():
    events: list[str] = []

    class _PlanLifecycleOperator(pf.PlanOperator):
        required_plan_fields = ("value",)

        def normalize_call(self, *args, **kwargs):
            events.append("normalize")
            return super().normalize_call(*args, **kwargs)

        def resolve_call_transitions(self, call):
            events.append("resolve")
            return super().resolve_call_transitions(call)

        def run(self, call, transitions):
            events.append("run")
            table = pd.DataFrame(
                {"value": [1]},
                index=pd.RangeIndex(1, name="plan_id"),
            )
            schema = pf.Schema(
                fields=(
                    pf.IndexField(name="plan_id"),
                    pf.ValueField(name="value", dtype=int),
                )
            )
            return self.build_plan_dataset(schema=schema, table=table)

        def validate_result(self, call, result):
            events.append("validate")
            super().validate_result(call, result)

        def apply_context_effects(self, call, result):
            events.append("effects")
            super().apply_context_effects(call, result)

    initial = _dataset([1])
    ctx = initial.context()

    with ctx:
        result = _PlanLifecycleOperator.instance()()

    assert ctx.dataset is initial
    assert result.table["value"].tolist() == [1]
    assert events == ["normalize", "resolve", "run", "validate", "effects"]


def test_make_plan_normalizes_target_as_dataset_operand_without_context_effect():
    target = pf.make_from_dataframe(
        pd.DataFrame(
            {"value": [1]},
            index=pd.Index(["a"], name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )
    ctx = target.context()
    handle = ctx.field("item_id")

    call = pf.make_plan.instance().normalize_call(
        handle,
        ["a"],
        source_index_field="source",
        plan_index_name="sample_id",
    )

    assert call.variant == "source_indexed"
    assert call.datasets == (target,)
    assert call.states == (target.state,)
    assert call.args == (["a"],)
    assert call.reference_contexts == (ctx,)
    assert call.context_effects == ()
    assert call.kwargs["source_index_field"] == "source"


def test_make_plan_rejects_field_handle_outside_target_parameter():
    target = pf.make_from_dataframe(
        pd.DataFrame(
            {"value": [1]},
            index=pd.Index(["a"], name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )
    ctx = target.context()

    with pytest.raises(TypeError, match="declared field-scoped parameters"):
        pf.make_plan.instance().normalize_call(target, [ctx.field("item_id")])


def test_concat_normalizes_axis_dispatch_without_own_context_effect():
    left = _dataset([1])
    right = _dataset([2])
    ctx = left.context()

    with ctx:
        call = pf.concat.instance().normalize_call(ctx.dataset, right, axis=1)

    assert call.variant == "columns"
    assert call.datasets == (left, right)
    assert call.context_effects == ()


def test_join_normalize_call_resolves_strategy_into_operator_call():
    left = _dataset([1])
    right = _dataset([1])

    call = pf.join.instance().normalize_call(left, right, on="value", how="left")

    assert call.datasets == (left, right)
    assert isinstance(call.kwargs["strategy"], pf.FieldEqualityJoin)
    assert call.kwargs["strategy"].on == ("value",)
    assert call.kwargs["strategy"].how == "left"
    assert call.context_effects == ()


def test_explode_normalize_call_uses_context_source_and_records_effect():
    source = pf.make_from_dataframe(
        pd.DataFrame(
            {"value": [1]},
            index=pd.Index(["a"], name="item_id"),
        ),
        pf.Schema(
            fields=(
                pf.IndexField(name="item_id"),
                pf.ValueField(name="value", dtype=int),
            )
        ),
    )
    plan = pf.make_plan(source, ["a"])
    plan = pf.assign(plan, value=[2])
    ctx = source.context()

    with ctx:
        call = pf.explode.instance().normalize_call(plan)

    assert call.datasets == (source, plan)
    assert len(call.context_effects) == 1
    assert call.context_effects[0].effect == "advance"
    assert call.context_effects[0].anchor is source


def test_field_handle_inputs_are_explicit_operator_convention():
    # The duality ops have migrated to OperatorSignatures; their field slots come
    # from the signature (the source the normalize-call machinery now reads), and
    # the legacy field_handle_inputs tuple is superseded (empty).
    assert pf.bind_slice.signature.field_slots() == ("slice_field", "data_field")
    assert pf.bind_slice.field_handle_inputs == ()
    assert pf.bind_materialize.signature.field_slots() == ("field",)
    assert pf.bind_materialize.field_handle_inputs == ()
    # bind_dimensions resolves handles nested in `bindings`; `slice_field` is the
    # produced field (a FieldOutput), not a field-input slot.
    assert pf.bind_dimensions.signature.field_slots() == ("bindings",)
    assert pf.bind_dimensions.signature.output_slots() == ("slice_field",)
    assert pf.bind_dimensions.field_handle_inputs == ()
    assert pf.consume.field_handle_inputs == ("target",)
    assert pf.make_plan.field_handle_inputs == ("target",)
    assert pf.window_expansion_plan.field_handle_inputs == ("field", "bindings")
    assert pf.concat.field_handle_inputs == ()
