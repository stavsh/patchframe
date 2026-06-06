"""Tests for the lazy/eager context-propagation law.

The rule (lazy-and-bundle.md / lazy-duality-plan.md):

- A **lazy** operation (handle operand) always **propagates** the
  ``DatasetContext`` — one context threads the chain forward, so the result
  shares it with the operands and handles minted off the facade follow it.
- An **eager** operation (``Dataset`` operand) **forks** — a new facade gets its
  own fresh context; the input's context is untouched.
- The explicit ``with ctx:`` cursor is the opt-in that additionally advances an
  eager op's ambient context.
"""

from __future__ import annotations

import pandas as pd

import patchframe as pf


def _dataset() -> pf.Dataset:
    return pf.make_from_dataframe(
        pd.DataFrame(
            {
                "value": pd.array([1, 2], dtype="Int64"),
                "value2": pd.array([3, 4], dtype="Int64"),
            }
        ),
        pf.Schema(
            fields=(
                pf.ValueField(name="value", dtype=int),
                pf.ValueField(name="value2", dtype=int),
            )
        ),
    )


class _NoopFieldOp(pf.DatasetOperator):
    """A schema/table-preserving op that accepts a single field-handle operand.

    Used to exercise the handle-operand (lazy-arm) path without depending on any
    builtin's field-type validation.
    """

    transitions = pf.TransitionPlan(
        schema=pf.SchemaTransition.preserve(),
        table=pf.TableTransition.preserve(),
    )
    field_handle_inputs = ("target",)


def test_handle_operand_op_propagates_the_context():
    ds = _dataset()
    handle = ds.field("value")
    context = handle.dataset_context
    assert context.dataset is ds

    result = _NoopFieldOp.instance()(ds.field("value"))

    # One context threaded forward to the result — propagation, not a fork.
    assert context.dataset is result
    assert context.dataset is not ds
    # Handles minted off the facade share that same, now-advanced context.
    assert ds.field("value").dataset_context is context
    # The pre-op handle is live and resolves against the propagated snapshot.
    assert handle.resolve().name == "value"


def test_dataset_operand_op_forks_the_context():
    ds = _dataset()

    result = pf.drop(ds, ["value2"])  # Dataset operand → eager → fork

    input_context = ds.field("value").dataset_context
    result_context = result.field("value").dataset_context

    assert input_context is not result_context  # forked, independent contexts
    assert input_context.dataset is ds  # input context was not advanced
    assert result_context.dataset is result


def test_explicit_cursor_advances_through_an_eager_op():
    ds = _dataset()
    ctx = ds.context()
    assert ctx.dataset is ds

    with ctx:
        result = pf.drop(ds, ["value2"])  # eager, but the ambient cursor opts in

    assert ctx.dataset is result
