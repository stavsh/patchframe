"""patchframe.ops.builtin.implode"""

from __future__ import annotations

from typing import Any

from patchframe.dataset.dataset import Dataset
from patchframe.ops.base import MISSING, Operator, OperatorCall
from patchframe.ops.signature import DatasetInput, FieldOutput, FieldReturn, ParamInput
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    PerRowIndependence,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)


class implode(Operator):
    """Collapse a dataset's rows into per-group fibers via a membership plan.

    The plan-driven collapse — the cardinality **dual of ``explode``**:
    ``explode(source, plan)`` gathers and *expands* (one flat row per plan row,
    1→N); ``implode(members, plan)`` gathers and *collapses* (one row per
    group, the members nested as a ``BundleField`` fiber, N→1). The ``plan`` is
    a **two-`ForeignIndex` correspondence** — ``member_ref`` (which member) and
    ``group_ref`` (which group) both reference existing datasets; that two-FK
    pair shape is what carries many-to-many (a single-FK plan is only
    one-to-many — ``join-dimensions-identity.md``). A ``match``/``join``
    correspondence is the common one, but any operator emitting that shape
    works (the open seam). A member mapped to several groups **replicates**
    into each — the reason this is more than a column ``partition``.
    ``domain`` makes the grouping total over the group namespace: a group that
    matched nothing gets an empty fiber.

    (It is also the grouped counterpart to ``merge``'s *wide* reading of a
    correspondence — ``merge`` → one flat row per pair, ``implode`` → one row
    per group with members nested — but the precise structural dual is
    ``explode``.)

    Builds a tall bundle (``partition``'s shape), so it is not coupling-able;
    its lazy arm lifts onto a ``BundleField`` carrier like ``merge``/``join``.
    Operands are datasets-first (``members, plan, domain``) so the deferred
    replay binds cells positionally; the refs and ``into`` are keyword-only.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.clear(),
        # The base index is the group namespace, built by the delegated
        # partition (inherits the domain / group_ref target identity); not a
        # plain single-input inherit, so the operator opts out here.
        index_identity=IndexIdentityTransition.custom(),
    )
    cardinality = Cardinality.UNKNOWN
    per_row_independent = PerRowIndependence.DEPENDENT  # a shuffle, like partition
    advances_dataset_context = False
    members = DatasetInput()
    plan = DatasetInput()
    domain = DatasetInput()
    group_ref = ParamInput(default="left_index")
    member_ref = ParamInput(default="right_index")
    into = ParamInput(default="matched")
    out = FieldOutput()
    returns = FieldReturn()

    def __call__(
        self,
        members: Dataset | Any = MISSING,
        plan: Dataset | Any = MISSING,
        domain: Dataset | None = None,
        *,
        group_ref: str = "left_index",
        member_ref: str = "right_index",
        into: str = "matched",
        out: str | None = None,
    ) -> Dataset:
        # Pass ``domain`` positionally only when present — a positional ``None``
        # would be bound as a (null) third operand by the bundle binder.
        operands = (members, plan) if domain is None else (members, plan, domain)
        return Operator.__call__(
            self,
            *operands,
            group_ref=group_ref,
            member_ref=member_ref,
            into=into,
            out=out,
        )

    def normalize_call(
        self,
        members: Dataset | Any = MISSING,
        plan: Dataset | Any = MISSING,
        domain: Dataset | None = None,
        *,
        group_ref: str = "left_index",
        member_ref: str = "right_index",
        into: str = "matched",
        out: str | None = None,
    ) -> OperatorCall:
        if not isinstance(members, Dataset) or not isinstance(plan, Dataset):
            raise TypeError(f"{self.name} requires `members` and `plan` Datasets.")
        if domain is not None and not isinstance(domain, Dataset):
            raise TypeError(f"{self.name}: `domain` must be a Dataset.")
        datasets = (members, plan) if domain is None else (members, plan, domain)
        return OperatorCall(
            operator=self,
            datasets=datasets,
            states=tuple(d.state for d in datasets),
            kwargs={
                "group_ref": group_ref,
                "member_ref": member_ref,
                "into": into,
                "has_domain": domain is not None,
            },
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        from patchframe.ops.builtin.concat import concat_columns
        from patchframe.ops.builtin.explode import explode
        from patchframe.ops.builtin.keep import keep
        from patchframe.ops.builtin.partition import partition

        members, plan = call.datasets[0], call.datasets[1]
        kwargs = dict(call.kwargs)
        group_ref: str = kwargs["group_ref"]
        member_ref: str = kwargs["member_ref"]
        into: str = kwargs["into"]
        domain = call.datasets[2] if kwargs["has_domain"] else None

        # Gather the member rows the plan references (replicating a member that
        # maps to several groups), tag each with its group + member label, then
        # group. The body is the validated explode+partition composition; the
        # operator wraps it with declarations + the lazy arm.
        matched = explode(members, plan, foreign_index_field=member_ref)
        matched = concat_columns(matched, keep(plan, [group_ref, member_ref]))
        return partition(matched, group_ref, domain=domain, into=into)

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)
