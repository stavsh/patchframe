"""patchframe.ops.builtin.link"""

from __future__ import annotations

from typing import Any

from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import (
    BundleField,
    DataField,
    DimensionedSliceField,
    ForeignIndexField,
    IndexColumnField,
    IndexField,
)
from patchframe.dataset.identity import primary_index_identity
from patchframe.dataset.schema import Schema
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


class link(Operator):
    """Type a value column as a reference into another dataset's index.

    ``link(ds, target, field)`` upgrades ``field`` from a value column to a
    ``ForeignIndexField`` carrying the target's ``IndexIdentity`` — the formal
    entry point into the plan language (join-dimensions-identity.md §4): the
    moment "this column references that dataset" becomes typed rather than
    conventional. It is the dual of ``set_index`` (which downgrades a primary
    index to an ``IndexColumnField`` keeping its old identity).

    Labels are validated ⊆ the target index; ``allow_dangling=True`` permits
    labels (including nulls) with no target row. The table is untouched — link
    is a schema-typing operation; the column keeps its values, dtype, and
    field identity (a retype, not a new field).

    Operand-dispatch law: a ``FieldHandle`` operand selects the lazy arm.
    ``link`` rewrites schema, so its lazy form lifts onto a ``BundleField``
    carrier — bundle both sides, then
    ``link(b.field("ds"), b.field("target"), field, out=...)`` records the
    deferred link and returns the chaining handle. ``field`` is replay data
    resolved against the (possibly deferred) ``ds`` at run time, never a
    handle operand — hence ``ParamInput``, like ``set_index``'s ``field`` and
    ``join``'s ``on=``. Dataset operands precede it so ``ApplyOperator``
    replays cells positionally.

    Linking is inherently post-composition: identity is minted at creation
    boundaries, so the target namespace does not exist when the referencing
    dataset is created. That ordering is consistent with the identity model,
    not a wart.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.rewrite(),
        table=TableTransition.preserve(),
        couplings=CouplingsTransition.derive(),
        sources=SourcesTransition.inherit(),
        index_identity=IndexIdentityTransition.inherit(),
    )
    cardinality = Cardinality.PRESERVE
    per_row_independent = PerRowIndependence.INDEPENDENT
    ds = DatasetInput()
    target = DatasetInput()
    field = ParamInput()
    out = FieldOutput()
    returns = FieldReturn()

    def __call__(
        self,
        ds: Dataset | Any = MISSING,
        target: Dataset | Any = MISSING,
        field: str | None = None,
        *,
        allow_dangling: bool = False,
        out: str | None = None,
    ) -> Any:
        # ``out`` flows through to the interpreter: with bundle handle operands
        # it names the deferred result cell; the eager path ignores it.
        return Operator.__call__(
            self, ds, target, field, allow_dangling=allow_dangling, out=out
        )

    def normalize_call(
        self,
        ds: Dataset | Any = MISSING,
        target: Dataset | Any = MISSING,
        field: str | None = None,
        *,
        allow_dangling: bool = False,
        out: str | None = None,
    ) -> OperatorCall:
        self._assert_field_handles_allowed(ds, target, field)
        if not isinstance(ds, Dataset) or not isinstance(target, Dataset):
            raise TypeError(f"{self.name} requires a Dataset and a target Dataset.")
        if not isinstance(field, str) or not field:
            raise TypeError(f"{self.name} requires `field`: the reference column name.")
        return OperatorCall(
            operator=self,
            datasets=(ds, target),
            kwargs={"field": field, "allow_dangling": bool(allow_dangling)},
        )

    def run(self, call: OperatorCall, _) -> Dataset:
        ds, target = call.datasets
        name: str = call.kwargs["field"]
        allow_dangling: bool = call.kwargs["allow_dangling"]

        old_field = self._validate_link_field(ds, name)
        if not allow_dangling:
            labels = ds.table[name]
            dangling = set(labels.unique()) - set(target.table.index)
            if dangling:
                sample = ", ".join(sorted(repr(label) for label in dangling)[:5])
                raise ValueError(
                    f"{self.name}: {name!r} labels not present in the target index: "
                    f"{sample}. Pass allow_dangling=True to permit them."
                )

        new_field = ForeignIndexField(
            name=old_field.name,
            dtype=old_field.dtype,
            nullable=old_field.nullable,
            metadata=old_field.metadata,
            index_identity=primary_index_identity(target.state),
            field_identity=old_field.field_identity,
        )
        schema = Schema(
            fields=tuple(
                new_field if existing.name == name else existing
                for existing in ds.schema
            )
        )
        return ds.replace_state(schema=schema)

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)

    def _validate_link_field(self, ds: Dataset, name: str):
        if not ds.schema.has(name):
            raise ValueError(f"{self.name}: field {name!r} is not in the schema.")
        field = ds.schema.get(name)
        if isinstance(field, IndexField):
            raise TypeError(
                f"{self.name}: {name!r} is the primary index; a dataset's own row "
                "identity cannot be a foreign reference (set_index first)."
            )
        if isinstance(field, ForeignIndexField):
            raise TypeError(f"{self.name}: {name!r} is already a ForeignIndexField.")
        if isinstance(
            field, (IndexColumnField, DataField, DimensionedSliceField, BundleField)
        ):
            raise TypeError(
                f"{self.name}: {name!r} is a {type(field).__name__}; link expects a "
                "label-valued column (ValueField or DimensionField)."
            )
        return field
