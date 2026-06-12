"""patchframe.ops.bundle

The eager wide-record Bundle and its flat<->bundle morphisms, as operators.

A blocking operator (``merge``, ``concat``, ``join``) is not per-row-independent,
so its only honest deferred representation is a one-row "wide record" dataset
whose ``BundleField`` cells hold the input datasets, plus an ``ApplyOperator``
coupling that records the operator to run between those cells.

The flat<->bundle morphisms (lazy-and-bundle.md §3) are first-class operators so
they pick up transition declarations, the normalize-call lifecycle, FieldHandle
handling, and context propagation like every other transform:

- ``bundle(left=..., right=...)`` — the lift: combine datasets into a carrier.
- ``extract(b.field("left"))`` / ``extract(b, "left")`` — pull one cell out.
- ``flatten(b)`` — the total space: ``concat_rows`` of every cell.

The terminal is internal ``_collect`` (``consume`` then ``extract``); the
user-facing exit bridge is the nullary ``FieldHandle.collect()`` method, which
dispatches to it (lazy-and-bundle.md §1).

Cell-eager only: a ``BundleField`` cell holds a ``Dataset``. The lazy
``DatasetAccessor`` cell, the tall/collection substrate, and the streaming
executor are out of scope here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from patchframe.dataset.couplings import (
    ApplyOperator,
    CallSpec,
    CouplingSet,
    warn_if_unpicklable,
)
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import BundleField, IndexField
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import CompositionOperator, DatasetOperator, OperatorCall
from patchframe.ops.signature import DatasetReturn, FieldInput
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)

#: Index name minted for a wide-record bundle's (throwaway) base row identity.
BUNDLE_INDEX_NAME = "bundle_id"

#: Default output column name for a deferred operator's result cell.
DEFAULT_OUTPUT_FIELD = "result"


def _carrier_state(
    cells: dict[str, Dataset],
    *,
    output: str | None = None,
) -> tuple[Schema, pd.DataFrame]:
    """Build the schema and one-row table for a wide-record bundle carrier."""

    if not cells:
        raise ValueError("a bundle requires at least one dataset.")

    index = pd.Index([0], name=BUNDLE_INDEX_NAME)
    table = pd.DataFrame(index=index)
    for name, dataset in cells.items():
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"bundle cell {name!r} must be a Dataset, got {type(dataset).__name__}."
            )
        cell = pd.Series(index=index, dtype=object)
        cell.iloc[0] = dataset
        table[name] = cell
    if output is not None:
        table[output] = pd.Series(index=index, dtype=object)  # unmaterialized cell

    fields = [IndexField(name=BUNDLE_INDEX_NAME)]
    fields.extend(BundleField(name=name) for name in cells)
    if output is not None:
        fields.append(BundleField(name=output))
    return Schema(fields=tuple(fields)), table


class bundle(CompositionOperator):
    """Lift datasets into a one-row wide-record ``BundleField`` carrier.

    The explicit lift (lazy-and-bundle.md §3): each input becomes a
    ``BundleField`` cell on a single base row. Positional inputs are named
    ``cell_0``, ``cell_1``, ...; keyword inputs keep their name — use those for
    operators that read the bundle by column (``merge(b.field("left"), ...)``).
    No couplings are added; apply operators in-level afterwards. The output is a
    sibling artifact — an eager construction that forks rather than advancing a
    cursor.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        sources=SourcesTransition.clear(),
        index_identity=IndexIdentityTransition.mint(),
    )
    cardinality = Cardinality.UNKNOWN
    advances_dataset_context = False

    def normalize_call(self, *datasets: Dataset, **named_datasets: Dataset) -> OperatorCall:
        cells: dict[str, Dataset] = {}
        for position, dataset in enumerate(datasets):
            cells[f"cell_{position}"] = dataset
        for name, dataset in named_datasets.items():
            if name in cells:
                raise ValueError(f"{self.name}: duplicate cell name {name!r}.")
            cells[name] = dataset
        if not cells:
            raise ValueError(f"{self.name} requires at least one dataset.")
        for name, dataset in cells.items():
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f"{self.name} cell {name!r} must be a Dataset, "
                    f"got {type(dataset).__name__}."
                )
        return OperatorCall(
            operator=self,
            datasets=tuple(cells.values()),
            states=tuple(dataset.state for dataset in cells.values()),
            kwargs={"cell_names": tuple(cells.keys())},
            variant="bundle",
        )

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        cell_names: tuple[str, ...] = call.kwargs["cell_names"]
        cells = dict(zip(cell_names, call.datasets, strict=True))
        schema, table = _carrier_state(cells)
        return Dataset(state=DatasetState(schema=schema, table=table))

    # ``run`` builds the carrier directly (it needs the Dataset facades, which
    # the aspect hooks do not receive); the aspect hooks stay unimplemented.
    def apply_schema(self, *states: DatasetState, **kwargs: Any) -> Schema:
        raise NotImplementedError(f"{self.name} builds its carrier in run().")

    def apply_table(self, *states: DatasetState, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} builds its carrier in run().")

    def apply_couplings(self, *states: DatasetState, **kwargs: Any) -> CouplingSet:
        raise NotImplementedError(f"{self.name} builds its carrier in run().")


class extract(DatasetOperator):
    """Pull one ``BundleField`` cell out of a one-row bundle (inverse of ``bundle``).

    ``extract(b, "left")`` (eager) or ``extract(b.field("left"))`` (the handle
    form — the first real ``FieldHandle``->``BundleField`` operand). The field
    name is inferred when the bundle has exactly one ``BundleField`` column.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.custom(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.custom(),
        sources=SourcesTransition.custom(),
        index_identity=IndexIdentityTransition.custom(),
    )
    cardinality = Cardinality.UNKNOWN
    field = FieldInput(field_type=BundleField)
    returns = DatasetReturn()  # an exit bridge: a handle operand still yields a Dataset

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        bundle_ds = call.datasets[0]
        name = call.args[0] if call.args else call.kwargs.get("field")
        return _extract_cell(bundle_ds, name)


class flatten(DatasetOperator):
    """Collapse a bundle to its total space: ``concat_rows`` of every cell.

    Gathers all materialized ``BundleField`` cells (across every row and column)
    and row-stacks them; null (unmaterialized) cells are skipped. Distinct from
    the terminal, which runs a deferred operator.
    """

    transitions = TransitionPlan(
        schema=SchemaTransition.custom(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.custom(),
        sources=SourcesTransition.custom(),
        index_identity=IndexIdentityTransition.custom(),
    )
    cardinality = Cardinality.UNKNOWN

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        return _flatten_cells(call.datasets[0])


def build_apply_bundle(
    operator: Any,
    *,
    inputs: dict[str, Dataset],
    output: str = DEFAULT_OUTPUT_FIELD,
    params: dict[str, Any] | None = None,
) -> Dataset:
    """Build the fused lift+apply leaf form: a carrier plus an ``ApplyOperator``.

    The 1% convenience that bundles ``inputs`` and records the deferred
    ``operator`` in one step. The 99% path is ``bundle(...)`` then an in-level
    operator call on the bundle's ``BundleField`` columns.
    """

    if not inputs:
        raise ValueError("build_apply_bundle requires at least one input dataset.")
    if output in inputs:
        raise ValueError(f"build_apply_bundle: output {output!r} collides with an input name.")

    schema, table = _carrier_state(dict(inputs), output=output)
    call = CallSpec(operator=operator, kwargs=dict(params or {}))
    warn_if_unpicklable(call)
    coupling = ApplyOperator(inputs=tuple(inputs.keys()), output=output, call=call)
    return Dataset(
        state=DatasetState(schema=schema, table=table, couplings=CouplingSet((coupling,)))
    )


def defer_in_level(
    operator: Any,
    *handles: Any,
    out: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """In-level lazy apply: record ``operator`` as a coupling on a bundle carrier.

    The 99% lazy path. ``handles`` are ``FieldHandle``s to ``BundleField`` columns
    of one carrier (the user bundled first). This records an ``ApplyOperator``
    producing a fresh ``BundleField`` column named ``out``, advances the carrier's
    context, and returns a handle to ``out`` — the chaining point. The deferred
    operator runs at ``collect()`` time, per fiber.

    Implemented as a direct carrier extension (extend the schema, add the
    unmaterialized output cell, append the coupling) rather than via ``add_column``
    to keep the ``BundleField`` cells out of column-normalization paths.
    """

    from patchframe.dataset.context import FieldHandle

    if not out:
        raise ValueError("defer_in_level requires an `out` name for the produced field.")
    if not handles:
        raise ValueError("defer_in_level requires at least one bundle field handle.")

    contexts: list[Any] = []
    for handle in handles:
        if not isinstance(handle, FieldHandle):
            raise TypeError("defer_in_level operands must be bundle FieldHandles.")
        if all(handle.dataset_context is not existing for existing in contexts):
            contexts.append(handle.dataset_context)
    if len(contexts) != 1:
        raise ValueError("defer_in_level: handles must share one DatasetContext.")

    context = contexts[0]
    carrier = context.dataset
    input_cols = tuple(handle.name for handle in handles)
    for col in input_cols:
        if not isinstance(carrier.schema.get(col), BundleField):
            raise TypeError(
                f"defer_in_level: {col!r} is not a BundleField; in-level apply "
                "needs bundle cells."
            )
    if out in carrier.schema.names():
        raise ValueError(f"defer_in_level: output {out!r} already exists on the carrier.")

    call = CallSpec(operator=operator, kwargs=dict(params or {}))
    warn_if_unpicklable(call)
    coupling = ApplyOperator(inputs=input_cols, output=out, call=call)
    new_table = carrier.table.copy()
    new_table[out] = pd.Series([pd.NA] * len(carrier), index=carrier.table.index, dtype=object)
    new_carrier = carrier.replace_state(
        schema=carrier.schema.add(BundleField(name=out)),
        table=new_table,
        couplings=carrier.couplings.add(coupling),
    )
    context.adopt(new_carrier)
    return context.field(out)


def _collect(
    bundle_ds: Dataset,
    output: str | None = None,
    *,
    context: Any | None = None,
) -> Dataset:
    """Internal terminal: complete the pending coupling(s) producing ``output``.

    Runs the couplings whose end node is ``output`` via ``consume``, which
    discharges them (consume is literal; lazy-duality-plan.md). When a
    ``context`` is given (the handle's shared cursor), it is advanced to the
    consumed snapshot, so a re-collect finds the work already done.
    Extraction is ``BundleField``-specific: when ``output`` is a
    ``BundleField`` the filled cell *is* a dataset, so it is ``extract``ed and
    returned; otherwise the container dataset (with ``output`` materialized)
    is returned. The user-facing entry point is ``FieldHandle.collect()``.
    """

    from patchframe.ops.builtin.consume import consume

    if output is None:
        output = _infer_apply_output(bundle_ds)
    filled = consume(bundle_ds, output)
    if context is not None:
        context.adopt(filled)
    if isinstance(filled.schema.get(output), BundleField):
        return extract(filled, output)
    return filled


def _extract_cell(bundle_ds: Dataset, name: str | None) -> Dataset:
    if len(bundle_ds) != 1:
        raise ValueError(
            "extract: expects a one-row (wide-record) bundle; use flatten for multi-row."
        )
    if name is None:
        name = _single_bundle_field(bundle_ds)
    field = bundle_ds.schema.get(name)
    if not isinstance(field, BundleField):
        raise TypeError(f"extract: {name!r} is not a BundleField column.")
    cell = bundle_ds.table[name].iloc[0]
    if not isinstance(cell, Dataset):
        raise TypeError(
            f"extract: cell {name!r} is not materialized to a Dataset "
            f"(got {type(cell).__name__}); collect the bundle first."
        )
    return cell


def _flatten_cells(bundle_ds: Dataset) -> Dataset:
    cells: list[Dataset] = []
    for field in bundle_ds.schema:
        if not isinstance(field, BundleField):
            continue
        for value in bundle_ds.table[field.name]:
            if isinstance(value, Dataset):
                cells.append(value)
            elif not pd.isna(value):
                raise TypeError(
                    f"flatten: cell in {field.name!r} is not a Dataset "
                    f"(got {type(value).__name__})."
                )
    if not cells:
        raise ValueError("flatten: bundle has no materialized BundleField cells.")
    if len(cells) == 1:
        return cells[0]

    from patchframe.ops.builtin.concat import concat_rows

    return concat_rows(*cells)


def _single_bundle_field(bundle_ds: Dataset) -> str:
    names = [field.name for field in bundle_ds.schema if isinstance(field, BundleField)]
    if len(names) != 1:
        raise ValueError(
            f"extract: specify name; bundle has {len(names)} BundleField columns."
        )
    return names[0]


def _infer_apply_output(bundle_ds: Dataset) -> str:
    outputs = [
        coupling.output_field()
        for coupling in bundle_ds.couplings.couplings
        if isinstance(coupling, ApplyOperator)
    ]
    if len(outputs) != 1:
        raise ValueError(
            f"collect: bundle has {len(outputs)} ApplyOperator couplings; "
            "pass output= to select one."
        )
    return outputs[0]
