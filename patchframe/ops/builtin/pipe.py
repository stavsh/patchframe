"""patchframe.ops.builtin.pipe"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import pandas as pd

from patchframe.dataset.coupling_engine import CouplingEngine
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.identity import (
    maybe_primary_index_field,
    mint_primary_index_identity,
    primary_index_identity,
    with_primary_index_identity,
)
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.base import DatasetOperator, OperatorCall
from patchframe.ops.signature import DatasetInput, ParamInput
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

#: The conservative class-level declaration. An escape hatch promises nothing
#: structurally until the call declares it (design-constraints §2.4: static
#: declarations stay conservative, ``resolve_transitions`` refines per call).
#: ``schema=custom`` is exactly "input lineage exists but the transform is not
#: one the structural vocabulary captures" — the documented total escape, here
#: made callable.
_CONSERVATIVE = TransitionPlan(
    schema=SchemaTransition.custom(),
    index_identity=IndexIdentityTransition.mint(),
    couplings=CouplingsTransition.clear(),
    sources=SourcesTransition.inherit(),
)

#: The friendly default when a call passes no ``transitions=``: the safe, common
#: shape — a row/value transform leaving the schema intact and keeping rows
#: inside the input's identity namespace (filter, sort, recompute). Safe because
#: the post-hoc check is fail-loud: a fn that violates it raises, it never
#: silently corrupts.
_DEFAULT = TransitionPlan(
    schema=SchemaTransition.preserve(),
    index_identity=IndexIdentityTransition.inherit(),
    couplings=CouplingsTransition.derive(),
    sources=SourcesTransition.inherit(),
)

#: The inferred default when ``transitions=`` is omitted but a ``schema=`` is
#: given: handing over an output schema means "rebuild to it" — a construct,
#: minting fresh row identity, with provenance still carried forward.
_CONSTRUCT_DEFAULT = TransitionPlan(
    schema=SchemaTransition.construct(),
    table=TableTransition.construct(),
    index_identity=IndexIdentityTransition.mint(),
    couplings=CouplingsTransition.clear(),
    sources=SourcesTransition.inherit(),
)

_SCHEMA_MODES = frozenset({"preserve", "construct"})
_IDENTITY_MODES = frozenset({"preserve", "inherit", "mint"})
_COUPLINGS_MODES = frozenset({"derive", "inherit", "clear"})
_SOURCES_MODES = frozenset({"inherit", "clear"})


class pipe(DatasetOperator):
    """Run a ``Dataset -> DataFrame`` function, re-validated against a declared transition.

    The **constrained** ``.table`` escape: the sanctioned alternative to reading
    ``Dataset.table`` and rebuilding with ``make_from_dataframe`` — which mints a
    parentless dataset, silently dropping source provenance, couplings, and row
    identity, and validating the result against nothing but a hand-written
    schema. ``pipe`` is instead an **anonymous operator**: the caller supplies
    the ``apply_table`` (the ``fn`` — it receives the input ``Dataset`` and
    returns its new table) *and declares the structural effect* through the
    ordinary ``TransitionPlan`` vocabulary; ``pipe`` runs the standard operator
    lifecycle and **re-validates the returned table against that declaration**,
    failing loud on a lie.

    That makes "use the escape" and "grow an operator" one continuum at
    different commitment levels: a built-in operator is a *named, tested,
    reusable* transition declaration with a fixed ``apply_table``; ``pipe`` is an
    *inline, anonymous, per-call* one. ``pipe`` is the **floor** (always strictly
    safer than ``.table`` + ``make_from_dataframe``); a named operator is the
    **ceiling** (the home for a recurring, vectorizable pattern). Reach for
    ``pipe`` when the transform is one-off, dataset-specific, or genuinely not in
    the vocabulary yet; promote it to a named operator when it recurs.

    Read vs transform. ``.table`` for **inspection** (``ds.table[col].mean()``)
    or inside a ``map_fields`` fiber-reduce stays sanctioned. ``.table`` as a
    **transform** surface — filter / reshape / resolve → rebuild — is the smell
    ``pipe`` replaces.

    Declaring the effect
    --------------------
    The structural effect is one ``transitions=TransitionPlan(...)`` — but for
    the common cases it is **inferred from whether you pass a ``schema=``**, so
    most calls need no ``transitions`` at all. There is no ``cardinality``
    argument (a ``pipe``'s cardinality genuinely varies per call, so the class
    stays ``UNKNOWN`` and the *index-identity* mode carries the row check). Two
    coherent shapes:

    - **No ``schema=`` → preserve / inherit** — a row/value transform (filter,
      sort, recompute). The output schema is **copied from the input**; output
      row labels must be a subset of the input index (rows stay in the input
      identity namespace). Couplings ``derive``, sources ``inherit``. The
      no-argument default (``pf.pipe(ds, fn)``).
    - **A ``schema=`` → construct / mint** — a rebuild (the ``attribute()``
      case). The output schema is the one you pass; it is checked with
      ``schema.validate_table`` against the returned frame, a fresh row identity
      is minted, couplings are cleared, and sources still ``inherit`` — the
      concrete win over ``make_from_dataframe``: provenance is carried forward.

    Pass ``transitions=`` explicitly only to override a default (e.g.
    ``sources=clear``) or to assert ``preserve`` while supplying a ``schema=``.
    ``schema=construct`` requires ``index_identity=mint`` (a construct has no
    input lineage, so its rows are a new namespace); ``schema=preserve`` also
    accepts ``mint`` (same columns, renumbered rows). Other schema modes fail
    loud with a pointer to the right operator (``extend`` → ``map_fields``/
    ``assign``; ``narrow`` → ``keep``/``drop``; ``rewrite`` → ``rename``/
    ``set_index``; ``compose`` → ``concat``/``merge``).

    Parameters
    ----------
    dataset:
        The input dataset.
    fn:
        A ``Callable[[Dataset], pd.DataFrame]`` — plain pandas. It receives the
        input ``Dataset`` (its ``.table`` is an isolated copy, so mutating it
        cannot corrupt the immutable input) and returns the **new table**; a
        non-DataFrame return fails loud. Should be a module-level function, not a
        lambda, for persistence.
    schema:
        Keyword-only. The declared output ``Schema``. Supplying it selects the
        rebuild (construct) shape; omitting it copies the input schema (preserve).
    transitions:
        Keyword-only ``TransitionPlan``. Optional — inferred from ``schema=`` when
        omitted; pass it to override a default or assert a specific mode.

    Usage
    -----
    >>> kept = pf.pipe(ds, drop_late_rows)              # preserve (schema copied)
    >>> attributed = pf.pipe(pairs, resolve, schema=attributed_schema)  # construct
    """

    transitions = _CONSERVATIVE
    cardinality = Cardinality.UNKNOWN
    per_row_independent = PerRowIndependence.UNKNOWN
    dataset = DatasetInput()
    fn = ParamInput()
    schema = ParamInput(default=None)
    # ``transitions=`` arrives as a leftover call kwarg (read in resolve/run); it
    # is deliberately *not* a signature slot, which would shadow the
    # ``transitions`` ClassVar above in the class body.

    def resolve_transitions(self, *args: Any, **kwargs: Any) -> TransitionPlan:
        plan = kwargs.get("transitions")
        if plan is not None:
            if not isinstance(plan, TransitionPlan):
                raise TypeError(
                    f"{self.name}: transitions= must be a TransitionPlan, got "
                    f"{type(plan).__name__}."
                )
            self._check_supported(plan)
            return plan
        # No explicit plan: infer from whether a schema was supplied. Handing
        # over an output schema means "rebuild to it" (construct, mint identity);
        # omitting it means "keep the input schema" (preserve, inherit). Either
        # way the output schema falls out without ceremony; transitions= stays
        # the power knob for sources/couplings control.
        return _CONSTRUCT_DEFAULT if kwargs.get("schema") is not None else _DEFAULT

    def _check_supported(self, plan: TransitionPlan) -> None:
        if plan.schema.mode not in _SCHEMA_MODES:
            raise ValueError(
                f"{self.name}: schema mode {plan.schema.mode!r} is not supported. "
                "Use SchemaTransition.preserve() for a row/value transform, or "
                "construct() with schema= for a rebuild. (extend -> map_fields/"
                "assign; narrow -> keep/drop; rewrite -> rename/set_index; "
                "compose -> concat/merge.)"
            )
        if plan.index_identity.mode not in _IDENTITY_MODES:
            raise ValueError(
                f"{self.name}: index identity mode {plan.index_identity.mode!r} is "
                "not supported; use inherit() (rows stay in the input namespace) "
                "or mint() (a fresh row identity)."
            )
        if plan.couplings.mode not in _COUPLINGS_MODES:
            raise ValueError(
                f"{self.name}: couplings mode {plan.couplings.mode!r} is not "
                "supported; use derive(), inherit(), or clear()."
            )
        if plan.sources.mode not in _SOURCES_MODES:
            raise ValueError(
                f"{self.name}: sources mode {plan.sources.mode!r} is not "
                "supported; use inherit() (carry provenance) or clear()."
            )
        if plan.schema.mode == "construct" and plan.index_identity.mode != "mint":
            raise ValueError(
                f"{self.name}: schema=construct rebuilds the table with no input "
                "lineage, so its rows are a new identity namespace; declare "
                "IndexIdentityTransition.mint()."
            )
        if plan.schema.mode == "construct" and plan.couplings.mode == "derive":
            raise ValueError(
                f"{self.name}: schema=construct has no field-identity lineage, "
                "so couplings=derive is not meaningful. Use "
                "CouplingsTransition.clear(), or explicitly inherit validated "
                "couplings."
            )

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        dataset = call.datasets[0]
        state = dataset.state
        fn, declared_schema = self._call_args(call)

        # ``fn`` receives the input *Dataset* (consistent with the wrapped
        # surface), backed by an isolated table copy so a transform that mutates
        # ``.table`` in place cannot corrupt the immutable input. It returns the
        # new *table* (a DataFrame); pipe re-validates and re-wraps it.
        working = dataset.replace_state(table=state.table.copy())
        new_table = fn(working)
        if not isinstance(new_table, pd.DataFrame):
            raise TypeError(
                f"{self.name}: fn must return its new table as a pandas DataFrame, "
                f"got {type(new_table).__name__}. (fn receives the Dataset and "
                "returns its table; pipe re-wraps it.)"
            )

        output_schema = self._output_schema(state, transitions.schema, declared_schema)
        output_schema, new_table = self._apply_identity(
            state, output_schema, new_table, transitions.index_identity
        )
        couplings = self._resolve_pipe_couplings(state, output_schema, transitions)
        sources = state.sources if transitions.sources.mode == "inherit" else ()
        # validate_result -> _validate_output runs schema.validate_table next:
        # the return-honesty check (the declared output schema must actually
        # describe the returned table), fail-loud.
        return dataset.replace_state(
            schema=output_schema,
            table=new_table,
            couplings=couplings,
            sources=sources,
        )

    @staticmethod
    def _call_args(call: OperatorCall) -> tuple[Callable[..., Any], Schema | None]:
        args = call.args
        kwargs = call.kwargs
        if len(args) > 1:
            raise TypeError("pipe: schema= and transitions= are keyword-only.")
        fn = args[0] if args else kwargs.get("fn")
        if not callable(fn):
            raise TypeError("pipe: fn must be a callable Dataset -> DataFrame.")
        return fn, kwargs.get("schema")

    def _output_schema(
        self,
        state: DatasetState,
        schema_transition: SchemaTransition,
        declared_schema: Schema | None,
    ) -> Schema:
        if schema_transition.mode == "preserve":
            if declared_schema is not None and tuple(
                f.name for f in declared_schema
            ) != tuple(f.name for f in state.schema):
                raise ValueError(
                    f"{self.name}: schema=preserve but the supplied schema= has "
                    "different fields from the input. Omit schema=, or declare "
                    "SchemaTransition.construct()."
                )
            return state.schema
        # construct
        if declared_schema is None:
            raise ValueError(
                f"{self.name}: schema=construct requires an explicit schema= "
                "describing the rebuilt table."
            )
        if not isinstance(declared_schema, Schema):
            raise TypeError(f"{self.name}: schema= must be a Schema.")
        return declared_schema

    def _apply_identity(
        self,
        state: DatasetState,
        output_schema: Schema,
        table: pd.DataFrame,
        identity_transition: IndexIdentityTransition,
    ) -> tuple[Schema, pd.DataFrame]:
        if identity_transition.mode in ("preserve", "inherit"):
            # Rows stay in the input's identity namespace, so their labels must
            # be a subset of the input index (filter/sort/recompute).
            if not table.index.isin(state.table.index).all():
                raise ValueError(
                    f"{self.name}: index_identity=inherit, but the returned table "
                    "has row labels outside the input index. The rows left the "
                    "input identity namespace — declare "
                    "IndexIdentityTransition.mint()."
                )
            table = self._name_index_axis(output_schema, table)
            try:
                identity = primary_index_identity(state)
            except ValueError:
                return output_schema, table
            return with_primary_index_identity(output_schema, identity), table
        # mint
        table = self._name_index_axis(output_schema, table)
        return mint_primary_index_identity(output_schema), table

    @staticmethod
    def _name_index_axis(schema: Schema, table: pd.DataFrame) -> pd.DataFrame:
        """Name the table axis after the schema's IndexField (cf. make_from_dataframe)."""

        index_field = maybe_primary_index_field(schema)
        if index_field is not None and table.index.name != index_field.name:
            table = table.rename_axis(index_field.name)
        return table

    def _resolve_pipe_couplings(
        self,
        state: DatasetState,
        output_schema: Schema,
        transitions: TransitionPlan,
    ) -> CouplingSet:
        mode = transitions.couplings.mode
        if mode == "clear":
            return CouplingSet()
        if mode == "inherit":
            CouplingEngine(schema=output_schema, couplings=state.couplings)
            return state.couplings
        # derive: keep couplings whose referenced fields survived by identity.
        # schema=construct + couplings=derive is rejected in _check_supported:
        # a fresh schema has no trustworthy field-identity lineage to derive
        # through.
        return self._derive_couplings(state, output_schema)


def table_transform(
    fn: Callable[..., pd.DataFrame] | None = None,
    *,
    schema: Schema | None = None,
    transitions: TransitionPlan | None = None,
) -> Any:
    """Bind a ``Dataset -> DataFrame`` fn to a fixed ``pipe`` contract — the named form.

    The reusable sibling of :class:`pipe`. Where ``pipe`` is the inline,
    anonymous form, ``table_transform`` is a decorator that **predefines** the
    structural contract (``schema`` + ``transitions``) once, turning a
    ``Dataset -> DataFrame`` function into a reusable ``Dataset -> Dataset``
    transform. The decorated fn receives the input ``Dataset`` (its ``.table`` is
    an isolated copy) and returns the new table; ``pipe`` re-validates + re-wraps::

        @table_transform(schema=OUT_SCHEMA)        # schema given -> construct
        def rebuild(dataset): ...

        out = rebuild(ds)   # == pipe(ds, <fn>, schema=OUT_SCHEMA)

    Extra positional/keyword arguments are forwarded to the wrapped fn (after the
    dataset), so a *parameterized* transform stays a one-liner::

        @table_transform(schema=OUT_SCHEMA)
        def resolve(dataset, *, lookback_days): ...

        out = resolve(ds, lookback_days=30)

    It is a ``pipe`` *wrapper* (a closure binding the contract), not a
    generated operator class: the ``schema``/``transitions`` are **static**. A
    transform whose schema *varies per call* or is *data-dependent* is out of
    scope — author a real operator (a small jump once you are already declaring
    transitions). Bare ``@table_transform`` (no contract) is the preserve default:
    a reusable row/value transform that copies the input schema.
    """

    def decorate(target: Callable[..., pd.DataFrame]) -> Callable[..., Dataset]:
        @wraps(target)
        def transform(dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
            if args or kwargs:
                def bound(inner_dataset: Dataset) -> pd.DataFrame:
                    return target(inner_dataset, *args, **kwargs)
            else:
                bound = target
            return pipe(dataset, bound, schema=schema, transitions=transitions)

        transform.__pipe_fn__ = target  # the underlying table -> table fn
        return transform

    # Support both @table_transform and @table_transform(schema=..., transitions=...).
    if fn is not None:
        return decorate(fn)
    return decorate
