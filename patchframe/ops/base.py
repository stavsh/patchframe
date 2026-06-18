"""
patchframe.ops.base

Core callable operator model for patchframe.

Three operator families, each with a distinct __call__ contract:

- DatasetOperator  -- unary dataset-to-dataset transform
- CreationOperator -- creates a dataset from external input
- CompositionOperator -- combines multiple datasets into one

All classes are directly callable for one-shot use; configured instances are
created via ``.instance(**params)``.
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import pandas as pd

from patchframe.data.manager import SourceManager, get_default_manager
from patchframe.data.source import DataSource
from patchframe.dataset.context import FieldHandle
from patchframe.dataset.couplings import CallSpec, Coupling, CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.fields import ForeignIndexField, IndexField
from patchframe.dataset.identity import (
    mint_primary_index_identity,
    primary_index_identity,
    resolve_foreign_index_field,
    with_primary_index_identity,
)
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.dispatch import compute_output_state
from patchframe.ops.signature import (
    DatasetInput,
    DatasetReturn,
    FieldInput,
    FieldOutput,
    FieldReturn,
    OperatorSignature,
    ParamInput,
    SelectionInput,
    SelectionReturn,
)
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

if TYPE_CHECKING:
    from patchframe.dataset.context import DatasetContext

MISSING = object()
ContextEffectKind = Literal["advance", "sibling", "none"]


@dataclass(frozen=True, slots=True)
class Parameter:
    """Declarative configurable operator parameter.

    Parameters declared as class attributes participate in ``.instance(...)``
    configuration and are available through ``bound_params()``.
    """

    default: Any = MISSING
    required: bool = False


@dataclass(frozen=True, slots=True)
class ContextEffect:
    """A post-validation DatasetContext side effect requested by an operator call."""

    context: DatasetContext
    anchor: Dataset | None = None
    effect: ContextEffectKind = "none"

    def __post_init__(self) -> None:
        if self.effect not in {"advance", "sibling", "none"}:
            raise ValueError(
                "ContextEffect.effect must be one of 'advance', 'sibling', or 'none'."
            )


@dataclass(frozen=True, slots=True)
class OperatorCall:
    """Normalized semantic operator call.

    ``OperatorCall`` carries operator-facing operands after call normalization.
    Field-aware operators resolve ``FieldHandle`` values to local field
    selectors before ``run``. The owning contexts remain recorded in
    ``reference_contexts`` so authoring provenance is not lost.
    """

    operator: Operator
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    datasets: tuple[Dataset, ...] = ()
    states: tuple[DatasetState, ...] = ()
    reference_contexts: tuple[DatasetContext, ...] = ()
    context_effects: tuple[ContextEffect, ...] = ()
    variant: str | None = None

    def __post_init__(self) -> None:
        args = tuple(self.args)
        kwargs = MappingProxyType(dict(self.kwargs))
        datasets = tuple(self.datasets)
        states = tuple(self.states) or tuple(dataset.state for dataset in datasets)
        reference_contexts = tuple(self.reference_contexts)
        context_effects = tuple(self.context_effects)

        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", kwargs)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "reference_contexts", reference_contexts)
        object.__setattr__(self, "context_effects", context_effects)

    def spec(self) -> CallSpec:
        """Return the serializable core of this call (the runtime↔persisted bridge).

        Keeps the operator + normalized ``args``/``kwargs`` + ``variant``; drops the
        process-local runtime fields (``datasets``/``states``/``reference_contexts``/
        ``context_effects``), which are never persisted (design-constraints §7).

        The operator is normalized to its *class* — the same canonical reference
        the bundle-defer path records (``defer_in_level(type(self), ...)``).
        Operators are code: the class pickles by reference (stable identity,
        compact), while dual-arm bound params are infra-only (``dataset_context``)
        and behavioral per-call data lives in ``kwargs``.
        """

        operator = self.operator
        operator_ref = operator if isinstance(operator, type) else type(operator)
        return CallSpec(
            operator=operator_ref,
            args=self.args,
            kwargs=dict(self.kwargs),
            variant=self.variant,
        )


class OperatorMeta(ABCMeta):
    """Metaclass: direct class-call execution + parameter/operand collection.

    Collects ``Parameter`` attributes into ``__parameters__`` and operand
    declarations — ``FieldInput`` / ``DatasetInput`` / ``SelectionInput`` plus a
    ``returns`` kind — into a built ``signature``, the dataclass-style
    declaration surface (the same collection pattern, in definition order).
    ``MyOperator(...)`` executes a temporary default-configured instance;
    configured instances come from ``MyOperator.instance(...)``.
    """

    def __new__(mcls, name, bases, namespace):
        params: dict[str, Parameter] = {}
        inputs: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        returns: Any = None
        for base in bases:
            params.update(getattr(base, "__parameters__", {}))
            base_signature = getattr(base, "signature", None)
            if isinstance(base_signature, OperatorSignature):
                inputs.update(base_signature.inputs)
                outputs.update(base_signature.outputs)
                returns = base_signature.returns
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Parameter):
                params[attr_name] = attr_value
            elif isinstance(attr_value, (DatasetInput, FieldInput, SelectionInput, ParamInput)):
                inputs[attr_name] = attr_value
            elif isinstance(attr_value, FieldOutput):
                outputs[attr_name] = attr_value
            elif attr_name == "returns" and isinstance(
                attr_value, (DatasetReturn, FieldReturn, SelectionReturn)
            ):
                returns = attr_value
        namespace["__parameters__"] = params
        if inputs or outputs:
            namespace["signature"] = OperatorSignature(
                inputs=inputs,
                outputs=outputs,
                returns=returns if returns is not None else DatasetReturn(),
            )
        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__()
        return instance.__call__(*args, **kwargs)


class Operator(metaclass=OperatorMeta):
    """Base callable/configurable operator."""

    transitions: ClassVar[TransitionPlan] = TransitionPlan()
    cardinality: ClassVar[Cardinality] = Cardinality.UNKNOWN
    per_row_independent: ClassVar[PerRowIndependence] = PerRowIndependence.UNKNOWN
    dataset_context: ClassVar[Parameter] = Parameter(default=None)
    field_handle_inputs: ClassVar[tuple[str, ...]] = ()
    signature: ClassVar[OperatorSignature | None] = None
    __parameters__: ClassVar[dict[str, Parameter]]

    def __init__(self, **bound_params: Any) -> None:
        self._bound_params = self._normalize_bound_params(bound_params)

    @classmethod
    def instance(cls, **params: Any) -> Self:
        """Return a configured callable operator instance."""
        return type.__call__(cls, **params)

    def with_params(self, **params: Any) -> Self:
        """Return a new instance with updated bound parameters."""
        merged = dict(self._bound_params)
        merged.update(params)
        return type(self).instance(**merged)

    def bound_params(self) -> dict[str, Any]:
        """Return the bound parameter mapping for this instance."""
        return dict(self._bound_params)

    @property
    def name(self) -> str:
        """Human-readable operator name."""
        return type(self).__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Dataset | FieldHandle:
        """Execute this operator: the lazy dual arm, or the eager lifecycle.

        Operand-type dispatch (lazy-duality-plan.md Phase 4). When a *dual*
        operator — one whose signature declares a ``FieldReturn``/
        ``SelectionReturn`` — is handed a ``FieldHandle`` operand, route to the
        lazy arm: a same-level coupling if the operator is ``coupling_able``,
        otherwise a ``BundleField`` carrier. A plain ``Dataset``/name call (no
        handles), an op with no signature, and ``custom`` ops all take the eager
        lifecycle unchanged.
        """

        if self._is_dual_lazy_call(args, kwargs):
            return self._dispatch_lazy(args, kwargs)
        return self._run_eager(*args, **kwargs)

    def _run_eager(self, *args: Any, **kwargs: Any) -> Dataset:
        """Execute the eager normalized-call lifecycle."""

        call = self.normalize_call(*args, **kwargs)
        transitions = self.resolve_call_transitions(call)
        result = self.run(call, transitions)
        self.validate_result(call, result)
        self.apply_context_effects(call, result)
        return result

    # -- Lazy dual arm (operand-type dispatch) ------------------------------

    def coupling_able(self) -> bool:
        """Whether this operator's lazy form can be a same-level coupling.

        The routing gate (lazy-duality-plan.md): a same-level lazy op records a
        *coupling*, and couplings are the add/fill subset — schema preserve or
        extend, one output row per input row, per-row-independent. Everything
        else needs a ``BundleField`` carrier. Derived from existing
        declarations; not a separate capability.
        """

        return (
            self.transitions.schema.mode in {"preserve", "extend"}
            and self.cardinality is Cardinality.PRESERVE
            and self.per_row_independent is PerRowIndependence.INDEPENDENT
        )

    def _is_dual_lazy_call(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
        """Whether this call routes to the lazy arm (a dual op handed handles)."""

        sig = self.signature
        if sig is None or sig.custom:
            return False
        if not isinstance(sig.returns, (FieldReturn, SelectionReturn)):
            return False
        contexts = self._field_handle_contexts(args, kwargs)
        if not contexts:
            return False
        if len(contexts) != 1:
            raise ValueError(f"{self.name}: FieldHandles must share one DatasetContext.")
        return True

    def _dispatch_lazy(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> FieldHandle:
        sig = self.signature
        assert sig is not None  # guarded by _is_dual_lazy_call
        context = self._field_handle_contexts(args, kwargs)[0]
        if self.coupling_able():
            return self._same_level_lazy(sig, context, args, kwargs)
        return self._bundle_lazy(sig, context, args, kwargs)

    def _same_level_lazy(
        self,
        sig: OperatorSignature,
        context: DatasetContext,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Coupling-able lazy arm: record the coupling, return a chaining handle.

        Run the eager lifecycle (it resolves the ambient dataset from the
        handle's context, records the coupling, and advances the cursor), then
        return a handle to the coupling output — the unifying same-level rule
        (output = ``coupling.output_field``), read from the declaration.
        """

        self._run_eager(*args, **dict(kwargs))
        output_names = self._lazy_output_names(sig, args, kwargs)
        if isinstance(sig.returns, SelectionReturn):
            from patchframe.dataset.context import FieldSelection

            return FieldSelection(tuple(context.field(name) for name in output_names))
        return context.field(output_names[0])

    def _bundle_lazy(
        self,
        sig: OperatorSignature,
        context: DatasetContext,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Needs-bundle lazy arm: defer the operator as an ``ApplyOperator``."""

        from patchframe.ops.bundle import defer_in_level

        handles, out, params = self._bind_bundle(sig, args, kwargs)
        return defer_in_level(type(self), *handles, out=out, params=params)

    def _bind_slots(
        self,
        sig: OperatorSignature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        """Bind raw call args to the ordered signature slots.

        Returns ``(bound, out, leftover)``: positional args fill declared slots
        in order (a ``variadic`` slot consumes the rest), keyword args bind by
        slot name, ``out`` is the produced-field slot's value, and ``leftover``
        is the undeclared keyword args (forwarded as params).
        """

        bound: dict[str, Any] = {}
        leftover = dict(kwargs)
        positional = list(args)
        index = 0
        for name, spec in sig.inputs.items():
            if getattr(spec, "variadic", False):
                bound[name] = tuple(positional[index:])
                index = len(positional)
            elif index < len(positional):
                bound[name] = positional[index]
                index += 1
            elif name in leftover:
                bound[name] = leftover.pop(name)
            elif isinstance(spec, ParamInput) and spec.has_default:
                bound[name] = spec.default
        if index < len(positional):
            raise TypeError(
                f"{self.name}: too many positional arguments for the declared operands."
            )
        out: Any = None
        out_slot = sig.output_slot_name()
        if out_slot is not None:
            out = leftover.pop(out_slot, None)
        return bound, out, leftover

    def _bind_bundle(
        self,
        sig: OperatorSignature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[list[Any], Any, dict[str, Any]]:
        """Split a bundle-arm call into ``(handles, out, params)``.

        Operand slots become the deferred ``*handles`` (in declaration order);
        ``ParamInput`` slots and undeclared keyword args become ``params`` (by
        name, so ``ApplyOperator`` can replay them as keywords).
        """

        bound, out, leftover = self._bind_slots(sig, args, kwargs)
        handles: list[Any] = []
        params: dict[str, Any] = dict(leftover)
        for name, spec in sig.inputs.items():
            if name not in bound:
                continue
            value = bound[name]
            if isinstance(spec, (DatasetInput, FieldInput, SelectionInput)):
                if getattr(spec, "variadic", False):
                    handles.extend(value)
                else:
                    handles.append(value)
            elif isinstance(spec, ParamInput):
                params[name] = value
        return handles, out, params

    def _lazy_output_names(
        self,
        sig: OperatorSignature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> list[str]:
        """Resolve the same-level lazy output field name(s) from the declaration.

        Fresh outputs are the caller-supplied ``FieldOutput`` value(s); an
        in-place output is the ``FieldInput`` marked ``output=True``, else the
        sole ``FieldInput``. Equivalent to the coupling's ``output_field`` by
        construction.
        """

        if sig.outputs:
            return [_selector_name(kwargs.get(out_name)) for out_name in sig.outputs]
        field_inputs = [
            (name, spec)
            for name, spec in sig.inputs.items()
            if isinstance(spec, FieldInput)
        ]
        marked = [(name, spec) for name, spec in field_inputs if spec.output]
        chosen = marked or field_inputs
        if len(chosen) != 1:
            raise ValueError(
                f"{self.name}: cannot determine the lazy output field; declare a "
                "FieldOutput or mark one FieldInput output=True."
            )
        bound, _, _ = self._bind_slots(sig, args, kwargs)
        return [_selector_name(bound.get(chosen[0][0]))]

    def normalize_call(self, *args: Any, **kwargs: Any) -> OperatorCall:
        """Return the normalized semantic call for this operator invocation."""

        if self._field_handle_contexts(args, kwargs):
            raise TypeError(
                f"{self.name}: override normalize_call to accept FieldHandle inputs."
            )
        return OperatorCall(
            operator=self,
            args=args,
            kwargs=kwargs,
        )

    def resolve_call_transitions(self, call: OperatorCall) -> TransitionPlan:
        """Resolve the transition plan for a normalized call."""

        return self.resolve_transitions(
            *call.states,
            *call.args,
            **dict(call.kwargs),
        )

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset|FieldHandle:
        """Execute a normalized call. Override in direct ``Operator`` subclasses."""

        raise NotImplementedError(
            f"{self.name}: override run(call, transitions) or __call__."
        )

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        """Validate the operator result before context side effects run."""

    def apply_context_effects(self, call: OperatorCall, result: Any) -> None:
        """Apply post-validation DatasetContext effects requested by ``call``."""

        for context_effect in call.context_effects:
            if context_effect.effect in {"none", "sibling"}:
                continue
            if context_effect.effect != "advance":
                raise ValueError(
                    f"{self.name}: unknown context effect "
                    f"{context_effect.effect!r}."
                )
            if not isinstance(result, Dataset):
                raise TypeError(
                    f"{self.name}: cannot advance DatasetContext from "
                    f"{type(result).__name__} result."
                )
            if (
                context_effect.anchor is not None
                and context_effect.context.dataset is not context_effect.anchor
            ):
                raise ValueError(
                    f"{self.name}: DatasetContext no longer points at the "
                    "snapshot selected during call normalization."
                )
            context_effect.context.adopt(result)

    def resolve_param(self, name: str, value: Any = MISSING) -> Any:
        """Resolve one parameter from call-time value or bound config."""
        if value is not MISSING:
            return value
        if name in self._bound_params:
            return self._bound_params[name]
        spec = self.__parameters__.get(name)
        if spec is None:
            raise KeyError(name)
        if spec.default is not MISSING:
            return spec.default
        if spec.required:
            raise ValueError(f"Required parameter '{name}' is not bound.")
        return None

    def resolve_dataset_context(self):
        """Return the explicitly bound or ambient DatasetContext, if any."""

        bound = self.resolve_param("dataset_context")
        if bound is not None:
            return bound

        from patchframe.dataset.context import get_active_dataset_context

        return get_active_dataset_context()

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        """Return declared parameter names in definition order."""
        return tuple(cls.__parameters__.keys())

    def resolve_transitions(self, *args: Any, **kwargs: Any) -> TransitionPlan:
        """Return the precise transition plan for this call.

        The default returns the class-level declaration. Operators with
        flag-dependent contracts override this to refine the conservative
        class-level plan from call-time inputs.
        """
        return self.transitions

    def _field_handle_contexts(self, *values: Any):
        from patchframe.dataset.context import field_handle_contexts

        return field_handle_contexts(*values)

    def _field_input_slots(self) -> tuple[str, ...]:
        """Ordered field-operand slot names.

        Sourced from the declared ``signature`` when present, else the legacy
        ``field_handle_inputs`` tuple. This is the single seam the signature
        feeds: the rest of the normalize-call machinery is unchanged.
        """

        if self.signature is not None:
            return self.signature.field_slots()
        return self.field_handle_inputs

    def _assert_field_handles_allowed(self, *values: Any):
        contexts = self._field_handle_contexts(*values)
        if contexts and not self._field_input_slots():
            raise TypeError(
                f"{self.name}: FieldHandle inputs are not accepted by this "
                "operator. FieldHandle inputs are reserved for field-scoped "
                "parameters declared by field-aware operators."
            )
        return contexts

    def _reject_unresolved_field_handles(self, *values: Any) -> None:
        if self._field_handle_contexts(*values):
            allowed = ", ".join(self._field_input_slots()) or "<none>"
            raise TypeError(
                f"{self.name}: FieldHandle inputs are only accepted for "
                f"declared field-scoped parameters: {allowed}."
            )

    def _resolve_field_handle_inputs(
        self,
        schema: Schema,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Resolve declared field-scoped handle inputs into local selectors."""

        slots = self._field_input_slots()
        if not slots:
            return args, dict(kwargs)

        from patchframe.dataset.context import resolve_field_selectors

        resolved_args = list(args)
        resolved_kwargs = dict(kwargs)
        for index, name in enumerate(slots):
            if index < len(resolved_args):
                resolved_args[index] = resolve_field_selectors(
                    resolved_args[index],
                    schema,
                    op_name=self.name,
                )
            if name in resolved_kwargs:
                resolved_kwargs[name] = resolve_field_selectors(
                    resolved_kwargs[name],
                    schema,
                    op_name=self.name,
                )
        self._reject_unresolved_field_handles(resolved_args, resolved_kwargs)
        return tuple(resolved_args), resolved_kwargs

    def _resolve_field_handles_for_dataset(
        self,
        dataset: Dataset,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any], tuple[Any, ...]]:
        """Validate and resolve field-handle inputs against an explicit dataset.

        The shared path for operators handed a primary ``dataset`` explicitly that
        also accept field-scoped ``FieldHandle`` operands referring to fields *in*
        it — ``DatasetOperator`` conceptually, and plan ops such as
        ``window_expansion_plan``. Any handles must share one ``DatasetContext``
        whose current snapshot is ``dataset``; declared field-slot handles are then
        resolved to local selectors. Returns ``(args, kwargs, reference_contexts)``.
        """

        contexts = self._assert_field_handles_allowed(*args, *tuple(kwargs.values()))
        if len(contexts) > 1:
            raise ValueError(f"{self.name}: FieldHandles must share one DatasetContext.")
        if contexts and dataset is not contexts[0].dataset:
            raise ValueError(
                f"{self.name}: FieldHandles resolve against their DatasetContext's "
                "current dataset snapshot."
            )
        resolved_args, resolved_kwargs = self._resolve_field_handle_inputs(
            dataset.schema, args, kwargs
        )
        return resolved_args, resolved_kwargs, contexts

    def _normalize_bound_params(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.__parameters__)
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise TypeError(
                f"Unknown operator parameters for {type(self).__name__}: {unknown_names}"
            )

        normalized: dict[str, Any] = {}
        for name, spec in self.__parameters__.items():
            if name in params:
                normalized[name] = params[name]
            elif spec.required and spec.default is MISSING:
                raise ValueError(
                    f"Missing required operator parameter '{name}' for {type(self).__name__}"
                )
            elif spec.default is not MISSING:
                normalized[name] = spec.default
        return normalized


class DatasetOperator(Operator):
    """Unary dataset-to-dataset transformer.

    Subclasses declare which aspects they modify via ``transitions`` and override
    only the corresponding ``apply_*`` hooks. Aspects declared ``"preserve"``
    (the default) or ``"inherit"`` are passed through automatically.

    Subclasses normally override the aspect hooks. Override ``normalize_call``,
    ``run``, or ``validate_result`` for lifecycle customization; override
    ``__call__`` directly only as a full escape hatch.
    """

    def __call__(
        self,
        dataset: Dataset | Any = MISSING,
        *args: Any,
        **kwargs: Any,
    ) -> Dataset:
        return Operator.__call__(self, dataset, *args, **kwargs)

    def _dispatch(
        self,
        dataset: Dataset | Any = MISSING,
        *args: Any,
        **kwargs: Any,
    ) -> Dataset:
        """Compatibility entrypoint for unary subclasses with custom signatures."""

        return Operator.__call__(self, dataset, *args, **kwargs)

    def normalize_call(
        self,
        dataset: Dataset | Any = MISSING,
        *args: Any,
        **kwargs: Any,
    ) -> OperatorCall:
        dataset, args, kwargs, dataset_context, handle_contexts = self._normalize_dataset_call(
            dataset,
            args,
            kwargs,
        )
        args, kwargs = self._resolve_field_handle_inputs(dataset.schema, args, kwargs)
        context_effects: tuple[ContextEffect, ...] = ()
        if dataset_context is not None and dataset is dataset_context.dataset:
            context_effects = (
                ContextEffect(
                    context=dataset_context,
                    anchor=dataset,
                    effect="advance",
                ),
            )
        return OperatorCall(
            operator=self,
            datasets=(dataset,),
            states=(dataset.state,),
            args=args,
            kwargs=kwargs,
            reference_contexts=handle_contexts,
            context_effects=context_effects,
        )

    def _normalize_dataset_call(
        self,
        dataset: Dataset | Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Dataset, tuple[Any, ...], dict[str, Any], Any, tuple[Any, ...]]:
        """Resolve one unary call's dataset without rewriting operator arguments."""

        dataset_context = self.resolve_dataset_context()
        handle_contexts = self._assert_field_handles_allowed(dataset, args, kwargs)
        if len(handle_contexts) > 1:
            raise ValueError(f"{self.name}: FieldHandles must share one DatasetContext.")
        if handle_contexts:
            handle_context = handle_contexts[0]
            if dataset_context is not None and dataset_context is not handle_context:
                raise ValueError(
                    f"{self.name}: FieldHandles belong to a different DatasetContext."
                )
            dataset_context = handle_context

        if isinstance(dataset, Dataset):
            if (
                handle_contexts
                and dataset_context is not None
                and dataset is not dataset_context.dataset
            ):
                raise ValueError(
                    f"{self.name}: FieldHandles resolve against their "
                    "DatasetContext's current dataset snapshot."
                )
            return dataset, args, kwargs, dataset_context, handle_contexts
        if dataset_context is None:
            raise TypeError(
                f"{self.name}: expected a Dataset as the first argument or an "
                "active DatasetContext."
            )
        if dataset is not MISSING:
            args = (dataset, *args)
        return dataset_context.dataset, args, kwargs, dataset_context, handle_contexts

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        dataset = call.datasets[0]
        args = call.args
        kwargs = dict(call.kwargs)
        new_state = compute_output_state(
            self,
            (dataset.state,),
            args,
            kwargs,
            declared_transitions=transitions,
        )
        result = dataset.replace_state(
            schema=new_state.schema,
            table=new_state.table,
            couplings=new_state.couplings,
            sources=new_state.sources,
        )
        return result

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        self._validate_output(result)

    def _apply(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        """Apply this unary operator without DatasetContext side effects."""

        call = replace(
            self.normalize_call(dataset, *args, **kwargs),
            context_effects=(),
        )
        transitions = self.resolve_call_transitions(call)
        result = self.run(call, transitions)
        self.validate_result(call, result)
        return result

    def apply_index_identity(
        self,
        state: DatasetState,
        schema: Schema,
        transition: IndexIdentityTransition,
        *args: Any,
        **kwargs: Any,
    ) -> Schema:
        """Apply the resolved primary index identity transition to ``schema``."""

        mode = transition.mode
        if mode == "preserve":
            try:
                return with_primary_index_identity(schema, primary_index_identity(state))
            except ValueError:
                return schema
        if mode == "mint":
            return mint_primary_index_identity(schema)
        raise ValueError(
            f"{self.name}: a DatasetOperator cannot use index identity mode "
            f"{mode!r}; expected 'preserve' or 'mint'."
        )

    def _resolve_couplings(
        self,
        state: DatasetState,
        transitions: TransitionPlan,
        output_schema: Schema,
        *args: Any,
        **kwargs: Any,
    ) -> CouplingSet:
        """Produce the output coupling set from the resolved transition plan."""
        mode = transitions.couplings.mode
        if mode == "clear":
            return CouplingSet()
        if mode in ("union", "construct"):
            return self.apply_couplings(state, *args, **kwargs)

        # mode == "derive": framework derives surviving couplings from
        # input/output FieldIdentity lineage, then appends any operator-declared
        # new couplings.
        derived = self._derive_couplings(state, output_schema)
        new = self.new_couplings(state, *args, **kwargs)
        if not new:
            return derived
        # Membership by equality, not hashing: a coupling may carry an unhashable
        # payload (e.g. a MapCoupling's CallSpec, whose kwargs is a dict).
        existing = list(derived.couplings)
        additions = tuple(c for c in new if c not in existing)
        return derived.add(*additions) if additions else derived

    def _derive_couplings(
        self,
        state: DatasetState,
        output_schema: Schema,
    ) -> CouplingSet:
        """Derive surviving couplings from input/output FieldIdentity lineage.

        Compare input and output schemas by ``field_identity``: a field whose
        identity reappears in the output under a different name was renamed;
        one whose identity is absent was dropped; others are preserved. Mode-
        agnostic — the same logic handles preserve / extend / narrow / rewrite
        / infer / construct.
        """
        couplings = state.couplings
        if not couplings.couplings:
            return couplings

        output_by_identity = {
            field.field_identity: field.name for field in output_schema
        }
        rename: dict[str, str] = {}
        for input_field in state.schema:
            output_name = output_by_identity.get(input_field.field_identity)
            if output_name is not None and output_name != input_field.name:
                rename[input_field.name] = output_name
        if rename:
            couplings = couplings.rewrite_field_names(rename)

        surviving = set(output_schema.names())
        retained = couplings.retain(surviving)
        dropped = len(couplings.couplings) - len(retained.couplings)
        if dropped:
            warnings.warn(
                f"{self.name}: dropped {dropped} coupling(s) referencing fields "
                "that did not survive the schema transition.",
                UserWarning,
                stacklevel=4,
            )
        return retained

    def new_couplings(
        self,
        state: DatasetState,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Coupling, ...]:
        """Couplings this operator adds, appended to the derived coupling set.

        Override for operators that append couplings (the ``bind_*`` family).
        Returns an empty tuple by default.
        """
        return ()

    def _validate_output(self, dataset: Dataset) -> None:
        """Validate the output dataset. Override to customize or suppress."""
        dataset.schema.validate_table(dataset.table)

    def apply_schema(self, state: DatasetState, *args: Any, **kwargs: Any) -> Schema:
        raise NotImplementedError

    def apply_table(self, state: DatasetState, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError

    def apply_couplings(self, state: DatasetState, *args: Any, **kwargs: Any) -> CouplingSet:
        raise NotImplementedError

    def apply_sources(
        self,
        state: DatasetState,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[DatasetSourceInfo, ...]:
        raise NotImplementedError


class CreationOperator(Operator):
    """Creates a dataset from external input.

    Subclasses implement ``make_source`` and ``build``. ``_create`` handles
    the manager wiring automatically:

    1. Resolve manager: ``source_manager`` Parameter (if bound) or global default.
    2. Call ``make_source()`` — subclass returns a live, opened DataSource.
    3. Register the source via ``manager.register_source(source)`` → ``source_desc_id``.
    4. Call ``generate_source_info()`` for provenance.
    5. Call ``build(..., source_desc_id=source_desc_id)`` — subclass builds the state.

    ``source_manager`` is a Parameter so it can be bound at instance level for
    isolation (e.g. in tests): ``MyOp.instance(source_manager=isolated_mgr)``.

    Subclasses normally override ``make_source``, ``generate_source_info``, and
    ``build``. Override ``normalize_call``, ``run``, or ``validate_result`` for
    lifecycle customization; override ``__call__`` directly only as a full
    escape hatch.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.construct(),
        sources=SourcesTransition.construct(),
        index_identity=IndexIdentityTransition.mint(),
    )
    source_manager: ClassVar[Parameter] = Parameter(default=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Dataset:
        return Operator.__call__(self, *args, **kwargs)

    def _create(self, *args: Any, **kwargs: Any) -> Dataset:
        """Compatibility entrypoint for creation subclasses."""

        return Operator.__call__(self, *args, **kwargs)

    def normalize_call(self, *args: Any, **kwargs: Any) -> OperatorCall:
        return OperatorCall(
            operator=self,
            args=args,
            kwargs=kwargs,
            reference_contexts=self._assert_field_handles_allowed(args, kwargs),
        )

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        args = call.args
        kwargs = dict(call.kwargs)
        mgr: SourceManager = self.resolve_param("source_manager") or get_default_manager()

        source = self.make_source(*args, **kwargs)
        source_desc_id = mgr.register_source(source) if source is not None else None

        source_info = self.generate_source_info(*args, **kwargs)
        state = self.build(
            *args,
            **{
                **kwargs,
                "source_desc_id": source_desc_id,
                "source_manager": mgr,
            },
        )
        state = replace(state, sources=(source_info,))
        return Dataset(state=state, source_manager=mgr)

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)

    def make_source(self, *args: Any, **kwargs: Any) -> DataSource | None:
        """Return a live, opened DataSource for this dataset.

        The returned source is registered into the manager via register_source(),
        which calls source.describe() to obtain the SourceDescriptor.

        Return None for operators that use no managed source (uncommon).
        """
        return None

    @abstractmethod
    def generate_source_info(self, *args: Any, **kwargs: Any) -> DatasetSourceInfo: ...

    @abstractmethod
    def build(self, *args: Any, **kwargs: Any) -> DatasetState: ...


class PlanOperator(Operator):
    """Creates an explicit plan dataset.

    A plan dataset is a normal Dataset whose rows describe a later operation.
    Subclasses normalize their call signature into ``OperatorCall`` and return
    a concrete plan from ``run``. Plan outputs are sibling artifacts by default:
    no DatasetContext is advanced unless a subclass explicitly adds a context
    effect.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        index_identity=IndexIdentityTransition.mint(),
    )
    plan_index_name: ClassVar[str] = "plan_id"
    required_plan_fields: ClassVar[tuple[str, ...]] = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Dataset:
        return Operator.__call__(self, *args, **kwargs)

    def normalize_call(self, *args: Any, **kwargs: Any) -> OperatorCall:
        # Plan ops that declare a source dataset (a ``DatasetInput`` slot) get the
        # same primary-dataset + field-handle normalization dataset ops do; the
        # rest keep the generic args/kwargs passthrough.
        sig = self.signature
        if sig is not None and sig.dataset_slots():
            return self._normalize_source_plan_call(args, kwargs)
        return OperatorCall(
            operator=self,
            args=args,
            kwargs=kwargs,
            reference_contexts=self._assert_field_handles_allowed(args, kwargs),
        )

    def _normalize_source_plan_call(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> OperatorCall:
        """Normalize a plan call whose first operand is an explicit source dataset.

        The plan-op counterpart to ``DatasetOperator.normalize_call``: the first
        positional is the source ``Dataset``, and field-scoped handles are resolved
        against it through the shared path. Plans are sibling artifacts, so no
        ``DatasetContext`` is advanced.
        """

        if not args or not isinstance(args[0], Dataset):
            raise TypeError(f"{self.name} expects a Dataset as the first argument.")
        dataset = args[0]
        rest, resolved_kwargs, contexts = self._resolve_field_handles_for_dataset(
            dataset, args[1:], dict(kwargs)
        )
        return OperatorCall(
            operator=self,
            datasets=(dataset,),
            states=(dataset.state,),
            args=rest,
            kwargs=resolved_kwargs,
            reference_contexts=contexts,
        )

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        options = self.plan_validation_options(call)
        self.validate_plan_schema(result.schema, result.table, **options)

    def plan_validation_options(self, call: OperatorCall) -> dict[str, Any]:
        """Return validation options for this normalized plan call."""

        return {
            "plan_index_name": self.plan_index_name,
            "required_plan_fields": self.required_plan_fields,
        }

    def build_plan_dataset(
        self,
        *,
        schema: Schema,
        table: pd.DataFrame,
        sources: tuple[DatasetSourceInfo, ...] = (),
        source_manager: SourceManager | None = None,
        metadata: dict[str, Any] | None = None,
        plan_index_name: str | None = None,
        required_plan_fields: tuple[str, ...] | None = None,
    ) -> Dataset:
        if self.transitions.index_identity.mode == "mint":
            schema = mint_primary_index_identity(schema)
        index_name = plan_index_name or self.plan_index_name
        required_fields = (
            self.required_plan_fields
            if required_plan_fields is None
            else required_plan_fields
        )
        self.validate_plan_schema(
            schema,
            table,
            plan_index_name=index_name,
            required_plan_fields=required_fields,
        )
        return Dataset(
            state=DatasetState(
                schema=schema,
                table=table,
                sources=sources,
                metadata=metadata or {},
            ),
            source_manager=source_manager,
        )

    def validate_plan_schema(
        self,
        schema: Schema,
        table: pd.DataFrame,
        *,
        plan_index_name: str | None = None,
        required_plan_fields: tuple[str, ...] | None = None,
    ) -> None:
        index_name = plan_index_name or self.plan_index_name
        required_fields = (
            self.required_plan_fields
            if required_plan_fields is None
            else required_plan_fields
        )
        if not schema.has(index_name) or not isinstance(schema.get(index_name), IndexField):
            raise ValueError(
                f"{self.name}: plan schema must include IndexField({index_name!r})."
            )
        if table.index.name != index_name:
            raise ValueError(f"{self.name}: plan table index must be named {index_name!r}.")

        missing_schema = [name for name in required_fields if not schema.has(name)]
        missing_table = [name for name in required_fields if name not in table.columns]
        missing = tuple(dict.fromkeys((*missing_schema, *missing_table)))
        if missing:
            raise ValueError(f"{self.name}: plan is missing required fields: {list(missing)}")


class PlanConsumerMixin:
    """Shared helpers for operators that consume explicit plan datasets."""

    required_plan_fields: ClassVar[tuple[str, ...]] = ()

    def validate_plan_dataset(
        self,
        plan: Dataset,
        *,
        required_fields: tuple[str, ...] | None = None,
        warn_on_couplings: bool = True,
    ) -> None:
        fields = self.required_plan_fields if required_fields is None else required_fields
        missing_schema = [name for name in fields if not plan.schema.has(name)]
        missing_table = [name for name in fields if name not in plan.table.columns]
        missing = tuple(dict.fromkeys((*missing_schema, *missing_table)))
        if missing:
            raise ValueError(f"{self.name}: plan is missing required fields: {list(missing)}")
        if warn_on_couplings and plan.couplings.couplings:
            warnings.warn(
                f"{self.name}: plan couplings are ignored. Consider consuming plan "
                "dataset bindings before applying the plan.",
                UserWarning,
                stacklevel=3,
            )

    def resolve_plan_foreign_index(
        self,
        plan: Dataset,
        target: Dataset,
        *,
        field_name: str | None = None,
    ) -> ForeignIndexField:
        target_identity = primary_index_identity(target)
        field = resolve_foreign_index_field(
            plan.schema,
            target_identity,
            field_name=field_name,
            op_name=self.name,
        )
        if field.name not in plan.table.columns:
            raise ValueError(
                f"{self.name}: ForeignIndexField {field.name!r} is missing from plan table."
            )
        return field

    def validate_foreign_index_labels(
        self,
        target: Dataset,
        labels: pd.Series,
        *,
        field_name: str,
        allow_null: bool = False,
    ) -> None:
        null_mask = pd.isna(labels)
        if not allow_null and null_mask.any():
            raise ValueError(
                f"{self.name}: foreign index field {field_name!r} contains null labels."
            )

        non_null_labels = labels[~null_mask]
        missing = non_null_labels[~non_null_labels.isin(target.table.index)].tolist()
        if missing:
            raise ValueError(
                f"{self.name}: foreign index field {field_name!r} references labels "
                f"missing from target dataset: {missing}"
            )


class CompositionOperator(Operator):
    """Combines multiple datasets into one.

    Sources are unioned by default (deduplicated by ``source_id`` or
    ``source_uri``). Override ``combine_sources`` to change this.

    Subclasses normally override the composition hooks. Override
    ``normalize_call``, ``run``, or ``validate_result`` for lifecycle
    customization; override ``__call__`` directly only as a full escape hatch.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema=SchemaTransition.compose(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.derive(),
        sources=SourcesTransition.derive(),
        index_identity=IndexIdentityTransition.coalesce(),
    )
    advances_dataset_context: ClassVar[bool] = True

    def __call__(self, *datasets: Dataset, **kwargs: Any) -> Dataset | FieldHandle:
        return Operator.__call__(self, *datasets, **kwargs)

    def _compose(self, *datasets: Dataset, **kwargs: Any) -> Dataset:
        """Compatibility entrypoint for composition subclasses."""

        return Operator.__call__(self, *datasets, **kwargs)

    def normalize_call(self, *datasets: Dataset, **kwargs: Any) -> OperatorCall:
        handle_contexts = self._assert_field_handles_allowed(datasets, kwargs)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(f"{self.name}: expected Dataset inputs.")

        dataset_context = self.resolve_dataset_context()
        context_effects: tuple[ContextEffect, ...] = ()
        if (
            self.advances_dataset_context
            and dataset_context is not None
            and any(dataset is dataset_context.dataset for dataset in datasets)
        ):
            context_effects = (
                ContextEffect(
                    context=dataset_context,
                    anchor=dataset_context.dataset,
                    effect="advance",
                ),
            )

        return OperatorCall(
            operator=self,
            datasets=tuple(datasets),
            states=tuple(dataset.state for dataset in datasets),
            kwargs=kwargs,
            reference_contexts=handle_contexts,
            context_effects=context_effects,
        )

    def run(self, call: OperatorCall, transitions: TransitionPlan) -> Dataset:
        datasets = call.datasets
        kwargs = dict(call.kwargs)
        new_state = compute_output_state(
            self,
            call.states,
            call.args,
            kwargs,
            declared_transitions=transitions,
        )
        source_manager = self.combine_source_managers(
            *datasets, composed_schema=new_state.schema
        )
        return Dataset(
            state=new_state,
            source_manager=source_manager,
        )

    def validate_result(self, call: OperatorCall, result: Any) -> None:
        if not isinstance(result, Dataset):
            raise TypeError(
                f"{self.name}: expected run() to return a Dataset, got "
                f"{type(result).__name__}."
            )
        result.schema.validate_table(result.table)

    def combine_sources(
        self,
        *states: DatasetState,
        composed_schema: Schema | None = None,
        **_: Any,
    ) -> tuple[DatasetSourceInfo, ...]:
        seen: dict[str, DatasetSourceInfo] = {}
        for state in states:
            for src in state.sources:
                seen[src.source_id or src.source_uri] = src
        return tuple(seen.values())

    def combine_source_managers(
        self,
        *datasets: Dataset,
        composed_schema: Schema | None = None,
    ) -> SourceManager | None:
        return None

    @abstractmethod
    def apply_schema(self, *states: DatasetState, **kwargs: Any) -> Schema: ...

    @abstractmethod
    def apply_table(self, *states: DatasetState, **kwargs: Any) -> pd.DataFrame: ...

    @abstractmethod
    def apply_couplings(self, *states: DatasetState, **kwargs: Any) -> CouplingSet: ...


def _selector_name(value: Any) -> str:
    """Resolve a local field name from a string selector or a ``FieldHandle``.

    A ``FieldHandle``'s ``.name`` resolves against its context's current
    snapshot, so callers must resolve output names *after* any context advance.
    """

    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    raise TypeError(
        f"expected a field name or FieldHandle, got {type(value).__name__}."
    )


def _is_null_label(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    missing = pd.isna(value)
    return isinstance(missing, bool) and missing
