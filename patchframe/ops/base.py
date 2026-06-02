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
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

import pandas as pd

from patchframe.data.manager import SourceManager, get_default_manager
from patchframe.data.source import DataSource
from patchframe.dataset.couplings import Coupling, CouplingSet
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
from patchframe.ops.transitions import (
    Cardinality,
    CouplingsTransition,
    IndexIdentityTransition,
    SchemaTransition,
    SourcesTransition,
    TableTransition,
    TransitionPlan,
)

MISSING = object()


@dataclass(frozen=True, slots=True)
class Parameter:
    """Declarative configurable operator parameter.

    Parameters declared as class attributes participate in ``.instance(...)``
    configuration and are available through ``bound_params()``.
    """

    default: Any = MISSING
    required: bool = False


class OperatorMeta(ABCMeta):
    """Metaclass providing direct class-call execution and parameter collection.

    ``MyOperator(...)`` executes a temporary default-configured instance.
    Configured instances are created through ``MyOperator.instance(...)``.
    """

    def __new__(mcls, name, bases, namespace):
        params: dict[str, Parameter] = {}
        for base in bases:
            params.update(getattr(base, "__parameters__", {}))
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Parameter):
                params[attr_name] = attr_value
        namespace["__parameters__"] = params
        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__()
        return instance.__call__(*args, **kwargs)


class Operator(metaclass=OperatorMeta):
    """Base callable/configurable operator."""

    transitions: ClassVar[TransitionPlan] = TransitionPlan()
    cardinality: ClassVar[Cardinality] = Cardinality.UNKNOWN
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

    Override ``__call__`` directly for a full escape hatch that bypasses aspect
    dispatch entirely.
    """

    def __call__(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        return self._apply(dataset, *args, **kwargs)

    def _apply(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        new_state = compute_output_state(self, (dataset.state,), args, kwargs)
        result = dataset.replace_state(
            schema=new_state.schema,
            table=new_state.table,
            couplings=new_state.couplings,
            sources=new_state.sources,
        )
        self._validate_output(result)
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
        existing = set(derived.couplings)
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

    Override ``__call__`` directly for a full escape hatch.
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
        return self._create(*args, **kwargs)

    def _create(self, *args: Any, **kwargs: Any) -> Dataset:
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
    Subclasses own their call signature and use ``build_plan_dataset`` to
    assemble and validate the concrete plan state.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema=SchemaTransition.construct(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.clear(),
        index_identity=IndexIdentityTransition.mint(),
    )
    plan_index_name: ClassVar[str] = "plan_id"
    required_plan_fields: ClassVar[tuple[str, ...]] = ()

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

    Override ``__call__`` directly for a full escape hatch.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema=SchemaTransition.compose(),
        table=TableTransition.construct(),
        couplings=CouplingsTransition.derive(),
        sources=SourcesTransition.derive(),
        index_identity=IndexIdentityTransition.coalesce(),
    )

    def __call__(self, *datasets: Dataset, **kwargs: Any) -> Dataset:
        return self._compose(*datasets, **kwargs)

    def _compose(self, *datasets: Dataset, **kwargs: Any) -> Dataset:
        states = tuple(d.state for d in datasets)
        new_state = compute_output_state(self, states, (), kwargs)
        source_manager = self.combine_source_managers(
            *datasets, composed_schema=new_state.schema
        )
        return Dataset(
            state=new_state,
            source_manager=source_manager,
        )

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


def _is_null_label(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    missing = pd.isna(value)
    return isinstance(missing, bool) and missing
