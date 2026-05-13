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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

import pandas as pd

from patchframe.data.manager import SourceManager, get_default_manager
from patchframe.data.source import DataSource
from patchframe.dataset.couplings import CouplingSet
from patchframe.dataset.dataset import Dataset
from patchframe.dataset.provenance import DatasetSourceInfo
from patchframe.dataset.schema import Schema
from patchframe.dataset.state import DatasetState
from patchframe.ops.transitions import AspectTransition, TransitionPlan

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
        t, s = self.transitions, dataset.state
        result = dataset.replace_state(
            schema=(
                self.apply_schema(s, *args, **kwargs)
                if t.schema.mode != "preserve"
                else s.schema
            ),
            table=(
                self.apply_table(s, *args, **kwargs)
                if t.table.mode != "preserve"
                else s.table
            ),
            couplings=(
                self.apply_couplings(s, *args, **kwargs)
                if t.couplings.mode != "preserve"
                else s.couplings
            ),
            sources=(
                self.apply_sources(s, *args, **kwargs)
                if t.sources.mode != "inherit"
                else s.sources
            ),
        )
        self._validate_output(result)
        return result

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


class CompositionOperator(Operator):
    """Combines multiple datasets into one.

    Sources are unioned by default (deduplicated by ``source_id`` or
    ``source_uri``). Override ``combine_sources`` to change this.

    Override ``__call__`` directly for a full escape hatch.
    """

    transitions: ClassVar[TransitionPlan] = TransitionPlan(
        schema    = AspectTransition("derive"),
        table     = AspectTransition("derive"),
        couplings = AspectTransition("derive"),
        sources   = AspectTransition("union"),
    )

    def __call__(self, *datasets: Dataset, **kwargs: Any) -> Dataset:
        return self._compose(*datasets, **kwargs)

    def _compose(self, *datasets: Dataset, **kwargs: Any) -> Dataset:
        states = tuple(d.state for d in datasets)
        return Dataset(
            state=DatasetState(
                schema    = self.apply_schema(*states, **kwargs),
                table     = self.apply_table(*states, **kwargs),
                couplings = self.apply_couplings(*states, **kwargs),
                sources   = self.combine_sources(*states),
            ),
            source_manager=self.combine_source_managers(*datasets),
        )

    def combine_sources(self, *states: DatasetState) -> tuple[DatasetSourceInfo, ...]:
        seen: dict[str, DatasetSourceInfo] = {}
        for state in states:
            for src in state.sources:
                seen[src.source_id or src.source_uri] = src
        return tuple(seen.values())

    def combine_source_managers(self, *datasets: Dataset) -> SourceManager | None:
        return None

    @abstractmethod
    def apply_schema(self, *states: DatasetState, **kwargs: Any) -> Schema: ...

    @abstractmethod
    def apply_table(self, *states: DatasetState, **kwargs: Any) -> pd.DataFrame: ...

    @abstractmethod
    def apply_couplings(self, *states: DatasetState, **kwargs: Any) -> CouplingSet: ...
