"""patchframe.ops.signature

Declarative operand + return contracts for operators (lazy-duality-plan.md
Phase 4). An ``OperatorSignature`` says, per operand slot, what kinds of value
the slot accepts, and what the call returns — which a shared ``normalize_call``
then interprets, replacing today's ``field_handle_inputs`` tuple and the
per-operator ``normalize_call`` boilerplate.

The return specs encode the consistent/honest return rule: a coupling-producing
lazy op hands back a handle to its output field(s) (chainable); only entry/exit
bridges return a whole ``Dataset``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Input slots
# ---------------------------------------------------------------------------


#: Sentinel for a ``ParamInput`` with no declared default.
_PARAM_MISSING = object()


@dataclass(frozen=True, slots=True)
class DatasetInput:
    """A whole-dataset operand slot.

    Accepts a ``Dataset`` (eager arm) or, when ``accepts_bundle_handle``, a
    ``FieldHandle`` to a ``BundleField`` (the lazy / per-fiber arm — the trigger
    for whole-dataset ops like ``where``/``merge``). ``variadic`` consumes all
    remaining positional operands (``merge``'s left/right/plan; ``concat``).
    """

    accepts_bundle_handle: bool = True
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class FieldInput:
    """A single field-operand slot: a typed ``FieldHandle`` or a field name.

    ``dataset`` names the dataset slot the field belongs to. ``field_type`` is
    the expected concrete ``Field`` subtype (e.g. ``DimensionedSliceField``);
    ``None`` accepts any. (Type enforcement is a later/ contract-suite step; the
    declaration lands now.) ``output=True`` marks a field that is also the
    operator's in-place output (``slice_data``'s ``data_field``), so the lazy
    arm returns a chaining handle to it.
    """

    dataset: str = "dataset"
    field_type: Any = None
    output: bool = False
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class SelectionInput:
    """A multi-field operand slot: a ``FieldSelection`` or a sequence of names."""

    dataset: str = "dataset"
    field_type: Any = None
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class ParamInput:
    """A per-call positional-or-keyword parameter slot — *not* an operand.

    The declared home for per-call data an operator consumes but that is not a
    dataset/field operand: a filter ``predicate``, a ``collision`` strategy, a
    mapping. Declaring it (in positional order, alongside the operand slots)
    lets the lazy-arm binder *name* a positional argument, so the deferred
    ``ApplyOperator`` can replay it as a keyword and eager/lazy calls stay
    positionally symmetric (``where(ds, pred)`` ⇄ ``where(handle, pred, out=…)``).

    Distinct from ``Parameter`` (instance-level behavioural config; see
    CLAUDE.md): ``ParamInput`` is per-call data.
    """

    default: Any = _PARAM_MISSING

    @property
    def has_default(self) -> bool:
        return self.default is not _PARAM_MISSING


# ---------------------------------------------------------------------------
# Returns — the eager/lazy seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetReturn:
    """Always returns a ``Dataset``.

    For ops with no coupling output: eager-only ops, the ``bundle`` entry
    constructor, and the ``extract``/``flatten``/``collect`` exit bridges.
    """


@dataclass(frozen=True, slots=True)
class FieldReturn:
    """Dual: eager → ``Dataset``; lazy → a ``FieldHandle`` to the coupling output.

    The handle is resolved from the recorded operation's ``output_field`` at run
    time, so it points at the field actually produced — honest and chainable.
    """


@dataclass(frozen=True, slots=True)
class SelectionReturn:
    """Dual: eager → ``Dataset``; lazy → a ``Selection`` over the coupling outputs.

    For multi-output ops (e.g. ``assign``).
    """


@dataclass(frozen=True, slots=True)
class FieldOutput:
    """A produced (fresh) output field whose name the caller supplies.

    The dual of ``FieldInput``: where ``FieldInput`` asks the caller *which
    existing field* to read, ``FieldOutput`` asks for *the name of the field to
    produce*. Declared on lifting operators (``merge``/``concat``/``where``/...)
    as the ``out`` slot — the caller passes the produced field's name
    (``merge(..., out="merged")``), the op produces a field of that name, and the
    lazy arm returns a handle to it: the chaining point, inherent to every
    operator of that kind. ``field_type`` is the produced field's type
    (``BundleField`` for the per-fiber lift result).

    In-place outputs (e.g. ``slice_data``'s ``data_field``) declare no
    ``FieldOutput`` — they keep a ``returns`` kind and resolve from the recorded
    op; bridges return a ``Dataset``.
    """

    field_type: Any = None


InputSpec = DatasetInput | FieldInput | SelectionInput | ParamInput
ReturnSpec = DatasetReturn | FieldReturn | SelectionReturn


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorSignature:
    """Declarative operand + return contract interpreted by ``normalize_call``.

    ``inputs`` is an ordered mapping of slot name → spec (order matters for
    positional binding). ``custom`` is the escape hatch for operators whose
    normalization the signature cannot describe (``explode``, the ``concat``
    dispatcher); those keep a hand-written ``normalize_call``.
    """

    inputs: Mapping[str, InputSpec] = field(default_factory=dict)
    outputs: Mapping[str, FieldOutput] = field(default_factory=dict)
    returns: ReturnSpec = field(default_factory=DatasetReturn)
    custom: bool = False

    def field_slots(self) -> tuple[str, ...]:
        """Slot names that accept field handles (``FieldInput``/``SelectionInput``)."""

        return tuple(
            name
            for name, spec in self.inputs.items()
            if isinstance(spec, (FieldInput, SelectionInput))
        )

    def dataset_slots(self) -> tuple[str, ...]:
        """Slot names that accept whole datasets (``DatasetInput``)."""

        return tuple(
            name for name, spec in self.inputs.items() if isinstance(spec, DatasetInput)
        )

    def param_slots(self) -> tuple[str, ...]:
        """Slot names that carry per-call parameters (``ParamInput``)."""

        return tuple(
            name for name, spec in self.inputs.items() if isinstance(spec, ParamInput)
        )

    def output_slots(self) -> tuple[str, ...]:
        """Names of the caller-named produced-field slots (``FieldOutput``)."""

        return tuple(self.outputs.keys())

    def output_slot_name(self) -> str | None:
        """The single produced-field slot name, or ``None`` if not exactly one.

        The lazy arm's chaining point: the bundle arm passes its value as
        ``defer_in_level(..., out=)``; a same-level fresh op reads it as the
        produced field's name. Multi-output ops use ``output_slots()`` instead.
        """

        names = tuple(self.outputs.keys())
        return names[0] if len(names) == 1 else None
