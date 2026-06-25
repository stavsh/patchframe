"""
patchframe.ops.transitions

Transition metadata for patchframe operators.

Operators in patchframe are defined over multiple dataset aspects, not just the
row table. A ``TransitionPlan`` declares those structural effects so the
operator base can route aspect work, derive couplings, and (later) drive
operator-contract tests.

Each aspect has its own typed transition class with a small, validated set of
modes. Construct them through the classmethod factories
(``SchemaTransition.preserve()``, ``CouplingsTransition.derive()``, ...).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal


class Cardinality(Enum):
    """How an operator maps input rows to output rows.

    Declared adjacent to ``TransitionPlan`` rather than as an aspect: it is an
    operator property, not an input/output aspect relation. Consumed later by
    contract tests and cache reasoning; declared now so the data exists.
    """

    PRESERVE = "preserve"   # one output row per input row
    FILTER = "filter"       # output rows are a subset of input rows
    EXPAND = "expand"       # one input row may yield several output rows
    REDUCE = "reduce"       # many input rows collapse to one aggregate row
    UNKNOWN = "unknown"     # no local cardinality guarantee


class PerRowIndependence(Enum):
    """Whether output row ``i`` depends only on input row ``i``.

    Declared adjacent to ``Cardinality`` (a ``ClassVar`` on the operator, not a
    ``TransitionPlan`` aspect). Together with cardinality and index-identity
    minting it forms the "3-part test" that decides whether an operator can be a
    same-level coupling or needs a ``BundleField`` carrier in the lazy arm (see
    ``lazy-and-bundle.md`` §4). ``UNKNOWN`` routes conservatively — an op that
    is dynamic (``consume``) or conditional (unaligned ``concat_columns``) fails
    the test and is treated as needing a bundle.
    """

    INDEPENDENT = "independent"  # output row i depends only on input row i
    DEPENDENT = "dependent"      # output depends on other rows (global/cross-row)
    UNKNOWN = "unknown"          # no local guarantee; resolved per call or dynamic


SchemaMode = Literal[
    "preserve", "extend", "narrow", "rewrite", "compose", "construct", "custom", "infer"
]
TableMode = Literal["preserve", "mutate", "construct"]
CouplingsMode = Literal[
    "derive", "inherit", "homogeneous", "construct", "clear", "custom"
]
SourcesMode = Literal["inherit", "derive", "compose", "construct", "clear", "custom"]
IndexIdentityMode = Literal[
    "preserve", "inherit", "mint", "coalesce", "derive", "custom"
]
AccessorsMode = Literal["preserve", "mutate"]


@dataclass(frozen=True, slots=True)
class SchemaTransition:
    """Declared schema-aspect effect of an operator.

    - ``preserve``  : output schema equals the selected input schema.
    - ``extend``    : input fields survive unchanged; new fields may be added.
    - ``narrow``    : some fields may be removed; survivors keep their identity.
    - ``rewrite``   : field identities survive but representation changes
      (rename/retype). The rename mapping is derived structurally from
      ``FieldIdentity`` lineage; no explicit declaration needed.
    - ``compose``   : N-ary composition through ``MergedField`` parents.
      Same-name fields with shared ``FieldIdentity`` keep it; divergent
      identities are unified under a freshly minted identity.
    - ``construct`` : output schema is newly assembled with no input lineage.
    - ``custom``    : input lineage exists but the operator's transformation
      is not one the structural vocabulary captures. Total escape hatch; the
      contract suite warns and asserts nothing about schema.
    - ``infer``     : deprecated. Use ``preserve`` if the schema does not
      change, or the appropriate structural mode otherwise.
    """

    mode: SchemaMode = "infer"
    input: int = 0

    _MODES = frozenset({
        "preserve", "extend", "narrow", "rewrite", "compose", "construct",
        "custom", "infer",
    })

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"SchemaTransition: unknown mode {self.mode!r}.")

    @classmethod
    def preserve(cls, *, input: int = 0) -> "SchemaTransition":
        return cls(mode="preserve", input=input)

    @classmethod
    def extend(cls, *, input: int = 0) -> "SchemaTransition":
        return cls(mode="extend", input=input)

    @classmethod
    def narrow(cls, *, input: int = 0) -> "SchemaTransition":
        return cls(mode="narrow", input=input)

    @classmethod
    def rewrite(cls, *, input: int = 0) -> "SchemaTransition":
        return cls(mode="rewrite", input=input)

    @classmethod
    def compose(cls) -> "SchemaTransition":
        return cls(mode="compose")

    @classmethod
    def construct(cls) -> "SchemaTransition":
        return cls(mode="construct")

    @classmethod
    def custom(cls) -> "SchemaTransition":
        return cls(mode="custom")

    @classmethod
    def infer(cls, *, input: int = 0) -> "SchemaTransition":
        warnings.warn(
            "SchemaTransition.infer is deprecated. Use preserve() when the "
            "schema does not change, or one of extend / narrow / rewrite / "
            "compose / construct / custom for the operator's actual effect.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(mode="infer", input=input)


@dataclass(frozen=True, slots=True)
class TableTransition:
    """Declared table-aspect effect of an operator.

    - ``preserve``  : the table is passed through unchanged (``apply_table`` is
      skipped).
    - ``mutate``    : table values, index, row count, order, or columns change.
    - ``construct`` : the table is newly built.
    """

    mode: TableMode = "mutate"
    input: int = 0

    _MODES = frozenset({"preserve", "mutate", "construct"})

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"TableTransition: unknown mode {self.mode!r}.")

    @classmethod
    def preserve(cls, *, input: int = 0) -> "TableTransition":
        return cls(mode="preserve", input=input)

    @classmethod
    def mutate(cls, *, input: int = 0) -> "TableTransition":
        return cls(mode="mutate", input=input)

    @classmethod
    def construct(cls) -> "TableTransition":
        return cls(mode="construct")


@dataclass(frozen=True, slots=True)
class CouplingsTransition:
    """Declared couplings-aspect effect of an operator.

    - ``derive``      : the framework computes output couplings from the
      schema transition (default). Under preserve/extend keeps couplings;
      under rewrite rewrites refs via identity lineage; under compose runs
      MergedField-aware pruning.
    - ``inherit``     : output couplings equal a selected input's couplings.
      ``input`` may be an int position.
    - ``homogeneous`` : output couplings equal the inputs' couplings iff all
      input ``CouplingSet``s are structurally equal; raise otherwise.
      Canonical use is row-stacking, where a coupling applying to only some
      of the input rows would be unsafe to propagate silently.
    - ``construct``   : the operator builds the coupling set itself.
    - ``clear``       : the output intentionally has no couplings.
    - ``custom``      : lineage exists but the operator opts out. Suite warns.

    ``rename_map``, ``dropped``, and ``superseded_per_input`` are populated
    by ``resolve_derived_transitions`` for ``derive`` aspects from input/
    output schema lineage. They are empty on freshly declared transitions
    and on non-``derive`` modes. The dispatch layer reads them to rewrite,
    prune, and supersede input couplings; the contract suite reads them to
    verify the framework's lineage analysis against the operator's output.
    """

    mode: CouplingsMode = "derive"
    input: int = 0
    rename_map: tuple[tuple[str, str], ...] = ()
    dropped: tuple[str, ...] = ()
    superseded_per_input: tuple[tuple[int, tuple[str, ...]], ...] = ()

    _MODES = frozenset({
        "derive", "inherit", "homogeneous", "construct", "clear", "custom",
    })

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"CouplingsTransition: unknown mode {self.mode!r}.")

    @classmethod
    def derive(cls, *, input: int = 0) -> "CouplingsTransition":
        return cls(mode="derive", input=input)

    @classmethod
    def inherit(cls, *, input: int = 0) -> "CouplingsTransition":
        return cls(mode="inherit", input=input)

    @classmethod
    def homogeneous(cls) -> "CouplingsTransition":
        return cls(mode="homogeneous")

    @classmethod
    def construct(cls) -> "CouplingsTransition":
        return cls(mode="construct")

    @classmethod
    def clear(cls) -> "CouplingsTransition":
        return cls(mode="clear")

    @classmethod
    def custom(cls) -> "CouplingsTransition":
        return cls(mode="custom")

    @classmethod
    def union(cls) -> "CouplingsTransition":
        raise ValueError(
            "CouplingsTransition.union has been removed. It meant two "
            "different things: use derive() for the compose-derive use case "
            "(MergedField-aware pruning union under compose schema), or "
            "homogeneous() for the row-stack use case (require all inputs to "
            "agree before propagating)."
        )


@dataclass(frozen=True, slots=True)
class SourcesTransition:
    """Declared sources-aspect effect of an operator.

    - ``inherit``   : use a selected input's source records.
    - ``derive``    : framework resolves from the schema transition; under
      compose it is a deduped union of input sources.
    - ``compose``   : combine source records from multiple inputs. Use when
      source lineage composes independently of schema lineage.
    - ``construct`` : a creation/source-producing operator mints source
      records.
    - ``clear``     : the output intentionally has no source records.
    - ``custom``    : operator opts out. Suite warns.
    """

    mode: SourcesMode = "inherit"
    input: int = 0

    _MODES = frozenset({
        "inherit", "derive", "compose", "construct", "clear", "custom",
    })

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"SourcesTransition: unknown mode {self.mode!r}.")

    @classmethod
    def inherit(cls, *, input: int = 0) -> "SourcesTransition":
        return cls(mode="inherit", input=input)

    @classmethod
    def derive(cls) -> "SourcesTransition":
        return cls(mode="derive")

    @classmethod
    def compose(cls) -> "SourcesTransition":
        return cls(mode="compose")

    @classmethod
    def construct(cls) -> "SourcesTransition":
        return cls(mode="construct")

    @classmethod
    def clear(cls) -> "SourcesTransition":
        return cls(mode="clear")

    @classmethod
    def custom(cls) -> "SourcesTransition":
        return cls(mode="custom")

    @classmethod
    def union(cls) -> "SourcesTransition":
        raise ValueError(
            "SourcesTransition.union has been removed. Use derive() when "
            "source behavior follows schema lineage, or compose() when source "
            "records combine independently of schema lineage."
        )


@dataclass(frozen=True, slots=True)
class IndexIdentityTransition:
    """Declared primary index-identity effect of an operator.

    - ``inherit``  : output identity equals a selected input's identity.
      ``input`` may be an int position or a string label.
    - ``mint``     : output rows enter a newly minted identity namespace.
    - ``coalesce`` : preserve when all inputs share one namespace, else mint.
    - ``derive``   : framework resolves from the schema transition.
    - ``custom``   : operator opts out. Suite warns.
    - ``preserve`` : deprecated. Use ``inherit`` — a single name for a single
      operation.
    """

    mode: IndexIdentityMode = "preserve"
    input: int | str = 0

    _MODES = frozenset({
        "preserve", "inherit", "mint", "coalesce", "derive", "custom",
    })

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"IndexIdentityTransition: unknown mode {self.mode!r}.")

    @classmethod
    def preserve(cls, *, input: int | str = 0) -> "IndexIdentityTransition":
        warnings.warn(
            "IndexIdentityTransition.preserve is deprecated. Use "
            "inherit(input=N) — a single name for a single operation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(mode="preserve", input=input)

    @classmethod
    def inherit(cls, *, input: int | str = 0) -> "IndexIdentityTransition":
        return cls(mode="inherit", input=input)

    @classmethod
    def mint(cls) -> "IndexIdentityTransition":
        return cls(mode="mint")

    @classmethod
    def coalesce(cls) -> "IndexIdentityTransition":
        return cls(mode="coalesce")

    @classmethod
    def derive(cls) -> "IndexIdentityTransition":
        return cls(mode="derive")

    @classmethod
    def custom(cls) -> "IndexIdentityTransition":
        return cls(mode="custom")


@dataclass(frozen=True, slots=True)
class AccessorsTransition:
    """Declared accessor-cache effect of an operator.

    Minimal for now — accessor caching is not yet a real consumer.
    """

    mode: AccessorsMode = "preserve"
    input: int = 0

    _MODES = frozenset({"preserve", "mutate"})

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"AccessorsTransition: unknown mode {self.mode!r}.")

    @classmethod
    def preserve(cls, *, input: int = 0) -> "AccessorsTransition":
        return cls(mode="preserve", input=input)

    @classmethod
    def mutate(cls) -> "AccessorsTransition":
        return cls(mode="mutate")


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    """Declared transitions of an operator across dataset aspects.

    Phase 1 keeps the current defaults (``infer``/``mutate``/``derive``/
    ``inherit``/``preserve``/``preserve``) so operators that rely on the bare
    default continue to behave as before. The doc-stated new defaults
    (``preserve``/``preserve``/``derive``/``derive``/``derive``/``preserve``)
    land in Phase 6 alongside the operator redeclaration pass, when every
    built-in operator declares its aspects explicitly.

    Default factories deliberately use the class constructor (e.g.
    ``SchemaTransition`` rather than ``SchemaTransition.infer``) for the
    deprecated default modes so bare ``TransitionPlan()`` construction does
    not emit deprecation warnings.
    """

    schema: SchemaTransition = field(default_factory=SchemaTransition)
    table: TableTransition = field(default_factory=TableTransition.mutate)
    couplings: CouplingsTransition = field(default_factory=CouplingsTransition.derive)
    sources: SourcesTransition = field(default_factory=SourcesTransition.inherit)
    index_identity: IndexIdentityTransition = field(
        default_factory=IndexIdentityTransition
    )
    accessors: AccessorsTransition = field(default_factory=AccessorsTransition.preserve)

    def _with(self, **changes: Any) -> "TransitionPlan":
        """Return a copy of this plan with the given aspect transitions replaced.

        Intended for ``resolve_transitions`` overrides that refine a
        conservative class-level plan based on call-time inputs.
        """
        return replace(self, **changes)
