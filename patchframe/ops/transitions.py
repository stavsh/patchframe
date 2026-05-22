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

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Mapping


class Cardinality(Enum):
    """How an operator maps input rows to output rows.

    Declared adjacent to ``TransitionPlan`` rather than as an aspect: it is an
    operator property, not an input/output aspect relation. Consumed later by
    contract tests and cache reasoning; declared now so the data exists.
    """

    PRESERVE = "preserve"   # one output row per input row
    FILTER = "filter"       # output rows are a subset of input rows
    EXPAND = "expand"       # one input row may yield several output rows
    UNKNOWN = "unknown"     # no local cardinality guarantee


SchemaMode = Literal["preserve", "extend", "narrow", "rewrite", "construct", "infer"]
TableMode = Literal["preserve", "mutate", "construct"]
CouplingsMode = Literal["derive", "union", "construct", "clear"]
SourcesMode = Literal["inherit", "union", "construct", "clear"]
IndexIdentityMode = Literal["preserve", "inherit", "mint", "coalesce"]
AccessorsMode = Literal["preserve", "mutate"]


@dataclass(frozen=True, slots=True)
class SchemaTransition:
    """Declared schema-aspect effect of an operator.

    - ``preserve``  : output schema equals the selected input schema.
    - ``extend``    : input fields survive unchanged; new fields may be added.
    - ``narrow``    : some fields may be removed; survivors keep their identity.
    - ``rewrite``   : field identities survive but representation changes
      (rename/retype). ``mapping`` carries old->new names when a rename occurs.
    - ``construct`` : output schema is newly assembled.
    - ``infer``     : the operator changes the schema but does not characterize
      how. Placeholder mode — treated conservatively (see coupling derivation
      in ``ops.base``). Real inference arrives with the FieldIdentity stage.
    """

    mode: SchemaMode = "infer"
    input: int = 0
    mapping: Mapping[str, str] | None = None

    _MODES = frozenset({"preserve", "extend", "narrow", "rewrite", "construct", "infer"})

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
    def rewrite(
        cls, *, mapping: Mapping[str, str] | None = None, input: int = 0
    ) -> "SchemaTransition":
        return cls(mode="rewrite", input=input, mapping=mapping)

    @classmethod
    def construct(cls) -> "SchemaTransition":
        return cls(mode="construct")

    @classmethod
    def infer(cls, *, input: int = 0) -> "SchemaTransition":
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

    - ``derive``    : the framework computes output couplings from the schema
      transition (default). Preserve/extend keep couplings; rewrite-with-mapping
      rewrites refs; narrow/infer/construct conservatively retain couplings
      whose fields survive.
    - ``union``     : couplings from multiple inputs are combined (composition).
    - ``construct`` : the operator builds the coupling set itself.
    - ``clear``     : the output intentionally has no couplings.
    """

    mode: CouplingsMode = "derive"
    input: int = 0

    _MODES = frozenset({"derive", "union", "construct", "clear"})

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"CouplingsTransition: unknown mode {self.mode!r}.")

    @classmethod
    def derive(cls, *, input: int = 0) -> "CouplingsTransition":
        return cls(mode="derive", input=input)

    @classmethod
    def union(cls) -> "CouplingsTransition":
        return cls(mode="union")

    @classmethod
    def construct(cls) -> "CouplingsTransition":
        return cls(mode="construct")

    @classmethod
    def clear(cls) -> "CouplingsTransition":
        return cls(mode="clear")


@dataclass(frozen=True, slots=True)
class SourcesTransition:
    """Declared sources-aspect effect of an operator.

    - ``inherit``   : use one input's source records (default for unary ops).
    - ``union``     : combine source records from multiple inputs (composition).
    - ``construct`` : a creation/source-producing operator mints source records.
    - ``clear``     : the output intentionally has no source records.
    """

    mode: SourcesMode = "inherit"
    input: int = 0

    _MODES = frozenset({"inherit", "union", "construct", "clear"})

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"SourcesTransition: unknown mode {self.mode!r}.")

    @classmethod
    def inherit(cls, *, input: int = 0) -> "SourcesTransition":
        return cls(mode="inherit", input=input)

    @classmethod
    def union(cls) -> "SourcesTransition":
        return cls(mode="union")

    @classmethod
    def construct(cls) -> "SourcesTransition":
        return cls(mode="construct")

    @classmethod
    def clear(cls) -> "SourcesTransition":
        return cls(mode="clear")


@dataclass(frozen=True, slots=True)
class IndexIdentityTransition:
    """Declared primary index-identity effect of an operator.

    - ``preserve`` : output rows keep the selected input identity namespace.
    - ``inherit``  : output identity is selected from a named non-primary input
      (e.g. a plan dataset). ``input`` may be an int position or a string label.
    - ``mint``     : output rows enter a newly minted identity namespace.
    - ``coalesce`` : preserve when all inputs share one namespace, else mint.
    """

    mode: IndexIdentityMode = "preserve"
    input: int | str = 0

    _MODES = frozenset({"preserve", "inherit", "mint", "coalesce"})

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"IndexIdentityTransition: unknown mode {self.mode!r}.")

    @classmethod
    def preserve(cls, *, input: int | str = 0) -> "IndexIdentityTransition":
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

    Defaults are worst-case-safe: an operator that declares nothing is assumed
    to change the schema (``infer``), rebuild the table (``mutate``), and derive
    its couplings from the schema transition. Operators narrow these to precise
    modes as part of their declaration.
    """

    schema: SchemaTransition = field(default_factory=SchemaTransition.infer)
    table: TableTransition = field(default_factory=TableTransition.mutate)
    couplings: CouplingsTransition = field(default_factory=CouplingsTransition.derive)
    sources: SourcesTransition = field(default_factory=SourcesTransition.inherit)
    index_identity: IndexIdentityTransition = field(
        default_factory=IndexIdentityTransition.preserve
    )
    accessors: AccessorsTransition = field(default_factory=AccessorsTransition.preserve)

    def _with(self, **changes: Any) -> "TransitionPlan":
        """Return a copy of this plan with the given aspect transitions replaced.

        Intended for ``resolve_transitions`` overrides that refine a
        conservative class-level plan based on call-time inputs.
        """
        return replace(self, **changes)
