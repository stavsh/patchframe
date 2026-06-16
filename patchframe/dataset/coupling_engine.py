"""
patchframe.dataset.coupling_engine

Compiled coupling interpretation layer for patchframe.

CouplingEngine validates coupling field references against a schema and computes
a single topo-sorted execution order at construction time. The same order is
used by per-row dispatch (``apply_row``) and bulk materialization (``consume``).

Ordering rules:
- If A.output_field is in B.input_fields, A precedes B.
- Multiple couplings sharing the same output_field form a chain in declaration
  order — earlier-declared runs first, each reading the previous output.
- Cycles raise.

Partial consumption is supported via ``couplings_up_to(target)``: returns the
transitive upstream of a specific coupling plus the coupling itself, excluding
any downstream chain mates. ``couplings_for_column(name)`` returns the full
chain for a column plus its upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from patchframe.dataset.couplings import Coupling, CouplingSet
from patchframe.dataset.schema import Schema

if TYPE_CHECKING:
    from patchframe.dataset.state import DatasetState


@dataclass(frozen=True, slots=True)
class CouplingEngine:
    """Compiled coupling view over a schema and coupling set."""

    schema: Schema
    couplings: CouplingSet
    _order: tuple[Coupling, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "_order", self._compute_order())

    @property
    def order(self) -> tuple[Coupling, ...]:
        """Couplings in topo-sorted execution order."""
        return self._order

    def validate(self) -> None:
        """Validate field references and reject in-place read/write aliasing."""
        field_names = set(self.schema.names())
        for c in self.couplings.couplings:
            for name in c.input_fields():
                if name not in field_names:
                    raise ValueError(
                        f"Coupling input not in schema: {name!r} ({type(c).__name__})"
                    )
            output = c.output_field()
            if output not in field_names:
                raise ValueError(
                    f"Coupling output not in schema: {output!r} ({type(c).__name__})"
                )
        self._check_inplace_aliasing()

    def _check_inplace_aliasing(self) -> None:
        """Reject a field transformed in place that an earlier coupling also reads.

        An *in-place* coupling both reads and writes a field ``N`` (``N`` is among
        its inputs and is its output), so ``N`` denotes two values — a before and
        an after. The topo-sort always runs the writer ahead of any reader of
        ``N`` (the output-in-inputs edge), so a *different* coupling that reads
        ``N`` is silently fed the after-value. That is intended only when the
        reader was declared to consume the result — i.e. recorded *after* the
        in-place transform (the ``materialize(f)`` → ``map_fields(f → x)`` chain).
        A reader recorded *before* the transform wanted the before-value, which the
        single-column model cannot hold: the order is ambiguous.

        This is the forked-handle hazard:
        ``h2 = op(ds.field("f")); op(ds.field("f"), out="f")`` — ``h2`` reads ``f``
        expecting its original value, but the second op rewrites ``f`` in place.
        Reject it (fail-loud) rather than silently miscompute. Note this does not
        flag a normal out-of-order chain (``c = f(b); b = g(a)``): there ``b`` has
        a single producer that does not read ``b``, so it is unambiguous.
        """

        couplings = self.couplings.couplings
        for w_idx, writer in enumerate(couplings):
            target = writer.output_field()
            if target not in writer.input_fields():
                continue  # ordinary producer, not an in-place transform
            for r_idx, reader in enumerate(couplings):
                if r_idx >= w_idx:
                    continue  # recorded after the transform: meant to see the result
                if reader is writer or reader.output_field() == target:
                    continue  # same-output chain mate, not an external reader
                if target in reader.input_fields():
                    raise ValueError(
                        f"Field {target!r} is transformed in place by "
                        f"{type(writer).__name__}, but an earlier pending coupling "
                        f"({type(reader).__name__}) also reads {target!r}; the "
                        f"execution order is ambiguous (the reader was declared "
                        f"first, but topo-ordering runs the in-place write first). "
                        f"Write to a fresh field name, or consume the reader before "
                        f"overwriting {target!r}."
                    )

    def _compute_order(self) -> tuple[Coupling, ...]:
        """Topo-sort with insertion-order tie-breaking and chain semantics."""
        couplings = self.couplings.couplings
        n = len(couplings)
        edges: list[list[int]] = [[] for _ in range(n)]
        in_degree = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                i_out = couplings[i].output_field()
                j_out = couplings[j].output_field()
                if i_out == j_out:
                    if i < j:
                        edges[i].append(j)
                        in_degree[j] += 1
                elif i_out in couplings[j].input_fields():
                    edges[i].append(j)
                    in_degree[j] += 1

        available = [i for i in range(n) if in_degree[i] == 0]
        result: list[int] = []
        while available:
            available.sort()
            i = available.pop(0)
            result.append(i)
            for j in edges[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    available.append(j)

        if len(result) != n:
            cyclic = [type(couplings[i]).__name__ for i in range(n) if i not in result]
            raise ValueError(f"Couplings contain a cycle involving: {cyclic}")
        return tuple(couplings[i] for i in result)

    def couplings_for_column(self, column: str) -> tuple[Coupling, ...]:
        """Couplings needed to produce ``column`` — full chain plus transitive upstream."""
        chain = [c for c in self._order if c.output_field() == column]
        if not chain:
            return ()
        needed = {id(c) for c in chain}
        queue = list(chain)
        while queue:
            c = queue.pop()
            for input_name in c.input_fields():
                for producer in self._order:
                    if producer.output_field() != input_name:
                        continue
                    if id(producer) in needed:
                        continue
                    needed.add(id(producer))
                    queue.append(producer)
        return tuple(c for c in self._order if id(c) in needed)

    def couplings_up_to(self, target: Coupling) -> tuple[Coupling, ...]:
        """Couplings needed to compute up to and including ``target`` (no downstream).

        Walks transitive upstream of ``target`` restricted to couplings that come
        before ``target`` in topo order. Useful for partial consumption — e.g.
        applying a slice without then materializing.
        """
        if target not in self._order:
            raise ValueError("Target coupling is not in this engine's coupling set.")
        target_idx = self._order.index(target)
        needed = {id(target)}
        queue = [target]
        while queue:
            c = queue.pop()
            for input_name in c.input_fields():
                for i in range(target_idx):
                    producer = self._order[i]
                    if producer.output_field() != input_name:
                        continue
                    if id(producer) in needed:
                        continue
                    needed.add(id(producer))
                    queue.append(producer)
        return tuple(c for c in self._order if id(c) in needed)

    def find_producers(self, output: str) -> tuple[Coupling, ...]:
        """Return all couplings that write to ``output`` in topo order."""
        return tuple(c for c in self._order if c.output_field() == output)

    def apply_row(
        self, row: dict[str, Any], state: "DatasetState"
    ) -> dict[str, Any]:
        """Apply all couplings in topo-sorted order to a raw row dict."""
        for coupling in self._order:
            row = coupling.apply_row(row, state)
        return row
