"""
patchframe.dataset.fields

Core field definitions for patchframe schemas.

Fields describe what kind of values a column holds and carry dtype metadata used
for column-level validation. Cross-field relationships are expressed through
couplings; structural changes through operator transitions.

dtype inputs (builtin type, numpy dtype, or pandas dtype) are always stored as
the corresponding pandas nullable type, e.g. int -> pd.Int64Dtype().
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from patchframe.data.dimensions import Dimension
from patchframe.dataset.identity import FieldIdentity, IndexIdentity, new_field_identity

if TYPE_CHECKING:
    # Annotation-only: ``CompositeField`` holds a ``Schema`` instance, but
    # ``schema`` imports ``fields`` — so the runtime import would be circular.
    # ``from __future__ import annotations`` keeps the annotation a string.
    from patchframe.dataset.schema import Schema

# ---------------------------------------------------------------------------
# Dtype conversion helpers
# ---------------------------------------------------------------------------

_BUILTIN_TO_NULLABLE: dict[type, pd.api.extensions.ExtensionDtype] = {
    int: pd.Int64Dtype(),
    float: pd.Float64Dtype(),
    str: pd.StringDtype(),
    bool: pd.BooleanDtype(),
}

_NUMPY_TO_NULLABLE: dict[np.dtype, pd.api.extensions.ExtensionDtype] = {
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
    np.dtype("float32"): pd.Float32Dtype(),
    np.dtype("float64"): pd.Float64Dtype(),
    np.dtype("bool"): pd.BooleanDtype(),
    np.dtype("object"): pd.StringDtype(),
}

# Reverse map: nullable dtype type -> numpy equivalent (for lenient validation)
_NULLABLE_TO_NUMPY: dict[type, np.dtype] = {
    pd.Int8Dtype: np.dtype("int8"),
    pd.Int16Dtype: np.dtype("int16"),
    pd.Int32Dtype: np.dtype("int32"),
    pd.Int64Dtype: np.dtype("int64"),
    pd.UInt8Dtype: np.dtype("uint8"),
    pd.UInt16Dtype: np.dtype("uint16"),
    pd.UInt32Dtype: np.dtype("uint32"),
    pd.UInt64Dtype: np.dtype("uint64"),
    pd.Float32Dtype: np.dtype("float32"),
    pd.Float64Dtype: np.dtype("float64"),
    pd.BooleanDtype: np.dtype("bool"),
}


def to_nullable_dtype(dtype: Any) -> pd.api.extensions.ExtensionDtype:
    """Convert a builtin type, numpy dtype, or pandas dtype to pandas nullable dtype."""
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return dtype
    if dtype in _BUILTIN_TO_NULLABLE:
        return _BUILTIN_TO_NULLABLE[dtype]
    try:
        np_dtype = np.dtype(dtype)
    except TypeError as err:
        raise TypeError(f"Cannot convert {dtype!r} to a pandas nullable dtype") from err
    if np_dtype in _NUMPY_TO_NULLABLE:
        return _NUMPY_TO_NULLABLE[np_dtype]
    raise TypeError(f"Cannot convert {dtype!r} to a pandas nullable dtype")


def dtype_compatible(col_dtype: Any, target: pd.api.extensions.ExtensionDtype) -> bool:
    """Return True if col_dtype exactly matches target or is its numpy equivalent.

    StringDtype variants (na_value=<NA> vs na_value=nan) differ across pandas
    versions but are semantically equivalent — compare by storage backend only.
    """
    if col_dtype == target:
        return True
    if isinstance(col_dtype, pd.StringDtype) and isinstance(target, pd.StringDtype):
        return col_dtype.storage == target.storage
    numpy_equiv = _NULLABLE_TO_NUMPY.get(type(target))
    return numpy_equiv is not None and col_dtype == numpy_equiv


# ---------------------------------------------------------------------------
# Row-exit conversion registry
# ---------------------------------------------------------------------------

#: Registered exit conversions by field type — the fallback for decorating
#: field types you do not own. Resolved by MRO walk, taking precedence over
#: the field's own ``exit_value`` method; registering a base type therefore
#: covers its subclasses unless they register more specifically.
_FIELD_EXITS: dict[type, Any] = {}


def register_field_exit(field_type: type, fn: Any) -> None:
    """Register a row-exit conversion ``fn(field_def, value) -> Any`` for a field type.

    Row access (``ds[item_id]``) exits the dataset world; each value converts
    through its field's exit. The conversion is owned by the field type
    (override ``Field.exit_value``); this registry is the extension path for
    field types you do not own.
    """

    _FIELD_EXITS[field_type] = fn


def exit_value(field_def: "Field", value: Any) -> Any:
    """Convert one evaluated cell value through its field's exit conversion."""

    for klass in type(field_def).__mro__:
        fn = _FIELD_EXITS.get(klass)
        if fn is not None:
            return fn(field_def, value)
    return field_def.exit_value(value)


# ---------------------------------------------------------------------------
# Field hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Field:
    """Base field definition.

    Parameters
    ----------
    name:
        Column name in the dataset table.
    dtype:
        Expected column dtype. Accepts builtin types, numpy dtypes, or pandas
        dtypes; always stored as the corresponding pandas nullable type.
        ``None`` disables dtype validation for this field.
    nullable:
        Whether null values are allowed in the column.
    primary:
        Whether this is the primary field of its type in the schema. At most
        one primary field of each concrete Field type may exist in a Schema.
    metadata:
        Optional field-level metadata. Must not contain executable logic.
    field_identity:
        Stable semantic identity for this field across operator transitions.
        Minted automatically at construction when not supplied; preserved by
        ``dataclasses.replace``. Excluded from structural equality.
    """

    logical_type: ClassVar[str] = ""

    name: str
    dtype: pd.api.extensions.ExtensionDtype | None = None
    nullable: bool = True
    primary: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    field_identity: FieldIdentity | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.dtype is not None:
            object.__setattr__(self, "dtype", to_nullable_dtype(self.dtype))
        if self.field_identity is None:
            object.__setattr__(self, "field_identity", new_field_identity())

    def validate_column(self, series: pd.Series) -> None:
        """Raise if the series dtype is incompatible with this field's dtype."""
        if self.dtype is None:
            return
        if not dtype_compatible(series.dtype, self.dtype):
            raise ValueError(
                f"Field '{self.name}': expected dtype {self.dtype!r}, got {series.dtype!r}"
            )

    def exit_value(self, value: Any) -> Any:
        """Convert one evaluated cell value at the row-exit boundary.

        Row access (``ds[item_id]``) is the exit point from the dataset world:
        the dict it returns holds plain Python values. The conversion is owned
        by the field type — the default is identity; container-like fields
        override (``BundleField`` exports its fiber as records). For field
        types you do not own, ``register_field_exit`` takes precedence over
        this method.
        """

        return value

    def table_columns(self) -> tuple[str, ...]:
        """The table column name(s) this field occupies.

        The centralized field->column mapping: one column by default. An index
        field occupies the table *index* (``()``); a ``CompositeField`` spans its
        dotted columns. Operators that enumerate a field's columns (validate,
        keep, drop, rename) route through this rather than assuming ``field.name``.
        """

        return (self.name,)

    def rename_table_columns(self, new_name: str) -> dict[str, str]:
        """Table-column renames implied by renaming this field to ``new_name``."""

        return {self.name: new_name}

    def validate_in_table(self, table: pd.DataFrame) -> None:
        """Validate this field against the table — its own column by default."""

        self.validate_column(table[self.name])


@dataclass(frozen=True, slots=True)
class IndexField(Field):
    """Identity field. Always primary; never nullable."""

    logical_type: ClassVar[str] = "index"
    nullable: bool = False
    primary: bool = True
    identity: IndexIdentity | None = None

    def __post_init__(self) -> None:
        # super() is unreliable with slots=True; call parent explicitly.
        Field.__post_init__(self)
        object.__setattr__(self, "primary", True)

    def validate_column(self, series: pd.Series) -> None:
        """Validate the index: dtype, plus that the table index is named for it.

        The DataFrame index *is* the ``IndexField``, so the two names must agree —
        a dataset's row identity should be self-describing. ``Schema.validate_table``
        passes ``table.index.to_series()``, whose ``name`` is the index name.
        """

        Field.validate_column(self, series)
        if series.name != self.name:
            raise ValueError(
                f"IndexField '{self.name}': the table index must be named "
                f"{self.name!r}, got {series.name!r}."
            )

    def table_columns(self) -> tuple[str, ...]:
        return ()  # the primary index is the table index, not a column

    def rename_table_columns(self, new_name: str) -> dict[str, str]:
        return {}  # the index axis renames separately, not a column

    def validate_in_table(self, table: pd.DataFrame) -> None:
        self.validate_column(table.index.to_series())

    def ensure_index_names(self, table: pd.DataFrame) -> pd.DataFrame:
        """Name the table's index axis after this field (self-describing identity)."""

        if table.index.name != self.name:
            return table.rename_axis(self.name)
        return table

    def level_names(self) -> tuple[str, ...]:
        """The index level names — one for a single index (this field's name)."""

        return (self.name,)

    def to_data_fields(self) -> tuple["Field", ...]:
        """The data column field(s) this index becomes when reset to columns.

        A single index demotes to one ``IndexColumnField`` that *remembers* its
        old namespace (so it can be re-promoted or identity-aligned); a composite
        index demotes to its level fields. Used by ``reset_index`` and by
        ``set_index``'s demote — index->column is the field's own knowledge, not
        an operator's ``isinstance`` branch.
        """

        return (
            IndexColumnField(
                name=self.name,
                dtype=self.dtype,
                nullable=True,
                metadata=self.metadata,
                index_identity=self.identity,
                field_identity=self.field_identity,
            ),
        )


@dataclass(frozen=True, slots=True)
class IndexColumnField(Field):
    """Table-backed index values from a secondary dataset index."""

    logical_type: ClassVar[str] = "index_column"
    nullable: bool = True
    primary: bool = False
    index_identity: IndexIdentity | None = None

    def __post_init__(self) -> None:
        Field.__post_init__(self)
        object.__setattr__(self, "primary", False)


@dataclass(frozen=True, slots=True)
class ForeignIndexField(IndexColumnField):
    """Table-backed labels that reference another index identity."""

    logical_type: ClassVar[str] = "foreign_index"

    def __post_init__(self) -> None:
        IndexColumnField.__post_init__(self)
        if self.index_identity is None:
            raise ValueError("ForeignIndexField requires index_identity.")

    @property
    def target_identity(self) -> IndexIdentity:
        """Return the referenced index identity."""

        if self.index_identity is None:
            raise ValueError("ForeignIndexField requires index_identity.")
        return self.index_identity


@dataclass(frozen=True, slots=True)
class ValueField(Field):
    """Regular scalar or object column."""

    logical_type: ClassVar[str] = "value"


@dataclass(frozen=True, slots=True)
class DimensionField(Field):
    """Scalar table value interpreted by a concrete Dimension."""

    logical_type: ClassVar[str] = "dimension"
    _: KW_ONLY
    dimension: Dimension

    @classmethod
    def from_dim(cls, dimension: Dimension, name: str, **kwargs: Any) -> DimensionField:
        # TODO: once Dimension carries dtype info, default dtype from dimension.dtype here
        return cls(name=name, dimension=dimension, **kwargs)


@dataclass(frozen=True, slots=True)
class DataField(Field):
    """Column of lazy DataAccessor values.

    The column dtype in the table is always ``object`` (Python objects).
    ``dtype`` here describes the materialized array dtype, not the column dtype,
    and is not used for column-level validation.
    """

    logical_type: ClassVar[str] = "data"

    def validate_column(self, series: pd.Series) -> None:
        pass  # column stores DataAccessor objects; dtype validation not applicable here


@dataclass(frozen=True, slots=True)
class DimensionedSliceField(Field):
    """Column of DimensionedSlice values."""

    logical_type: ClassVar[str] = "dimensioned_slice"

    def validate_column(self, series: pd.Series) -> None:
        pass  # column stores DimensionedSlice objects


@dataclass(frozen=True, slots=True)
class BundleField(Field):
    """Column whose cells hold whole Datasets.

    The dataset-valued analogue of ``DataField``: a ``DataField`` cell holds an
    array (eager) or a ``DataAccessor`` (lazy); a ``BundleField`` cell holds a
    ``Dataset`` (eager) — the lazy ``DatasetAccessor`` form is future work. The
    column dtype is always ``object``; a not-yet-materialized cell is null.

    A ``BundleField`` carries a ``FieldIdentity`` like any field; that is the
    column's identity as a schema entity and is orthogonal to the row
    identities the contained datasets carry.
    """

    logical_type: ClassVar[str] = "bundle"

    def validate_column(self, series: pd.Series) -> None:
        pass  # cells hold Dataset objects (or null when unmaterialized)

    def exit_value(self, value: Any) -> Any:
        """Export a fiber as a list of records (recursively exited row dicts).

        Row access is the exit point, so a cell-resident sub-dataset leaves as
        plain Python: one dict per fiber row, each produced by the fiber's own
        row access — evaluation and exit compose recursively. A pending (null)
        cell exits as-is. When lazy fiber navigation is wanted, hold the fiber
        as a ``Dataset`` via the storage surface (``ds.table`` / ``ds["col"]``)
        instead.
        """

        from patchframe.dataset.dataset import Dataset

        if not isinstance(value, Dataset):
            return value
        return [value[label] for label in value.table.index]


@dataclass(frozen=True, slots=True)
class CompositeField(Field):
    """A schema-level grouping of several columns under one logical field.

    The un-boxed, column-axis sibling of ``BundleField``: where a ``BundleField``
    bundles *rows* into one column of sub-datasets (cardinality-changing),
    ``CompositeField`` groups *columns*. It points at an index-less ``sub_schema``
    whose fields describe N **native** table columns, named
    ``f"{name}.{sub.name}"`` (a ``location`` field over ``{lat, lon}`` -> columns
    ``location.lat``, ``location.lon``). Nothing is boxed, so the table stays
    pandas-native and cardinality is preserved.

    Only the ``CompositeField`` is a top-level schema field; its sub-fields are
    reached through it (``column_names`` / ``validate_columns``). The sub-schema
    must be index-less — it groups columns, not a standalone dataset.

    See ``docs/design/composite-field.md``. The index variant (``MultiIndex``
    levels rather than columns) is future work.
    """

    logical_type: ClassVar[str] = "composite"
    _: KW_ONLY
    sub_schema: "Schema"

    def __post_init__(self) -> None:
        Field.__post_init__(self)
        if self.primary:
            raise ValueError(
                f"CompositeField {self.name!r} cannot be primary / the index — it "
                "groups columns. A composite *index* is a separate type "
                "(CompositeIndexField), so the two register distinct composition "
                "policies."
            )
        if not self.sub_schema.fields:
            raise ValueError(f"CompositeField {self.name!r}: sub_schema must be non-empty.")
        for sub in self.sub_schema:
            if isinstance(sub, IndexField):
                raise ValueError(
                    f"CompositeField {self.name!r}: sub_schema must be index-less "
                    f"(found IndexField {sub.name!r}); it groups columns, not a dataset."
                )

    def column_names(self) -> tuple[str, ...]:
        """The native table column names this field spans: ``{name}.{sub}``."""

        return tuple(f"{self.name}.{sub.name}" for sub in self.sub_schema)

    def table_columns(self) -> tuple[str, ...]:
        return self.column_names()

    def rename_table_columns(self, new_name: str) -> dict[str, str]:
        return {
            f"{self.name}.{sub.name}": f"{new_name}.{sub.name}"
            for sub in self.sub_schema
        }

    def validate_in_table(self, table: pd.DataFrame) -> None:
        """Validate each spanned column against its sub-schema field."""

        for sub in self.sub_schema:
            sub.validate_column(table[f"{self.name}.{sub.name}"])

    def validate_column(self, series: pd.Series) -> None:
        # A CompositeField spans N columns; single-series validation cannot
        # apply. Schema.validate_table routes composites to validate_in_table.
        raise TypeError(
            f"CompositeField {self.name!r} spans multiple columns "
            f"({', '.join(self.column_names())}); validate with validate_in_table(table)."
        )


@dataclass(frozen=True, slots=True)
class CompositeIndexField(IndexField):
    """A composite row identity: an ordered set of pandas ``MultiIndex`` levels.

    The index-axis counterpart of ``CompositeField`` — a **separate type, not a
    subclass** (they compose differently and register distinct composition
    policies; the registry resolves by MRO). It subclasses ``IndexField``, so
    ``isinstance(., IndexField)`` and ``primary_index_field`` still return exactly
    one index field and the single-index call sites are untouched; the
    ``MultiIndex`` is encapsulated here. The index-less ``sub_schema`` fields *are*
    the levels (plain level names = sub-field names). One composite
    ``IndexIdentity`` for the whole tuple; the row identity is the tuple, unique.

    See ``docs/design/composite-field.md`` §2.
    """

    _: KW_ONLY
    sub_schema: "Schema"

    def __post_init__(self) -> None:
        IndexField.__post_init__(self)  # primary=True, mint field_identity
        if not self.sub_schema.fields:
            raise ValueError(f"CompositeIndexField {self.name!r}: sub_schema must be non-empty.")
        for sub in self.sub_schema:
            if isinstance(sub, IndexField):
                raise ValueError(
                    f"CompositeIndexField {self.name!r}: sub_schema must be index-less "
                    f"(found IndexField {sub.name!r}); it describes the index levels."
                )

    def level_names(self) -> tuple[str, ...]:
        """The ordered ``MultiIndex`` level names (the sub-field names)."""

        return tuple(sub.name for sub in self.sub_schema)

    def table_columns(self) -> tuple[str, ...]:
        return ()  # the composite is the index, not table columns

    def rename_table_columns(self, new_name: str) -> dict[str, str]:
        return {}  # a level rename is a sub-structure op (dedicated, future)

    def validate_in_table(self, table: pd.DataFrame) -> None:
        index = table.index
        expected = self.level_names()
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(
                f"CompositeIndexField {self.name!r}: the table index must be a "
                f"MultiIndex with levels {expected}, got a single-level index."
            )
        if tuple(index.names) != expected:
            raise ValueError(
                f"CompositeIndexField {self.name!r}: index levels "
                f"{tuple(index.names)} do not match the sub-schema {expected}."
            )
        for sub in self.sub_schema:
            sub.validate_column(index.get_level_values(sub.name).to_series())

    def ensure_index_names(self, table: pd.DataFrame) -> pd.DataFrame:
        expected = list(self.level_names())
        if list(table.index.names) != expected:
            return table.rename_axis(expected)
        return table

    def to_data_fields(self) -> tuple["Field", ...]:
        # Decompose to the level fields — a level that was a ForeignIndexField
        # keeps its reference, so a rollup after reset re-aligns by identity.
        return tuple(self.sub_schema.fields)
