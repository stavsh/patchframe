"""Tests for the OperatorSignature data model (Phase 4, declaration only)."""

from __future__ import annotations

import pytest

import patchframe as pf
from patchframe.dataset.fields import BundleField, DataField, DimensionedSliceField
from patchframe.ops.signature import (
    DatasetInput,
    DatasetReturn,
    FieldInput,
    FieldOutput,
    FieldReturn,
    OperatorSignature,
    SelectionInput,
    SelectionReturn,
)


def test_signature_classifies_field_and_dataset_slots_in_order():
    sig = OperatorSignature(
        inputs={
            "slice_field": FieldInput(field_type=DimensionedSliceField),
            "data_field": FieldInput(field_type=DataField),
        },
        returns=FieldReturn(),
    )

    assert sig.field_slots() == ("slice_field", "data_field")
    assert sig.dataset_slots() == ()
    assert isinstance(sig.returns, FieldReturn)


def test_signature_separates_dataset_slots():
    sig = OperatorSignature(
        inputs={
            "dataset": DatasetInput(),
            "predicate": FieldInput(),  # a field-scoped operand on the same dataset
        }
    )

    assert sig.dataset_slots() == ("dataset",)
    assert sig.field_slots() == ("predicate",)
    # Default return is the conservative Dataset return.
    assert isinstance(sig.returns, DatasetReturn)


def test_dataset_input_accepts_bundle_handle_by_default():
    assert DatasetInput().accepts_bundle_handle is True
    assert DatasetInput(accepts_bundle_handle=False).accepts_bundle_handle is False


def test_selection_slot_is_a_field_slot():
    sig = OperatorSignature(
        inputs={"on": SelectionInput()},
        returns=SelectionReturn(),
    )

    assert sig.field_slots() == ("on",)
    assert isinstance(sig.returns, SelectionReturn)


def test_specs_are_frozen():
    with pytest.raises(Exception):
        FieldInput().field_type = DataField  # type: ignore[misc]
    with pytest.raises(Exception):
        OperatorSignature().custom = True  # type: ignore[misc]


def test_bind_slice_signature_drives_its_field_slots():
    # slice_data is the migration proof: its field-operand slots come from the
    # signature, which the normalize-call machinery reads via _field_input_slots.
    assert pf.slice_data.signature is not None
    assert pf.slice_data.signature.field_slots() == ("slice_field", "data_field")
    assert pf.slice_data.instance()._field_input_slots() == ("slice_field", "data_field")
    # The legacy tuple is superseded.
    assert pf.slice_data.field_handle_inputs == ()


def test_metaclass_collects_field_inputs_from_class_attrs():
    # The operands are declared dataclass-style as class attributes; OperatorMeta
    # collects them (in definition order) into the built signature, the same way
    # it collects Parameter attributes.
    assert isinstance(pf.slice_data.slice_field, FieldInput)
    assert pf.slice_data.slice_field.field_type is DimensionedSliceField
    assert isinstance(pf.slice_data.data_field, FieldInput)
    assert pf.slice_data.data_field.field_type is DataField

    assert tuple(pf.slice_data.signature.inputs) == ("slice_field", "data_field")
    assert isinstance(pf.slice_data.signature.returns, FieldReturn)


def test_metaclass_collects_field_outputs_for_lifting_ops():
    # A lifting op declares a caller-named produced field (the chaining point);
    # OperatorMeta collects FieldOutput attrs into signature.outputs, mirroring
    # the FieldInput collection. (Acting on it is Phase 6.)
    class _Lift(pf.DatasetOperator):
        cell = FieldInput(field_type=BundleField)
        out = FieldOutput(field_type=BundleField)

    assert _Lift.signature.field_slots() == ("cell",)
    assert _Lift.signature.output_slots() == ("out",)
    assert isinstance(_Lift.signature.outputs["out"], FieldOutput)
    assert _Lift.signature.outputs["out"].field_type is BundleField


def test_bind_slice_in_place_output_has_no_field_output_slot():
    # In-place ops keep a returns kind and declare no FieldOutput.
    assert pf.slice_data.signature.output_slots() == ()
    assert isinstance(pf.slice_data.signature.returns, FieldReturn)
