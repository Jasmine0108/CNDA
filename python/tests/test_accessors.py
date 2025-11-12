import pytest
import cnda

def test_import():
    """Make sure cnda module can be imported."""
    assert hasattr(cnda, "ContiguousND_int")

def test_create_and_basic_properties():
    """Test creation and basic properties"""
    arr = cnda.ContiguousND_int([2, 3])  # Create a 2x3 array
    shape = arr.shape()
    strides = arr.strides()
    ndim = arr.ndim()
    size = arr.size()

    assert shape == [2, 3]
    assert strides == [3, 1]
    assert ndim == 2
    assert size == 6

def test_set_and_get_item():
    """Test __getitem__ and __setitem__"""
    arr = cnda.ContiguousND_int([2, 3])
    arr[[0, 0]] = 42
    arr[[1, 2]] = 99

    assert arr[[0, 0]] == 42
    assert arr[[1, 2]] == 99

def test_data_method_returns_correct_length():
    """Test data() method returns correct length"""
    arr = cnda.ContiguousND_int([2, 3])
    data = arr.data()
    assert isinstance(data, list)
    assert len(data) == arr.size()

