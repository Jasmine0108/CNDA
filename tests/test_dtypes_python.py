import pytest
import cnda

DTYPES = [
    cnda.ContiguousND_float,
    cnda.ContiguousND_double,
    cnda.ContiguousND_int32,
    cnda.ContiguousND_int64,
]

@pytest.mark.parametrize("ArrayType", DTYPES)
def test_create_and_basic_properties(ArrayType):
    """Test creation and basic properties (moved from test_accessors_python)."""
    arr = ArrayType([2, 3])

    assert arr.shape() == [2, 3]
    assert arr.strides() == [3, 1]
    assert arr.ndim() == 2
    assert arr.size() == 6
    # cross-check index() mapping
    off = arr.index([1, 1])
    assert off == 4


@pytest.mark.parametrize("ArrayType", DTYPES)
def test_set_and_get_item(ArrayType):
    """Test __getitem__ and __setitem__ (moved from test_accessors_python)."""
    arr = ArrayType([2, 3])
    # write/read using operator
    arr[[0, 0]] = 42
    arr[[1, 2]] = 99

    assert arr[[0, 0]] == 42
    assert arr[[1, 2]] == 99


@pytest.mark.parametrize("ArrayType", DTYPES)
def test_data_method_returns_correct_length(ArrayType):
    """Test data() method returns correct length (moved from test_accessors_python)."""
    arr = ArrayType([2, 3])
    data = arr.data()

    assert isinstance(data, list)
    assert len(data) == arr.size()
