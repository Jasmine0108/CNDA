import pytest
import cnda

def test_import_module_has_types():
    """Module import sanity check (moved from test_accessors_python)."""
    assert hasattr(cnda, "ContiguousND_int32")
    assert hasattr(cnda, "ContiguousND_int64")
    assert hasattr(cnda, "ContiguousND_float")
    assert hasattr(cnda, "ContiguousND_double")


def test_zero_sized_dimension_and_data_callable():
    """Zero-sized dimension should yield size 0 and data() callable."""
    A = cnda.ContiguousND_int32([0, 7])
    assert A.size() == 0
    assert A.ndim() == 2
    # data() should be callable and return a list (possibly empty)
    d = A.data()
    assert isinstance(d, list)


def test_operator_1d_and_3d_read_write():
    """Test operator() for 1D and 3D arrays (read/write)."""
    # 1D
    a1 = cnda.ContiguousND_int32([5])
    for i in range(5):
        a1[i] = i * 10
    assert a1[0] == 0
    assert a1[4] == 40

    # 3D
    a3 = cnda.ContiguousND_int32([2, 3, 4])
    v = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                a3[i, j, k] = v
                v += 1
    # With strides [12,4,1], (1,1,2) -> 1*12 + 1*4 + 2 = 18
    assert a3[1, 1, 2] == 18


