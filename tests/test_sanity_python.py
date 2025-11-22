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

def test_variadic_4d_and_5d_and_const_access():
    """Test variadic N-dimensional operator() for 4D/5D and const correctness."""
    # 4D int
    a4 = cnda.ContiguousND_int32([2, 3, 4, 5])
    a4[0, 0, 0, 0] = 1000
    a4[1, 2, 3, 4] = 9999
    a4[0, 1, 2, 3] = 5555
    assert a4[0, 0, 0, 0] == 1000
    assert a4[1, 2, 3, 4] == 9999
    assert a4[0, 1, 2, 3] == 5555

    # 5D float
    a5 = cnda.ContiguousND_float([2, 2, 2, 2, 2])
    a5[0, 0, 0, 0, 0] = 1.5
    a5[1, 1, 1, 1, 1] = 2.5
    assert a5[0, 0, 0, 0, 0] == 1.5
    assert a5[1, 1, 1, 1, 1] == 2.5

    # const correctness: read via const reference
    # create an int4D and write then read via a const reference
    a4b = cnda.ContiguousND_int32([2, 3, 4, 5])
    a4b[1, 2, 3, 4] = 777
    ca = a4b
    assert ca[1, 2, 3, 4] == 777

def test_edge_cases_single_element_and_all_ones():
    """Edge cases: single-element arrays and all-dimensions-equal-to-one."""
    s = cnda.ContiguousND_int32([1])
    s[0] = 55
    assert s[0] == 55

    ones = cnda.ContiguousND_int32([1, 1, 1, 1])
    ones[0, 0, 0, 0] = 88
    assert ones[0, 0, 0, 0] == 88
    assert ones.size() == 1


