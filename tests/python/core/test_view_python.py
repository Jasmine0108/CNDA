import pytest
import cnda


def test_non_owning_view_constructor_basic_functionality():
    '''
    Test: basic construction of a non-owning view from an external buffer.
    Expectation: verify ndim, size, shape, strides, and is_view() return values.
    '''
    buf = list(range(12))
    view = cnda.make_view([3, 4], buf, dtype="int32")

    assert view.ndim() == 2
    assert view.size() == 12
    sh = view.shape()
    assert sh[0] == 3
    assert sh[1] == 4
    strd = view.strides()
    assert strd[0] == 4
    assert strd[1] == 1
    assert view.is_view() is True


def test_non_owning_view_reads_correct_values_from_external_buffer():
    '''
    Test: a non-owning view reads values correctly from an external double buffer.
    Expectation: multidimensional indexing returns expected float values.
    '''
    buf = [float(i * 10) for i in range(12)]
    view = cnda.make_view([3, 4], buf, dtype="double")

    assert view[0, 0] == 0.0
    assert view[0, 1] == 10.0
    assert view[1, 2] == 60.0
    assert view[2, 3] == 110.0


def test_non_owning_view_can_modify_external_buffer():
    '''
    Test: modifying the external buffer through the view updates the original data.
    Expectation: writes are observable via data() and index().
    '''
    buf = [0] * 12
    view = cnda.make_view([3, 4], buf, dtype="int32")

    view[0, 0] = 42
    view[1, 2] = 99
    view[2, 3] = 777

    # Use index() to compute flat offsets
    assert view.data()[0] == 42
    assert view.data()[view.index([1, 2])] == 99
    assert view.data()[view.index([2, 3])] == 777


def test_non_owning_view_shares_data_with_external_buffer():
    '''
    Test: multiple non-owning views sharing the same external buffer see each other's changes.
    Expectation: modifying one view is reflected in the other view sharing the buffer.
    '''
    buf = [1, 2, 3, 4, 5, 6]
    v1, v2 = cnda.make_two_views([2, 3], [6], buf, dtype="int32")

    assert v1.is_view() is True
    assert v2.is_view() is True

    assert v1[0, 0] == 1
    assert v2[0] == 1

    v1[1, 2] = 999

    assert v2[5] == 999


def test_non_owning_view_data_points_to_external_buffer():
    '''
    Test: verify that data() content matches the initialized buffer.
    Note: current binding's data() returns a Python list copy, so compare by value.
    '''
    buf = [10, 20, 30, 40]
    view = cnda.make_view([2, 2], buf, dtype="int32")

    # data() returns a python list copy of the underlying buffer in these bindings
    data = view.data()
    assert data[0] == 10
    assert data[3] == 40


def test_non_owning_view_with_different_shapes():
    '''
    Test: behavior of non-owning views with different shapes (1D, 3D, 4D).
    Expectation: verify ndim, size, strides, and multi-dimensional indexing and writes.
    '''
    # 1D view
    buf1 = [1, 2, 3, 4, 5]
    v1 = cnda.make_view([5], buf1, dtype="int32")
    assert v1.ndim() == 1
    assert v1.size() == 5
    assert v1.strides()[0] == 1
    assert v1.is_view() is True
    assert v1[0] == 1
    assert v1[4] == 5

    # 3D view
    buf3 = list(range(24))
    v3 = cnda.make_view([2, 3, 4], buf3, dtype="int32")
    assert v3.ndim() == 3
    assert v3.size() == 24
    s = v3.strides()
    assert s[0] == 12
    assert s[1] == 4
    assert s[2] == 1
    assert v3.is_view() is True
    assert v3[1, 1, 2] == 18

    # 4D view
    buf4 = [0.0] * 120
    v4 = cnda.make_view([2, 3, 4, 5], buf4, dtype="double")
    assert v4.ndim() == 4
    assert v4.size() == 120
    assert v4.is_view() is True
    v4[1, 2, 3, 4] = 3.14
    # last element index 119
    assert v4.data()[119] == pytest.approx(3.14)


def test_owning_constructor_is_view_false():
    '''
    Test: an owning ContiguousND constructed normally should report is_view() == False.
    Also check that data() returns a non-empty container.
    '''
    owned = cnda.ContiguousND_int32([3, 4])
    assert owned.is_view() is False
    assert len(owned.data()) > 0


def test_non_owning_view_index_works():
    '''
    Test: index([...]) returns the correct flat offset and can be used with data().
    '''
    buf = [i * 100 for i in range(12)]
    view = cnda.make_view([3, 4], buf, dtype="int32")

    assert view.index([0, 0]) == 0
    assert view.index([1, 2]) == 6
    assert view.index([2, 3]) == 11
    assert view.data()[view.index([1, 2])] == 600


def test_non_owning_view_with_const_access():
    '''
    Test: const-like access of the view (read-only semantics in tests); Python does not enforce const.
    '''
    buf = [1, 2, 3, 4, 5, 6]
    view = cnda.make_view([2, 3], buf, dtype="int32")
    # Python doesn't enforce constness; ensure reads work
    assert view[0, 0] == 1
    assert view[1, 2] == 6
    assert view.data()[0] == 1
    assert view.is_view() is True


def test_multiple_non_owning_views_can_share_same_buffer():
    '''
    Test: multiple non-owning views sharing the same buffer reflect modifications across views.
    '''
    buf = [10, 20, 30, 40, 50, 60]
    v1, v2 = cnda.make_two_views([2, 3], [6], buf, dtype="int32")
    assert v1.is_view() is True
    assert v2.is_view() is True
    assert v1[0, 0] == 10
    assert v2[0] == 10
    v1[1, 2] = 999
    assert v2[5] == 999


def test_non_owning_view_with_zero_sized_dimension():
    '''
    Test: create a view with a zero-sized dimension and verify size, ndim, and shape.
    '''
    buf = []
    view = cnda.make_view([0, 5], buf, dtype="int32")
    assert view.is_view() is True
    assert view.size() == 0
    assert view.ndim() == 2
    sh = view.shape()
    assert sh[0] == 0
    assert sh[1] == 5
