"""
Bounds checking tests for CNDA Python bindings.

Tests the .at() method which provides bounds-checked array access,
ensuring IndexError is raised for out-of-bounds or rank-mismatched indices.
"""

import pytest
import cnda


# ==============================================================================
# .at() Method Tests
# ==============================================================================

class TestAtMethod:
    """Test the .at() method for safe access."""

    def test_at_in_bounds_2d(self):
        """Test .at() with in-range indices for 2D array."""
        arr = cnda.ContiguousND_float([3, 4])
        arr[1, 2] = 42.5
        assert arr.at((1, 2)) == 42.5

    def test_at_in_bounds_1d(self):
        """Test .at() with in-range indices for 1D array."""
        arr = cnda.ContiguousND_int32([5])
        arr[3] = 100
        assert arr.at((3,)) == 100

    def test_at_in_bounds_3d(self):
        """Test .at() with in-range inditest_bounds_checking_pythonces for 3D array."""
        arr = cnda.ContiguousND_double([2, 3, 4])
        arr[1, 2, 3] = 99.99
        assert arr.at((1, 2, 3)) == 99.99

    def test_at_out_of_bounds_first_dim(self):
        """Test .at() with out-of-bounds on first dimension raises IndexError."""
        arr = cnda.ContiguousND_float([3, 4])
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((3, 0))

    def test_at_out_of_bounds_second_dim(self):
        """Test .at() with out-of-bounds on second dimension raises IndexError."""
        arr = cnda.ContiguousND_float([3, 4])
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((0, 4))

    def test_at_out_of_bounds_negative_index(self):
        """Test .at() with negative index raises IndexError."""
        arr = cnda.ContiguousND_int32([3, 4])
        # Assuming negative indices are not supported and should raise IndexError
        with pytest.raises(IndexError):
            arr.at((-1, 0))

    def test_at_wrong_ndim_too_few(self):
        """Test .at() with too few indices raises IndexError."""
        arr = cnda.ContiguousND_float([3, 4])
        with pytest.raises(IndexError, match=r"at\(\): rank mismatch"):
            arr.at((1,))

    def test_at_wrong_ndim_too_many(self):
        """Test .at() with too many indices raises IndexError."""
        arr = cnda.ContiguousND_float([3, 4])
        with pytest.raises(IndexError, match=r"at\(\): rank mismatch"):
            arr.at((1, 2, 3))

    def test_at_boundary_values(self):
        """Test .at() at boundary indices (0 and max valid index)."""
        arr = cnda.ContiguousND_int32([3, 4])
        arr[0, 0] = 10
        arr[2, 3] = 20
        
        assert arr.at((0, 0)) == 10
        assert arr.at((2, 3)) == 20

    def test_at_multi_dimensional_out_of_bounds(self):
        """Test .at() with out-of-bounds on multiple dimensions."""
        arr = cnda.ContiguousND_int64([2, 3, 4])
        
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((2, 0, 0))
        
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((0, 3, 0))
        
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((0, 0, 4))

    def test_at_with_all_dtype_variants(self):
        """Test .at() works correctly with all data types."""
        # float
        arr_f = cnda.ContiguousND_float([2, 2])
        arr_f[0, 0] = 1.5
        assert arr_f.at((0, 0)) == 1.5
        
        # double
        arr_d = cnda.ContiguousND_double([2, 2])
        arr_d[0, 0] = 2.5
        assert arr_d.at((0, 0)) == 2.5
        
        # int32
        arr_i32 = cnda.ContiguousND_int32([2, 2])
        arr_i32[0, 0] = 42
        assert arr_i32.at((0, 0)) == 42
        
        # int64
        arr_i64 = cnda.ContiguousND_int64([2, 2])
        arr_i64[0, 0] = 1000000
        assert arr_i64.at((0, 0)) == 1000000

    def test_at_on_zero_sized_array(self):
        """Test .at() on array with zero-sized dimension."""
        arr = cnda.ContiguousND_int32([0, 5])
        assert arr.size() == 0
        
        # Any access to zero-sized array should raise IndexError
        with pytest.raises(IndexError):
            arr.at((0, 0))

    def test_at_single_element_array(self):
        """Test .at() on single-element array."""
        arr = cnda.ContiguousND_float([1])
        arr[0] = 3.14
        assert arr.at((0,)) == 3.14
        
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((1,))

    def test_at_vs_direct_access(self):
        """Test that .at() returns same values as direct access for valid indices."""
        arr = cnda.ContiguousND_int32([3, 4])
        
        # Populate array
        for i in range(3):
            for j in range(4):
                arr[i, j] = i * 10 + j
        
        # Verify .at() matches direct access
        for i in range(3):
            for j in range(4):
                assert arr.at((i, j)) == arr[i, j]


# ==============================================================================
# Edge Cases and Integration Tests
# ==============================================================================

class TestBoundsCheckingEdgeCases:
    """Test edge cases for bounds checking."""

    def test_large_indices(self):
        """Test with very large out-of-bounds indices."""
        arr = cnda.ContiguousND_int32([2, 2])
        
        with pytest.raises(IndexError, match=r"at\(\): index out of bounds"):
            arr.at((1000000, 0))

    def test_empty_indices(self):
        """Test .at() with empty tuple on 0D-like array."""
        # Note: If 0D arrays are not supported, this test can be skipped
        # For now, test rank mismatch
        arr = cnda.ContiguousND_int32([5])
        
        with pytest.raises(IndexError, match=r"at\(\): rank mismatch"):
            arr.at(())

    def test_sequential_bounds_checks(self):
        """Test multiple bounds checks in sequence."""
        arr = cnda.ContiguousND_float([2, 3])
        
        # Valid access
        arr.at((0, 0))
        arr.at((1, 2))
        
        # Invalid access
        with pytest.raises(IndexError):
            arr.at((2, 0))
        
        # Valid access again to ensure state is correct
        arr.at((1, 1))