import ctypes
import pytest
import cnda

def test_vec2f_stride_and_indexing():
    arr = cnda.ContiguousND_Vec2f([3, 4])
    strides = arr.strides()

    assert len(strides) == 2
    assert strides[0] == 4
    assert strides[1] == 1
    assert arr.index([1, 2]) == 6

    # Now verify actual addresses using pointer helpers exposed from C++
    elem_size = ctypes.sizeof(ctypes.c_float) * 2  # Vec2f: 2 floats
    base_addr = arr.data_ptr()
    elem_addr = arr.element_ptr([1, 2])
    assert elem_addr == base_addr + 6 * elem_size


def test_cell3d_pointer_arithmetic():
    arr = cnda.ContiguousND_Cell3D([2, 3])
    strides = arr.strides()
    assert strides[0] == 3
    assert strides[1] == 1

    arr[1, 2] = cnda.Cell3D(1.0, 2.0, 3.0, 4)
    idx = arr.index([1, 2])
    assert idx == 5

    # byte offset using pointer helpers
    elem_size = 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
    base_addr = arr.data_ptr()
    elem_addr = arr.element_ptr([1, 2])
    assert elem_addr == base_addr + idx * elem_size

    # field values are visible
    cell = arr[1, 2]
    assert cell.u == pytest.approx(1.0)
    assert cell.v == pytest.approx(2.0)
    assert cell.w == pytest.approx(3.0)
    assert cell.flag == 4


def test_particle_array_memory_continuity():
    particles = cnda.ContiguousND_Particle([10])
    assert particles.strides()[0] == 1

    for i in range(10):
        particles[i] = cnda.Particle(float(i), float(i * 2), float(i * 3), float(i * 4), float(i * 5), float(i * 6), 1.0)

    elem_size = 7 * ctypes.sizeof(ctypes.c_double)  # 7 doubles
    base_addr = particles.data_ptr()
    for i in range(10):
        idx = particles.index([i])
        assert idx == i
        elem_addr = particles.element_ptr([i])
        assert elem_addr == base_addr + i * elem_size
        assert particles[i].x == pytest.approx(float(i))
        assert particles[i].y == pytest.approx(float(i * 2))


def test_3d_array_element_spacing():
    arr = cnda.ContiguousND_Vec3f([2, 3, 4])
    strides = arr.strides()

    assert strides[0] == 12
    assert strides[1] == 4
    assert strides[2] == 1

    # test some element indices
    assert arr.index([0, 0, 0]) == 0
    assert arr.index([0, 0, 1]) == 1
    assert arr.index([0, 1, 0]) == 4
    assert arr.index([1, 0, 0]) == 12
    assert arr.index([1, 2, 3]) == 23

    elem_size = 3 * ctypes.sizeof(ctypes.c_float)
    base_addr = arr.data_ptr()
    elem_addr = arr.element_ptr([1, 2, 3])
    assert elem_addr == base_addr + 23 * elem_size
    assert arr.index([1, 2, 3]) == 23


def test_aos_vs_scalar_sizeof_comparison():
    # To verify that AoS and scalar arrays of same shape have different total byte sizes
    scalar_arr = cnda.ContiguousND_float([3, 4])
    aos_arr = cnda.ContiguousND_Vec2f([3, 4])

    assert scalar_arr.strides() == aos_arr.strides()
    assert scalar_arr.size() == aos_arr.size()

    # scalar size * sizeof(float) == 48
    assert scalar_arr.size() * ctypes.sizeof(ctypes.c_float) == 48
    # aos array total bytes
    aos_elem_size = 2 * ctypes.sizeof(ctypes.c_float)
    assert aos_arr.size() * aos_elem_size == 96


def test_pointer_arithmetic_scales_by_sizeof_T():
    arr = cnda.ContiguousND_Cell3D([5, 5])

    # verify element index distances correspond to expected element counts
    # Get actual addresses and verify byte differences scale by sizeof(T)
    p0_addr = arr.element_ptr([0, 0])
    p1_addr = arr.element_ptr([0, 1])
    p2_addr = arr.element_ptr([1, 0])
    # int_32 is for the flag field
    assert p1_addr - p0_addr == ctypes.sizeof(ctypes.c_float) * 3 + ctypes.sizeof(ctypes.c_int32)
    assert p2_addr - p0_addr == 5 * (ctypes.sizeof(ctypes.c_float) * 3 + ctypes.sizeof(ctypes.c_int32))
