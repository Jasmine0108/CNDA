import pytest
import cnda

# Python-side tests for AoS (Array-of-Structures) bindings.

def test_vec2f_basic_operations_binding():
    # Construction and initialization
    arr = cnda.ContiguousND_Vec2f([3, 4])
    assert arr.ndim() == 2
    assert arr.size() == 12
    assert arr.shape() == [3, 4]

    # Element access and modification
    arr = cnda.ContiguousND_Vec2f([2, 3])
    arr[0, 0] = cnda.Vec2f(1.0, 2.0)
    arr[0, 1] = cnda.Vec2f(3.0, 4.0)
    arr[1, 2] = cnda.Vec2f(5.0, 6.0)

    assert arr[0, 0].x == pytest.approx(1.0)
    assert arr[0, 0].y == pytest.approx(2.0)
    assert arr[0, 1].x == pytest.approx(3.0)
    assert arr[0, 1].y == pytest.approx(4.0)
    assert arr[1, 2].x == pytest.approx(5.0)
    assert arr[1, 2].y == pytest.approx(6.0)

    # Field access via reference
    arr = cnda.ContiguousND_Vec2f([2, 2])
    arr[0, 0] = cnda.Vec2f(10.0, 20.0)
    elem = arr[0, 0]
    assert elem.x == pytest.approx(10.0)
    assert elem.y == pytest.approx(20.0)
    elem.x = 30.0
    assert arr[0, 0].x == pytest.approx(30.0)

    # Memory layout is contiguous
    arr = cnda.ContiguousND_Vec2f([2, 3])
    for i in range(2):
        for j in range(3):
            v = float(i * 10 + j)
            arr[i, j] = cnda.Vec2f(v, v + 100.0)

    # Verify flat data access
    # Can read back in expected linear order, 
    # indicating underlying storage is contiguous without unexpected rearrangement.
    flat = arr.data()
    assert flat[0].x == pytest.approx(0.0)
    assert flat[0].y == pytest.approx(100.0)
    assert flat[1].x == pytest.approx(1.0)
    assert flat[5].x == pytest.approx(12.0)


def test_vec3f_basic_operations_binding():
    arr = cnda.ContiguousND_Vec3f([10, 10, 10])
    assert arr.ndim() == 3
    assert arr.size() == 1000

    arr[1, 1, 1] = cnda.Vec3f(1.5, 2.5, 3.5)
    assert arr[1, 1, 1].x == pytest.approx(1.5)
    assert arr[1, 1, 1].y == pytest.approx(2.5)
    assert arr[1, 1, 1].z == pytest.approx(3.5)


def test_cell2d_fluid_example_binding():
    grid = cnda.ContiguousND_Cell2D([5, 5])
    assert grid.size() == 25
    #Initialize boundaries and fluid cells
    for i in range(5):
        for j in range(5):
            is_boundary = (i == 0 or i == 4 or j == 0 or j == 4)
            u = 0.0 if is_boundary else 1.0
            v = 0.0 if is_boundary else 0.5
            flag = -1 if is_boundary else 1
            grid[i, j] = cnda.Cell2D(u, v, flag)
        
    # Verify grid[i,j] = cnda.Cell2D(...) can be used to write the entire POD struct into the array correctly
    # Verify grid[i, j].<field> can correctly read back the field values just written
    assert grid[0, 0].flag == -1
    assert grid[2, 2].u == pytest.approx(1.0)
    assert grid[2, 2].v == pytest.approx(0.5)
    assert grid[2, 2].flag == 1


def test_cell3d_basic_binding():
    grid = cnda.ContiguousND_Cell3D([10, 10, 10])
    assert grid.size() == 1000

    grid[1, 1, 1] = cnda.Cell3D(1.0, 2.0, 3.0, 1)
    assert grid[1, 1, 1].u == pytest.approx(1.0)
    assert grid[1, 1, 1].v == pytest.approx(2.0)
    assert grid[1, 1, 1].w == pytest.approx(3.0)
    assert grid[1, 1, 1].flag == 1


def test_particle_system_binding():
    particles = cnda.ContiguousND_Particle([5])
    assert particles.size() == 5

    dt = 0.1
    for i in range(5):
        particles[i] = cnda.Particle(0.0, 0.0, 0.0, float(i), 0.0, 0.0, 1.0)
        particles[i].x += particles[i].vx * dt

    assert particles[0].x == pytest.approx(0.0)
    assert particles[1].x == pytest.approx(0.1)
    assert particles[4].x == pytest.approx(0.4)
    assert particles[1].vx == pytest.approx(1.0)
    assert particles[0].mass == pytest.approx(1.0)


def test_materialpoint_grid_binding():
    grid = cnda.ContiguousND_MaterialPoint([3, 3])
    assert grid.size() == 9

    grid[0, 0] = cnda.MaterialPoint(1.0, 300.0, 101.3, 1)
    grid[1, 1] = cnda.MaterialPoint(1000.0, 293.0, 101.3, 2)
    grid[2, 2] = cnda.MaterialPoint(7850.0, 293.0, 101.3, 3)

    assert grid[0, 0].density == pytest.approx(1.0)
    assert grid[1, 1].density == pytest.approx(1000.0)
    assert grid[2, 2].density == pytest.approx(7850.0)
    assert grid[0, 0].id == 1
    assert grid[1, 1].id == 2
    assert grid[2, 2].id == 3


def test_at_bounds_checking_binding():
    # at() method with bounds checking
    # Verify `at()` enforces bounds checking (raises on invalid indices) and
    # Return correct field values for valid accesses.
    arr = cnda.ContiguousND_Vec2f([2, 3])
    arr[0, 0] = cnda.Vec2f(1.0, 2.0)
    assert arr.at([0, 0]).x == pytest.approx(1.0)

    grid = cnda.ContiguousND_Cell2D([5, 5])
    grid[2, 3] = cnda.Cell2D(1.5, 2.5, 10)
    assert grid.at([2, 3]).u == pytest.approx(1.5)
    assert grid.at([2, 3]).v == pytest.approx(2.5)
    assert grid.at([2, 3]).flag == 10

    # out-of-range indices should raise an IndexError on the Python side
    with pytest.raises(IndexError):
        _ = arr.at([2, 0])
    with pytest.raises(IndexError):
        _ = arr.at([0, 3])
    with pytest.raises(IndexError):
        _ = arr.at([5, 5])


def test_memory_size_validation_binding():

    # Verify `sizeof` for AoS types match the expected scalar multiples using
    #cnda.sizeof_aos` (C++ sizeof exposed) and Python ctypes.sizeof
    import ctypes

    # Basic scalar expectations
    assert cnda.sizeof_aos("Vec2f") == 2 * ctypes.sizeof(ctypes.c_float)
    assert cnda.sizeof_aos("Vec3f") == 3 * ctypes.sizeof(ctypes.c_float)
    assert cnda.sizeof_aos("Particle") == 7 * ctypes.sizeof(ctypes.c_double)

    # Compound types must be at least the sum of their components (allowing possible padding)
    assert cnda.sizeof_aos("Cell2D") >= 2 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
    assert cnda.sizeof_aos("Cell3D") >= 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
    assert cnda.sizeof_aos("MaterialPoint") >= 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)

    # Verify contiguous-like mapping by writing via index and reading back from `data()`
    arr = cnda.ContiguousND_Vec2f([100])
    for i in range(100):
        arr[i] = cnda.Vec2f(float(i), float(i + 100.0))

    flat = arr.data()
    assert flat[0].x == pytest.approx(0.0)
    assert flat[99].x == pytest.approx(99.0)
    # Ensure element retrieved from data() matches indexed access
    assert flat[50].x == pytest.approx(arr[50].x)


def test_pod_type_guarantees_binding():
    # Use ctypes to memcpy between two arrays of the same POD layout
    # Validate fields match after the raw copy 
    # Mirror std::memcpy usage in the C++ test

    import ctypes

    class Vec2f_ct(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    # Create source and destination arrays (2x2)
    src = (Vec2f_ct * 4)()
    dst = (Vec2f_ct * 4)()

    # Initialize src entries like the C++ test
    src[0].x, src[0].y = 1.0, 2.0
    src[3].x, src[3].y = 3.0, 4.0

    # Raw memory copy
    ctypes.memmove(ctypes.addressof(dst), ctypes.addressof(src), ctypes.sizeof(src))

    # Validate fields survived the memcpy
    assert dst[0].x == pytest.approx(1.0)
    assert dst[0].y == pytest.approx(2.0)
    assert dst[3].x == pytest.approx(3.0)
    assert dst[3].y == pytest.approx(4.0)
