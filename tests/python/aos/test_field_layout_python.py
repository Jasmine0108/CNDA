import ctypes
import pytest
import cnda

# ctypes mirrors for offsetof/sizeof validation
class Vec2f_ct(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class Vec3f_ct(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class Cell2D_ct(ctypes.Structure):
    _fields_ = [("u", ctypes.c_float), ("v", ctypes.c_float), ("flag", ctypes.c_int32)]


class Cell3D_ct(ctypes.Structure):
    _fields_ = [("u", ctypes.c_float), ("v", ctypes.c_float), ("w", ctypes.c_float), ("flag", ctypes.c_int32)]


class Particle_ct(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("z", ctypes.c_double),
                ("vx", ctypes.c_double), ("vy", ctypes.c_double), ("vz", ctypes.c_double),
                ("mass", ctypes.c_double)]


class MaterialPoint_ct(ctypes.Structure):
    _fields_ = [("density", ctypes.c_float), ("temperature", ctypes.c_float), ("pressure", ctypes.c_float), ("id", ctypes.c_int32)]


def test_field_offset_validation_ctypes():
    # Field memory offsets and per-field offsetof validation using ctypes

    # Vec2f
    assert getattr(Vec2f_ct, "x").offset == 0
    assert getattr(Vec2f_ct, "y").offset == ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Vec2f_ct) == 2 * ctypes.sizeof(ctypes.c_float)

    # Vec3f
    assert getattr(Vec3f_ct, "x").offset == 0
    assert getattr(Vec3f_ct, "y").offset == ctypes.sizeof(ctypes.c_float)
    assert getattr(Vec3f_ct, "z").offset == 2 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Vec3f_ct) == 3 * ctypes.sizeof(ctypes.c_float)

    # Cell2D
    assert getattr(Cell2D_ct, "u").offset == 0
    assert getattr(Cell2D_ct, "v").offset == ctypes.sizeof(ctypes.c_float)
    assert getattr(Cell2D_ct, "flag").offset == 2 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Cell2D_ct) == 2 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)

    # Cell3D
    assert getattr(Cell3D_ct, "u").offset == 0
    assert getattr(Cell3D_ct, "v").offset == ctypes.sizeof(ctypes.c_float)
    assert getattr(Cell3D_ct, "w").offset == 2 * ctypes.sizeof(ctypes.c_float)
    assert getattr(Cell3D_ct, "flag").offset == 3 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Cell3D_ct) == 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)

    # Particle (doubles)
    assert getattr(Particle_ct, "x").offset == 0
    assert getattr(Particle_ct, "y").offset == ctypes.sizeof(ctypes.c_double)
    assert getattr(Particle_ct, "z").offset == 2 * ctypes.sizeof(ctypes.c_double)
    assert ctypes.sizeof(Particle_ct) == 7 * ctypes.sizeof(ctypes.c_double)

    # MaterialPoint
    assert getattr(MaterialPoint_ct, "density").offset == 0
    assert getattr(MaterialPoint_ct, "temperature").offset == ctypes.sizeof(ctypes.c_float)
    assert getattr(MaterialPoint_ct, "pressure").offset == 2 * ctypes.sizeof(ctypes.c_float)
    assert getattr(MaterialPoint_ct, "id").offset == 3 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(MaterialPoint_ct) == 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)


def test_field_access_api_consistency_binding():
    # Field Access API Consistency: low-level checks using element_ptr and ctypes

    assert hasattr(cnda.ContiguousND_Vec2f, 'element_ptr') or hasattr(cnda.ContiguousND_Vec2f, 'element_ptr')

    # Vec2f: assign and verify via element_ptr -> float* arithmetic
    arr = cnda.ContiguousND_Vec2f([1])
    arr[0] = cnda.Vec2f(1.5, 2.5)
    p = arr.element_ptr(0)
    float_p = ctypes.cast(ctypes.c_void_p(p), ctypes.POINTER(ctypes.c_float))
    assert float_p[0] == pytest.approx(1.5)
    assert float_p[1] == pytest.approx(2.5)

    x_addr = p + getattr(Vec2f_ct, 'x').offset
    y_addr = p + getattr(Vec2f_ct, 'y').offset
    assert (y_addr - x_addr) == ctypes.sizeof(ctypes.c_float)

    # Vec3f: float pointer arithmetic
    arr3 = cnda.ContiguousND_Vec3f([1])
    arr3[0] = cnda.Vec3f(10.0, 20.0, 30.0)
    p3 = arr3.element_ptr(0)
    float_p3 = ctypes.cast(ctypes.c_void_p(p3), ctypes.POINTER(ctypes.c_float))
    assert float_p3[0] == pytest.approx(10.0)
    assert float_p3[1] == pytest.approx(20.0)
    assert float_p3[2] == pytest.approx(30.0)
    
    assert arr3[0].x == pytest.approx(10.0)
    assert arr3[0].y == pytest.approx(20.0)
    assert arr3[0].z == pytest.approx(30.0)

    # Cell2D: floats then int32 flag at offset
    c2 = cnda.ContiguousND_Cell2D([1])
    c2[0] = cnda.Cell2D(5.5, 6.5, 42)
    pc2 = c2.element_ptr(0)
    float_pc2 = ctypes.cast(ctypes.c_void_p(pc2), ctypes.POINTER(ctypes.c_float))
    assert float_pc2[0] == pytest.approx(5.5)
    assert float_pc2[1] == pytest.approx(6.5)
    flag_addr = pc2 + getattr(Cell2D_ct, 'flag').offset
    flag_val = ctypes.cast(ctypes.c_void_p(flag_addr), ctypes.POINTER(ctypes.c_int32)).contents.value
    assert flag_val == 42

    # Cell3D: floats then int32 flag
    c3 = cnda.ContiguousND_Cell3D([1])
    c3[0] = cnda.Cell3D(1.0, 2.0, 3.0, 99)
    pc3 = c3.element_ptr(0)
    float_pc3 = ctypes.cast(ctypes.c_void_p(pc3), ctypes.POINTER(ctypes.c_float))
    assert float_pc3[0] == pytest.approx(1.0)
    assert float_pc3[1] == pytest.approx(2.0)
    assert float_pc3[2] == pytest.approx(3.0)
    flag_addr3 = pc3 + getattr(Cell3D_ct, 'flag').offset
    flag_val3 = ctypes.cast(ctypes.c_void_p(flag_addr3), ctypes.POINTER(ctypes.c_int32)).contents.value
    assert flag_val3 == 99

    # Particle: use double pointer arithmetic
    p_arr = cnda.ContiguousND_Particle([1])
    p_arr[0] = cnda.Particle(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    pp = p_arr.element_ptr(0)
    double_pp = ctypes.cast(ctypes.c_void_p(pp), ctypes.POINTER(ctypes.c_double))
    assert double_pp[0] == pytest.approx(1.0)
    assert double_pp[1] == pytest.approx(2.0)
    assert double_pp[2] == pytest.approx(3.0)
    assert double_pp[3] == pytest.approx(4.0)
    assert double_pp[4] == pytest.approx(5.0)
    assert double_pp[5] == pytest.approx(6.0)
    assert double_pp[6] == pytest.approx(7.0)

    # MaterialPoint: floats and int32 id
    m = cnda.ContiguousND_MaterialPoint([1])
    m[0] = cnda.MaterialPoint(1.0, 300.0, 101.3, 1)
    pm = m.element_ptr(0)
    float_pm = ctypes.cast(ctypes.c_void_p(pm), ctypes.POINTER(ctypes.c_float))
    assert float_pm[0] == pytest.approx(1.0)
    assert float_pm[1] == pytest.approx(300.0)
    assert float_pm[2] == pytest.approx(101.3)
    id_addr = pm + getattr(MaterialPoint_ct, 'id').offset
    id_val = ctypes.cast(ctypes.c_void_p(id_addr), ctypes.POINTER(ctypes.c_int32)).contents.value
    assert id_val == 1


def test_field_mixup_prevention_binding():
    # Ensure fields are distinct and not mixed between elements or fields

    # Vec2f distinct
    a = cnda.ContiguousND_Vec2f([2])
    a[0] = cnda.Vec2f(1.0, 2.0)
    a[1] = cnda.Vec2f(10.0, 20.0)
    assert a[0].x != a[0].y
    assert a[1].x != a[1].y
    # Values should be correct
    assert a[0].x == pytest.approx(1.0)
    assert a[0].y == pytest.approx(2.0)

    # Vec3f distinct
    b = cnda.ContiguousND_Vec3f([2])
    b[0] = cnda.Vec3f(1.0, 2.0, 3.0)
    b[1] = cnda.Vec3f(10.0, 20.0, 30.0)
    assert b[0].x != b[0].y
    assert b[0].y != b[0].z
    # Values should be correct
    assert b[0].x == pytest.approx(1.0)
    assert b[0].y == pytest.approx(2.0)
    assert b[0].z == pytest.approx(3.0)

    # Cell2D distinct
    c = cnda.ContiguousND_Cell2D([2])
    c[0] = cnda.Cell2D(5.5, 6.5, 42)
    assert c[0].u != c[0].v
    assert c[0].flag == 42
    # Values should be correct
    assert c[0].u == pytest.approx(5.5)
    assert c[0].v == pytest.approx(6.5)

    # Particle position vs velocity
    pa = cnda.ContiguousND_Particle([1])
    pa[0] = cnda.Particle(1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0)
    assert pa[0].x != pa[0].vx
    assert pa[0].mass == pytest.approx(100.0)
    # Values should be correct
    assert pa[0].x == pytest.approx(1.0)
    assert pa[0].vx == pytest.approx(10.0)
    assert pa[0].mass == pytest.approx(100.0)


def test_contiguous_layout_behavior_binding():
    # check for contiguous layout/stride.
    v = cnda.ContiguousND_Vec2f([3])
    v[0] = cnda.Vec2f(1.0, 1.0)
    v[1] = cnda.Vec2f(2.0, 2.0)
    v[2] = cnda.Vec2f(3.0, 3.0)

    p0 = v.element_ptr(0)
    p1 = v.element_ptr(1)
    p2 = v.element_ptr(2)

    # element_ptr should return an integer address
    assert isinstance(p0, int)

    # Consecutive element pointers should be separated by sizeof(Vec2f)
    assert (p1 - p0) == ctypes.sizeof(Vec2f_ct)
    assert (p2 - p1) == ctypes.sizeof(Vec2f_ct)

    # test for ContiguousND_Cell3D
    inst3 = cnda.ContiguousND_Cell3D([5])
    assert hasattr(inst3, 'element_ptr') and callable(getattr(inst3, 'element_ptr'))

    # Check consecutive element addresses equal sizeof(Cell3D_ct)
    for i in range(4):
        p_i = inst3.element_ptr(i)
        p_i1 = inst3.element_ptr(i + 1)
        assert isinstance(p_i, int)
        assert isinstance(p_i1, int)
        assert (p_i1 - p_i) == ctypes.sizeof(Cell3D_ct)


def test_sizeof_validation_all_aos():
    # sizeof() validation for all AoS types using ctypes sizes.
    
    assert ctypes.sizeof(Vec2f_ct) == 2 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Vec3f_ct) == 3 * ctypes.sizeof(ctypes.c_float)
    assert ctypes.sizeof(Cell2D_ct) == 2 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
    assert ctypes.sizeof(Cell3D_ct) == 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
    assert ctypes.sizeof(Particle_ct) == 7 * ctypes.sizeof(ctypes.c_double)
    assert ctypes.sizeof(MaterialPoint_ct) == 3 * ctypes.sizeof(ctypes.c_float) + ctypes.sizeof(ctypes.c_int32)
