#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // suport std::vector
#include <cnda/contiguous_nd.hpp>  // include/cnda/
// AoS types (Vec2f, Vec3f, Cell2D, ...)
#include <cnda/aos_types.hpp>
#include <cstddef>
#include <cstdint>

//py is the abbrivation of pybind11
namespace py = pybind11;
using namespace cnda;
using namespace cnda::aos;

// Use template to do binding for different types.
// It helps to bind the C++ class ContiguousND<T> to a Python class.
template <typename T>
// Bind c++ function to python function
void bind_contiguous_nd(py::module_ &m, const std::string &class_name) {
    py::class_<ContiguousND<T>>(m, class_name.c_str())
        //Bind c++ constructor to python __init__
        .def(py::init<std::vector<std::size_t>>(), py::arg("shape")) // size_t -> python int
        .def("shape", &ContiguousND<T>::shape) // std::vector<size_t> -> python list
        .def("strides", &ContiguousND<T>::strides)
        .def("ndim", &ContiguousND<T>::ndim)
        .def("size", &ContiguousND<T>::size)
        .def("is_view", &ContiguousND<T>::is_view) // Whether this is a non-owning view
        // index() method - accepts a list/tuple and returns flat offset
        .def("index", [](const ContiguousND<T>& self, const std::vector<std::size_t>& idxs) -> std::size_t {
            const auto& sh = self.shape();
            const auto& str = self.strides();
            
            if (idxs.size() != sh.size()) {
                throw std::runtime_error("index: rank mismatch");
            }
            
            std::size_t off = 0;
            for (std::size_t a = 0; a < idxs.size(); ++a) {
                if (idxs[a] >= sh[a]) {
                    throw std::out_of_range("index: out of bounds");
                }
                off += idxs[a] * str[a];
            }
            return off;
        })
        // Because python does not support pointer, we convert the data to vector
        .def("data", [](ContiguousND<T> &self) {
            return std::vector<T>(self.data(), self.data() + self.size());
        })
        // To allow type int, list and tuple as indices (support arbitrary ndim)
        .def("__getitem__", [](ContiguousND<T>& self, py::object key) -> T& {
            if (py::isinstance<py::int_>(key)) {
                std::size_t i = key.cast<std::size_t>();
                return self(i);
            }
            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                std::vector<std::size_t> idx = key.cast<std::vector<std::size_t>>();
                const auto &sh = self.shape();
                const auto &str = self.strides();
                if (idx.size() != sh.size()) throw std::runtime_error("index: rank mismatch");
                std::size_t off = 0;
                for (std::size_t a = 0; a < idx.size(); ++a) {
                    if (idx[a] >= sh[a]) throw std::out_of_range("index: out of bounds");
                    off += idx[a] * str[a];
                }
                return self.data()[off];
            }

            throw std::runtime_error("Unsupported index type");
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](ContiguousND<T>& self, py::object key, T value) {
            if (py::isinstance<py::int_>(key)) {
                self(key.cast<std::size_t>()) = value;
                return;
            }

            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                std::vector<std::size_t> idx = key.cast<std::vector<std::size_t>>();
                const auto &sh = self.shape();
                const auto &str = self.strides();
                if (idx.size() != sh.size()) throw std::runtime_error("index: rank mismatch");
                std::size_t off = 0;
                for (std::size_t a = 0; a < idx.size(); ++a) {
                    if (idx[a] >= sh[a]) throw std::out_of_range("index: out of bounds");
                    off += idx[a] * str[a];
                }
                self.data()[off] = value;
                return;
            }
            throw std::runtime_error("Unsupported index type");
        })
        // Expose raw pointer helpers (as integer addresses) so Python tests
        // can perform byte-level offset checks against C++ expectations.
        .def("data_ptr", [](ContiguousND<T> &self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.data());
        })
        .def("element_ptr", [](ContiguousND<T> &self, py::object key) -> std::uintptr_t {
            // Support int index or tuple/list of indices similar to __getitem__ / at
            std::size_t off = 0;
            if (py::isinstance<py::int_>(key)) {
                off = key.cast<std::size_t>();
            } else if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                std::vector<std::size_t> idx = key.cast<std::vector<std::size_t>>();
                const auto &sh = self.shape();
                const auto &str = self.strides();
                if (idx.size() != sh.size()) throw std::runtime_error("index: rank mismatch");
                for (std::size_t a = 0; a < idx.size(); ++a) {
                    if (idx[a] >= sh[a]) throw std::out_of_range("index: out of bounds");
                    off += idx[a] * str[a];
                }
            } else {
                throw std::runtime_error("Unsupported index type for element_ptr");
            }
            return reinterpret_cast<std::uintptr_t>(&self.data()[off]);
        })
        // .at() method for bounds-checked access
        .def("at", [](ContiguousND<T>& self, py::object key) -> T& {
            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                // Try to cast as signed integers first to detect negative indices
                std::vector<py::ssize_t> signed_idx;
                try {
                    signed_idx = key.cast<std::vector<py::ssize_t>>();
                } catch (...) {
                    throw py::index_error("at(): invalid index type");
                }
                
                const auto &sh = self.shape();
                const auto &str = self.strides();
                
                // Rank check
                if (signed_idx.size() != sh.size()) {
                    throw py::index_error("at(): rank mismatch");
                }
                
                // Check for negative indices and convert to unsigned
                std::vector<std::size_t> idx(signed_idx.size());
                for (std::size_t a = 0; a < signed_idx.size(); ++a) {
                    if (signed_idx[a] < 0) {
                        throw py::index_error("at(): negative indices not supported");
                    }
                    idx[a] = static_cast<std::size_t>(signed_idx[a]);
                }
                
                // Bounds check for each dimension
                std::size_t off = 0;
                for (std::size_t a = 0; a < idx.size(); ++a) {
                    if (idx[a] >= sh[a]) {
                        throw py::index_error("at(): index out of bounds");
                    }
                    off += idx[a] * str[a];
                }
                return self.data()[off];
            }
            throw py::index_error("at(): requires tuple or list of indices");
        }, py::return_value_policy::reference_internal);
}

// Templated helpers
// These functions allocate a std::shared_ptr owner that holds the backing
// std::vector<T> and then construct a non-owning ContiguousND<T> that
// points into the owner's data. Using templates avoids duplicating code
// for each supported numeric type.
template <typename T>
ContiguousND<T> make_view_t(std::vector<std::size_t> shape, std::vector<T> buf) {
    // Create a shared owner for the buffer 
    auto owner = std::make_shared<std::vector<T>>(std::move(buf));
    // Construct a non-owning view pointing to the owner's data pointer.
    return ContiguousND<T>(std::move(shape), owner->data(), owner);
}

template <typename T>
py::tuple make_two_views_t(std::vector<std::size_t> shape1, std::vector<std::size_t> shape2, std::vector<T> buf) {
    // Create a single shared owner and construct two views that share it.
    auto owner = std::make_shared<std::vector<T>>(std::move(buf));
    ContiguousND<T> v1(shape1, owner->data(), owner);
    ContiguousND<T> v2(shape2, owner->data(), owner);
    // Use move semantics for the return to avoid copy issues
    return py::make_tuple(std::move(v1), std::move(v2));
}

// The dispatchers accept a Python sequence for the buffer and a required
// `dtype` string. We do not attempt silent type inference anymore; callers
// must explicitly pass one of: "int32", "int64", "float", or "double".
// This keeps behavior deterministic and avoids surprising defaults.
static py::object make_view_dispatch(std::vector<std::size_t> shape, py::object buf_obj, const std::string &dtype) {
    if (dtype.empty()) {
        throw std::runtime_error("make_view: dtype is required (e.g. dtype='int32'|'int64'|'float'|'double')");
    }

    // Use explicit dtype to cast the Python sequence into the corresponding
    // std::vector<T> and call the templated helper.
    if (dtype == "int32") return py::cast(make_view_t<int32_t>(std::move(shape), buf_obj.cast<std::vector<int32_t>>()));
    if (dtype == "int64") return py::cast(make_view_t<int64_t>(std::move(shape), buf_obj.cast<std::vector<int64_t>>()));
    if (dtype == "float") return py::cast(make_view_t<float>(std::move(shape), buf_obj.cast<std::vector<float>>()));
    if (dtype == "double") return py::cast(make_view_t<double>(std::move(shape), buf_obj.cast<std::vector<double>>()));
    throw std::runtime_error("Unsupported dtype string");
}

static py::object make_two_views_dispatch(std::vector<std::size_t> shape1, std::vector<std::size_t> shape2, py::object buf_obj, const std::string &dtype) {
    if (dtype.empty()) {
        throw std::runtime_error("make_two_views: dtype is required (e.g. dtype='int32'|'int64'|'float'|'double')");
    }

    if (dtype == "int32") return make_two_views_t<int32_t>(std::move(shape1), std::move(shape2), buf_obj.cast<std::vector<int32_t>>());
    if (dtype == "int64") return make_two_views_t<int64_t>(std::move(shape1), std::move(shape2), buf_obj.cast<std::vector<int64_t>>());
    if (dtype == "float") return make_two_views_t<float>(std::move(shape1), std::move(shape2), buf_obj.cast<std::vector<float>>());
    if (dtype == "double") return make_two_views_t<double>(std::move(shape1), std::move(shape2), buf_obj.cast<std::vector<double>>());
    throw std::runtime_error("Unsupported dtype string");
}


PYBIND11_MODULE(cnda, m) {
    m.doc() = "Python bindings for ContiguousND C++ template class";
    bind_contiguous_nd<int32_t>(m, "ContiguousND_int32");
    bind_contiguous_nd<int64_t>(m, "ContiguousND_int64");
    bind_contiguous_nd<float>(m, "ContiguousND_float");
    bind_contiguous_nd<double>(m, "ContiguousND_double");
    // Bind AoS structs from cnda::aos and expose ContiguousND specializations
    py::class_<aos::Vec2f>(m, "Vec2f")
        .def(py::init([](float x, float y){ return aos::Vec2f{x,y}; }), py::arg("x")=0.0f, py::arg("y")=0.0f)
        .def_readwrite("x", &aos::Vec2f::x)
        .def_readwrite("y", &aos::Vec2f::y);

    py::class_<aos::Vec3f>(m, "Vec3f")
        .def(py::init([](float x, float y, float z){ return aos::Vec3f{x,y,z}; }), py::arg("x")=0.0f, py::arg("y")=0.0f, py::arg("z")=0.0f)
        .def_readwrite("x", &aos::Vec3f::x)
        .def_readwrite("y", &aos::Vec3f::y)
        .def_readwrite("z", &aos::Vec3f::z);

    py::class_<aos::Cell2D>(m, "Cell2D")
        .def(py::init([](float u, float v, int flag){ return aos::Cell2D{u,v,static_cast<std::int32_t>(flag)}; }), py::arg("u")=0.0f, py::arg("v")=0.0f, py::arg("flag")=0)
        .def_readwrite("u", &aos::Cell2D::u)
        .def_readwrite("v", &aos::Cell2D::v)
        .def_readwrite("flag", &aos::Cell2D::flag);

    py::class_<aos::Cell3D>(m, "Cell3D")
        .def(py::init([](float u, float v, float w, int flag){ return aos::Cell3D{u,v,w,static_cast<std::int32_t>(flag)}; }), py::arg("u")=0.0f, py::arg("v")=0.0f, py::arg("w")=0.0f, py::arg("flag")=0)
        .def_readwrite("u", &aos::Cell3D::u)
        .def_readwrite("v", &aos::Cell3D::v)
        .def_readwrite("w", &aos::Cell3D::w)
        .def_readwrite("flag", &aos::Cell3D::flag);

    py::class_<aos::Particle>(m, "Particle")
        .def(py::init([](double x,double y,double z,double vx,double vy,double vz,double mass){ return aos::Particle{x,y,z,vx,vy,vz,mass}; }),
             py::arg("x")=0.0, py::arg("y")=0.0, py::arg("z")=0.0, py::arg("vx")=0.0, py::arg("vy")=0.0, py::arg("vz")=0.0, py::arg("mass")=1.0)
        .def_readwrite("x", &aos::Particle::x)
        .def_readwrite("y", &aos::Particle::y)
        .def_readwrite("z", &aos::Particle::z)
        .def_readwrite("vx", &aos::Particle::vx)
        .def_readwrite("vy", &aos::Particle::vy)
        .def_readwrite("vz", &aos::Particle::vz)
        .def_readwrite("mass", &aos::Particle::mass);

    py::class_<aos::MaterialPoint>(m, "MaterialPoint")
        .def(py::init([](float density,float temperature,float pressure,int id){ return aos::MaterialPoint{density,temperature,pressure,static_cast<std::int32_t>(id)}; }), py::arg("density")=0.0f, py::arg("temperature")=0.0f, py::arg("pressure")=0.0f, py::arg("id")=0)
        .def_readwrite("density", &aos::MaterialPoint::density)
        .def_readwrite("temperature", &aos::MaterialPoint::temperature)
        .def_readwrite("pressure", &aos::MaterialPoint::pressure)
        .def_readwrite("id", &aos::MaterialPoint::id);

    // Expose ContiguousND specializations for AoS structs
    bind_contiguous_nd<aos::Vec2f>(m, "ContiguousND_Vec2f");
    bind_contiguous_nd<aos::Vec3f>(m, "ContiguousND_Vec3f");
    bind_contiguous_nd<aos::Cell2D>(m, "ContiguousND_Cell2D");
    bind_contiguous_nd<aos::Cell3D>(m, "ContiguousND_Cell3D");
    bind_contiguous_nd<aos::Particle>(m, "ContiguousND_Particle");
    bind_contiguous_nd<aos::MaterialPoint>(m, "ContiguousND_MaterialPoint");
    // Expose sizeof helper for AoS types to Python tests
    m.def("sizeof_aos", [](const std::string &name) -> std::size_t {
        if (name == "Vec2f") return sizeof(aos::Vec2f);
        if (name == "Vec3f") return sizeof(aos::Vec3f);
        if (name == "Cell2D") return sizeof(aos::Cell2D);
        if (name == "Cell3D") return sizeof(aos::Cell3D);
        if (name == "Particle") return sizeof(aos::Particle);
        if (name == "MaterialPoint") return sizeof(aos::MaterialPoint);
        throw std::runtime_error("sizeof_aos: unknown AoS type '" + name + "'");
    }, py::arg("name"));
    // Accept (shape, buf, dtype) where dtype is required and must be one of
    // "int32", "int64", "float", or "double".
    m.def("make_view", &make_view_dispatch, py::arg("shape"), py::arg("buf"), py::arg("dtype"));
    m.def("make_two_views", &make_two_views_dispatch, py::arg("shape1"), py::arg("shape2"), py::arg("buf"), py::arg("dtype"));
}
