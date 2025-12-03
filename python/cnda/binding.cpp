#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // suport std::vector
#include <cnda/contiguous_nd.hpp>  // include/cnda/

//py is the abbrivation of pybind11
namespace py = pybind11;
using namespace cnda;

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
        .def("index", (std::size_t (ContiguousND<T>::*)(const std::vector<std::size_t>&) const) &ContiguousND<T>::index)
        // Because python does not support pointer, we convert the data to vector
        .def("data", [](ContiguousND<T> &self) {
            return std::vector<T>(self.data(), self.data() + self.size());
        })
        // To allow type int, list and tuple as indices (support arbitrary ndim)
        .def("__getitem__", [](ContiguousND<T>& self, py::object key) -> T {
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
        })
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
        // .at() method for bounds-checked access
        .def("at", [](ContiguousND<T>& self, py::object key) -> T {
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
        });
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
    // Accept (shape, buf, dtype) where dtype is required and must be one of
    // "int32", "int64", "float", or "double".
    m.def("make_view", &make_view_dispatch, py::arg("shape"), py::arg("buf"), py::arg("dtype"));
    m.def("make_two_views", &make_two_views_dispatch, py::arg("shape1"), py::arg("shape2"), py::arg("buf"), py::arg("dtype"));
}
