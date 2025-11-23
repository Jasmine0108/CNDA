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
        });
}

PYBIND11_MODULE(cnda, m) {
    m.doc() = "Python bindings for ContiguousND C++ template class";
    bind_contiguous_nd<int32_t>(m, "ContiguousND_int32");
    bind_contiguous_nd<int64_t>(m, "ContiguousND_int64");
    bind_contiguous_nd<float>(m, "ContiguousND_float");
    bind_contiguous_nd<double>(m, "ContiguousND_double");
}