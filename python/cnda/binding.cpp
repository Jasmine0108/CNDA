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
        // Because python does not support pointer, we convert the data to vector
        .def("data", [](ContiguousND<T> &self) {
            return std::vector<T>(self.data(), self.data() + self.size());
        })
        .def("__getitem__", [](ContiguousND<T> &self, std::vector<std::size_t> idx) {
            if (idx.size() == 1) return self(idx[0]);
            else if (idx.size() == 2) return self(idx[0], idx[1]);
            else if (idx.size() == 3) return self(idx[0], idx[1], idx[2]);
            else throw std::runtime_error("Unsupported ndim");
        })
        .def("__setitem__", [](ContiguousND<T> &self, std::vector<std::size_t> idx, T value) {
            if (idx.size() == 1) self(idx[0]) = value;
            else if (idx.size() == 2) self(idx[0], idx[1]) = value;
            else if (idx.size() == 3) self(idx[0], idx[1], idx[2]) = value;
            else throw std::runtime_error("Unsupported ndim");
        });
}

PYBIND11_MODULE(cnda, m) {
    m.doc() = "Python bindings for ContiguousND C++ template class";
    bind_contiguous_nd<int32_t>(m, "ContiguousND_int32");
    bind_contiguous_nd<int64_t>(m, "ContiguousND_int64");
    bind_contiguous_nd<float>(m, "ContiguousND_float");
    bind_contiguous_nd<double>(m, "ContiguousND_double");
}