// Minimal pybind11 bindings to allow building the cnda extension.
// If you already have a real bindings file, replace this with the project one.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // suport std::vector
#include <cnda/contiguous_nd.hpp>  // include/cnda/
import sys
print(sys.path) 
print(cnda.__file__) 

//py is the abbrivation of pybind11
namespace py = pybind11;
using namespace cnda;
template <typename T>
// Bind c++ function to python function
void bind_contiguous_nd(py::module_ &m, const std::string &class_name) {
    //empty binding to ensure compile succeeds and make test fail
    py::class_<ContiguousND<T>>(m, class_name.c_str())
    //Bind c++ constructor to python __init__
    .def(py::init<std::vector<std::size_t>>(), py::arg("shape")); // size_t -> python int
}
PYBIND11_MODULE(cnda, m) {
    m.doc() = "cnda minimal pybind11 bindings";
    // Keep minimal to ensure compile succeeds in CI; extend with real API as needed.
    bind_contiguous_nd<int>(m, "ContiguousND_int");
}
