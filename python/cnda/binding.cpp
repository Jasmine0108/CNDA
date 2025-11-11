// Minimal pybind11 bindings to allow building the cnda extension.
// If you already have a real bindings file, replace this with the project one.
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cnda, m) {
    m.doc() = "cnda minimal pybind11 bindings";
    // Keep minimal to ensure compile succeeds in CI; extend with real API as needed.
}
