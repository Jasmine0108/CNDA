"""cnda python package placeholder.

This package contains the compiled extension module `cnda` (built from
`bindings.cpp`). The file below exists so `find_packages()` can discover
the package during packaging; it's safe to keep it minimal.
"""

# Expose compiled extension if importable; otherwise allow normal Python imports.
try:
    # compiled extension module name is `cnda` (top-level)
    from . import cnda as _cnda  # type: ignore
except Exception:
    # extension might not be built in some environments (editable install, docs, etc.)
    _cnda = None

__all__ = []
