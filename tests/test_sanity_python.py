import pytest
import cnda

def test_import_module_has_types():
    """Module import sanity check (moved from test_accessors_python)."""
    assert hasattr(cnda, "ContiguousND_int32")
    assert hasattr(cnda, "ContiguousND_int64")
    assert hasattr(cnda, "ContiguousND_float")
    assert hasattr(cnda, "ContiguousND_double")
