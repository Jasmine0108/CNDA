from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import platform

class get_pybind_include(object):
    """Delay import of pybind11 until it is actually needed"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Platform-specific compile arguments
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/EHsc']
else:
    extra_compile_args = ['-std=c++17']
    if sys.platform == 'darwin':
        extra_compile_args.append('-stdlib=libc++')
        extra_link_args.append('-stdlib=libc++')

ext_modules = [
    Extension(
        'cnda',                                   # module name (Python import name)
        ['python/cnda/binding.cpp'],      
        include_dirs=[
            get_pybind_include(),                 # get include table of contents of pybind11
            'include',                        # The location of .hpp files
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='cnda',
    version='0.1.0',
    description='Python bindings for C++ code using pybind11',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    # Don't include python/cnda as a package to avoid conflicts with the compiled extension
    packages=[],
    python_requires='>=3.9',
)
