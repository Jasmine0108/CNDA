from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class get_pybind_include(object):
    """Delay import of pybind11 until it is actually needed"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'cnda',                                   # module name (Python import name)
        ['python/cnda/binding.cpp'],      
        include_dirs=[
            get_pybind_include(),                 # get include table of contents of pybind11
            'include',                        # The location of .hpp files
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='cnda',
    version='0.1.0',
    description='Python bindings for C++ code using pybind11',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    packages=find_packages(),
    python_requires='>=3.10',
)
