from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """延遲導入 pybind11，直到安裝時"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'cnda',                                   # module name(Python import name)
        ['python/cnda/bindings.cpp'],             # C++ 原始檔
        include_dirs=[
            get_pybind_include(),                 # pybind11 的 include 目錄
            'python/cnda',                        # 你的 .hpp 所在位置
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
    packages=setuptools.find_packages(where='python'),
    package_dir={'': 'python'},
    python_requires='>=3.10',
)
