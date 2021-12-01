# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os
import platform
from distutils.core import setup, Extension

import pybind11

if platform.system() == "Darwin":
    cpp_args = ['-std=c++11', '-stdlib=libc++', '-O3']
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"
else:
    cpp_args = ['-std=c++11', '-O3']

ext_modules = [
    Extension(
        'CalcLargestConnectedComponentsIds',
        ['CalcLargestConnectedComponentsIds.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args
    ),
]

setup(
    name='ICCV2021-NGC-LLC',
    version='0.0.1',
    ext_modules=ext_modules
)
