# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import importlib
import os.path as osp
import sys


def _import_ext_module_dir(module_dir):
    if not osp.isdir(module_dir):
        raise f"Path {module_dir} is not a directory."
    module_abs_path = osp.abspath(osp.expanduser(module_dir))
    module_dir = osp.dirname(module_abs_path)
    module_name = osp.basename(module_abs_path)
    sys.path.insert(0, module_dir)
    importlib.import_module(module_name)
    # Keep module_dir in sys.path, multiprocessing context will utilize this variable to avoid 'import error'
    # sys.path.pop(0)


def _import_ext_module_py(py_path):
    module_abs_path = osp.abspath(osp.expanduser(py_path))
    module_dir = osp.dirname(module_abs_path)
    module_name = osp.basename(module_abs_path).replace(".py", "")
    sys.path.insert(0, module_dir)
    importlib.import_module(module_name)
    # Keep module_dir in sys.path, multiprocessing context will utilize this variable to avoid 'import error'
    # sys.path.pop(0)


def import_ext_module(module_dir):
    """ Import extension module from
    1. directory;
    2. python file;
    3. a list contains directory or python file.
    So that we do not need to write 'import xxx' statements.

    Args:
        module_dir (str): Path to module directory or module file. If contains more than one path, split by ';'.
    """

    module_dirs = module_dir.split(";")
    for dir_path in module_dirs:
        if osp.exists(dir_path):
            if osp.isdir(dir_path):
                _import_ext_module_dir(dir_path)
            else:
                _import_ext_module_py(dir_path)
