#!/usr/bin/python3
# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))

from essmc2.utils.ext_module import import_ext_module

from essmc2.utils.registry import Registry
from essmc2.utils.config import Config


def display_all_modules():
    for registry in Registry.REGISTRY_LIST:
        print(registry)
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext_module", default=None, type=str,
                        help="Extension module path to be imported for custom modules, default is empty")

    parser.add_argument("-m", "--module", default=None, type=str,
                        help=f"Display a specify registry.")

    parser.add_argument("-t", "--type", default=None, type=str,
                        help="Display a specify type parameters.")

    return parser.parse_args()


def main():
    args = parse_args()

    ext_module = args.ext_module
    if ext_module:
        import_ext_module(ext_module)

    if args.module is None or args.type is None:
        display_all_modules()
        print("You can specify -m and -t to list detailed parameters.")
    else:
        module = args.module
        type_name = args.type
        specify_module = [t for t in Registry.REGISTRY_LIST if t.name == module]
        if len(specify_module) == 0:
            print(f"{module} not found.")
            display_all_modules()
            exit(0)

        specify_module = specify_module[0]

        if not specify_module.contains(type_name):
            print(f"{type_name} not found in {module}")
            print(specify_module)
            exit(0)

        type_args = specify_module.fetch_parameters(type_name)

        cfg = Config(dict(
            __key__=dict(
                type=type_name,
                **type_args
            )
        ))

        print(cfg)


if __name__ == "__main__":
    main()
