# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import logging
import os
import os.path as osp
import shutil
from typing import Optional

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


@FILE_SYSTEMS.register_class()
class LocalFs(BaseFs):
    def __init__(self):
        super(LocalFs, self).__init__()

    def get_prefix(self) -> str:
        return "file://"

    def support_write(self) -> bool:
        return True

    def support_link(self) -> bool:
        return True

    def map_to_local(self, target_path) -> (str, bool):
        return target_path, False

    def get_object_to_local_file(self, target_path, local_path=None) -> Optional[str]:
        target_path = osp.abspath(target_path)

        if local_path is not None:
            local_path = osp.abspath(local_path)
            if local_path != target_path:
                # copy target_path to local_path
                os.makedirs(osp.dirname(local_path), exist_ok=True)
                try:
                    shutil.copy(target_path, local_path)
                except:
                    return None

            return local_path
        return target_path

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        target_path = osp.abspath(target_path)
        local_path = osp.abspath(local_path)
        if local_path != target_path:
            try:
                shutil.copy(local_path, target_path)
            except:
                return False
        return True

    def make_dir(self, target_dir) -> bool:
        if osp.exists(target_dir):
            if osp.isfile(target_dir):
                print(f"{target_dir} already exists as a file!")
                return False
            return True
        try:
            os.makedirs(target_dir)
        except Exception as e:
            print(e)
            return False
        return True

    def make_link(self, target_link_path, target_path) -> bool:
        try:
            if osp.lexists(target_link_path):
                os.remove(target_link_path)
            os.symlink(target_path, target_link_path)
            return True
        except:
            return False

    def remove(self, target_path) -> bool:
        if osp.exists(target_path):
            try:
                os.remove(target_path)
            except:
                return False
        return True

    def get_logging_handler(self, target_logging_path):
        return logging.FileHandler(target_logging_path)

    def put_dir_from_local_dir(self, local_dir, target_dir) -> bool:
        local_dir = osp.abspath(local_dir)
        target_dir = osp.abspath(target_dir)
        if local_dir == target_dir:
            return True
        # cp -f local_dir/* target_dir/*
        if not osp.exists(target_dir):
            status = os.system(f"mkdir -p {target_dir}")
            if status != 0:
                return False
        try:
            shutil.copytree(local_dir, target_dir, symlinks=True)
        except:
            return False
        return True

    def exists(self, target_path) -> bool:
        return osp.exists(target_path)

    def isfile(self, target_path) -> bool:
        return osp.isfile(target_path)

    def isdir(self, target_path) -> bool:
        return osp.isdir(target_path)
