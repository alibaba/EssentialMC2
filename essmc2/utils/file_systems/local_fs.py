# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import logging
import os

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


@FILE_SYSTEMS.register_class()
class LocalFs(BaseFs):
    def __init__(self):
        super(LocalFs, self).__init__()

    def convert_to_local_path(self, target_path):
        return target_path

    def get_object_to_local_file(self, path) -> str:
        return path

    def get_object_to_memory(self, path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def remove_local_file(self, local_path):
        # Do not delete local file in local file system
        return

    def put_object_from_local_file(self, local_path, target_path):
        # Do Not.
        return

    def get_prefix(self):
        return "file://"

    def support_write(self):
        return True

    def get_logging_handler(self, logging_path):
        return logging.FileHandler(logging_path)

    def make_link(self, link_path, target_path):
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(target_path, link_path)

    def put_dir_from_local_dir(self, local_dir, target_dir):
        # Do not.
        return

    def exists(self, target_path):
        return os.path.exists(target_path)

    def remove(self, target_path):
        if os.path.exists(target_path):
            os.remove(target_path)




