# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS
import logging


@FILE_SYSTEMS.register_class()
class LocalFs(BaseFs):
    def __init__(self):
        super(LocalFs, self).__init__()

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
