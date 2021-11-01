# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp
import tempfile
from abc import ABCMeta, abstractmethod


class BaseFs(object, metaclass=ABCMeta):
    def __init__(self):
        self.local_mapper = {}
        self.to_removes = set()

    @abstractmethod
    def get_object_to_local_file(self, path) -> str:
        """ Transfer different file system object to local file.

        Args:
            path (str): path of object in different file systems

        Returns:
            Local file path of the object.
        """
        pass

    @abstractmethod
    def get_object_to_memory(self, path) -> bytes:
        """ Get target file object to memory in bytes.

        Args:
            path (str): path of object in different file systems

        Returns:
            Bytes.
        """
        pass

    @abstractmethod
    def remove_local_file(self, local_path):
        """ Delete local file is used, avoid too much space usage.

        Args:
            local_path (str): result returned by `get_local_file` function
        """
        pass

    @abstractmethod
    def put_object_from_local_file(self, local_path, target_path):
        """ Put local file to target file system path.

        Args:
            local_path (str): local file path of the object
            target_path (str): target file path of the object
        """
        pass

    @abstractmethod
    def get_prefix(self):
        """ Get supported path prefix to determine which handler to use

        Returns:
            A prefix.
        """
        pass

    @abstractmethod
    def support_write(self):
        """ Return flag if this file system supports write operation

        Returns:
            Bool.
        """
        pass

    def add_target_local_map(self, target_dir, local_dir):
        """ Map target directory to local file system directory

        Args:
            target_dir (str): Target directory in non-local file system.
            local_dir (str): Directory in local file system.
        """
        self.local_mapper[target_dir] = local_dir

    def convert_to_local_path(self, target_path):
        """ According to self.local_mapper, map target_path to local_path

        Args:
            target_path (str): Target directory in non-local file system.

        Returns:
            File path in local file system.
        """
        for target_dir, local_dir in self.local_mapper.items():
            if target_path.startswith(target_dir):
                return osp.join(local_dir, osp.relpath(target_path, target_dir))
        else:
            return osp.join(tempfile.gettempdir(), osp.basename(target_path))

    @abstractmethod
    def get_logging_handler(self, logging_path):
        pass

    def _clear(self):
        for remove in list(self.to_removes):
            self.remove_local_file(remove)
        self.to_removes.clear()

    def __enter__(self):
        self._clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._clear()
