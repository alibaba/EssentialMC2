# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import datetime
import os
import os.path as osp
import random
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Optional

from .utils import remove_temp_path


class BaseFs(object, metaclass=ABCMeta):
    def __init__(self):
        self._target_local_mapper = {}
        self._temp_files = set()

    # Functions without io
    @abstractmethod
    def get_prefix(self) -> str:
        """ Get supported path prefix to determine which handler to use.

        Returns:
            A prefix.
        """
        pass

    @abstractmethod
    def support_write(self) -> bool:
        """ Return flag if this file system supports write operation.

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def support_link(self) -> bool:
        """ Return if this file system supports create a soft link.

        Returns:
            Bool.
        """
        pass

    def add_target_local_map(self, target_dir, local_dir):
        """ Map target directory to local file system directory

        Args:
            target_dir (str): Target directory.
            local_dir (str): Directory in local file system.
        """
        self._target_local_mapper[target_dir] = local_dir

    def map_to_local(self, target_path) -> (str, bool):
        """ Map target path to local file path. (NO IO HERE).

        Args:
            target_path (str): Target file path.

        Returns:
            A local path and a flag indicates if the local path is a temporary file.
        """
        for target_dir, local_dir in self._target_local_mapper.items():
            if target_path.startswith(target_dir):
                local_path = osp.join(local_dir, osp.relpath(target_path, target_dir))
                os.makedirs(osp.dirname(local_path), exist_ok=True)
                return local_path, False
        else:
            return self._make_temporary_file(target_path), True

    def convert_to_local_path(self, target_path) -> str:
        """ Deprecated. Use map_to_local() function instead.
        """
        warnings.warn("Function convert_to_local_path is deprecated, use map_to_local() function instead.")
        local_path, _ = self.map_to_local(target_path)
        return local_path

    def basename(self, target_path) -> str:
        """ Get file name from target_path

        Args:
            target_path (str): Target file path.

        Returns:
            A file name.
        """
        return osp.basename(target_path)

    # Functions with heavy io
    @abstractmethod
    def get_object_to_local_file(self, target_path, local_path=None) -> Optional[str]:
        """ Transfer file object to local file.
        If local_path is not None,
            if path can be searched in local_mapper, download it as a persistent file
            else, download it as a temporary file
        else
            download it as a persistent file

        Args:
            target_path (str): path of object in different file systems
            local_path (Optional[str]): If not None, will write path to local_path.

        Returns:
            Local file path of the object, none means a failure happened.
        """
        pass

    @abstractmethod
    def put_object_from_local_file(self, local_path, target_path) -> bool:
        """ Put local file to target file system path.

        Args:
            local_path (str): local file path of the object
            target_path (str): target file path of the object

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def make_link(self, target_link_path, target_path) -> bool:
        """ Make soft link to target_path.

        Args:
            target_link_path (str):
            target_path (str)

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def make_dir(self, target_dir) -> bool:
        """ Make a directory.
        If target_dir is already exists, return True.

        Args:
            target_dir (str):

        Returns:
            True if target_dir exists or created.
        """
        pass

    @abstractmethod
    def remove(self, target_path) -> bool:
        """ Remove target file.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def get_logging_handler(self, target_logging_path):
        """ Get logging handler to target logging path.

        Args:
            target_logging_path:

        Returns:
            A handler which has a type of subclass of logging.Handler.
        """
        pass

    @abstractmethod
    def put_dir_from_local_dir(self, local_dir, target_dir) -> bool:
        """ Upload all contents in local_dir to target_dir, keep the file tree.

        Args:
            local_dir (str):
            target_dir (str):

        Returns:
            Bool.
        """
        pass

    def _make_temporary_file(self, target_path):
        """ Make a temporary file for target_path, which should have the same suffix.

        Args:
            target_path (str):

        Returns:
            A path (str).
        """
        file_name = self.basename(target_path)
        _, suffix = osp.splitext(file_name)
        rand_name = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()) + '_' + ''.join(
            [str(random.randint(1, 10)) for _ in range(5)])
        if suffix:
            rand_name += f'{suffix}'
        tmp_file = osp.join(tempfile.gettempdir(), rand_name)
        return tmp_file

    # Functions only for status, light io
    @abstractmethod
    def exists(self, target_path) -> bool:
        """ Check if target_path exists.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def isfile(self, target_path) -> bool:
        """ Check if target_path is a file.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        pass

    @abstractmethod
    def isdir(self, target_path) -> bool:
        """ Check if target_path is a directory.

        Args:
            target_path (str):

        Returns:
            Bool.
        """

    def add_temp_file(self, tmp_file):
        self._temp_files.add(tmp_file)

    def clear(self):
        """Delete all temp files
        """
        for temp_local_file in self._temp_files:
            remove_temp_path(temp_local_file)

    # Functions for context
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for temp_local_file in self._temp_files:
            remove_temp_path(temp_local_file)

    def __del__(self):
        self.clear()

    def copy(self):
        obj = copy(self)
        obj._temp_files = set()  # A new obj to avoid confusing in multi-thread context.
        return obj
