# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import warnings
from contextlib import contextmanager
import os.path as osp

from .base_fs import BaseFs
from .local_fs import LocalFs
from .registry import FILE_SYSTEMS
from .utils import check_if_local_path


def _check_dict_of_str_dict(input_dict, contains_type=False):
    if not isinstance(input_dict, dict):
        return False
    for key, value in input_dict.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False
        if contains_type and 'type' not in value:
            return False

    return True


class ReadException(Exception):
    pass


class WriteException(Exception):
    pass


class FileSystem(object):
    def __init__(self):
        self._prefix_to_clients = {}
        self._default_client = LocalFs()

    def init_fs_client(self, cfg=None):
        """ Initialize file system backend
        Supported backend:
            1. Local file system, e.g. /home/admin/work_dir, work_dir_bk/imagenet_pretrain
            2. Aliyun Oss, e.g. oss://bucket_name/work_dir
            3. Http, only support to read content, e.g. https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png
            4. other fs backend...

        Args:
            cfg (list, dict, optional):
                list: list of file system configs to be initialized
                dict: a dict contains file system configs as values or a file system config dict
                optional: Will only use default LocalFs
        """
        fs_cfg_list = cfg or []
        if isinstance(fs_cfg_list, dict):
            if _check_dict_of_str_dict(fs_cfg_list, contains_type=True):
                fs_cfg_list = list(fs_cfg_list.values())
            else:
                fs_cfg_list = [fs_cfg_list]

        fs_cfg_list = [t for t in fs_cfg_list if t.get("type") != "LocalFs"]

        for fs_cfg in fs_cfg_list:
            fs_client = FILE_SYSTEMS.build(fs_cfg)
            _prefix = fs_client.get_prefix()
            if _prefix in self._prefix_to_clients:
                warnings.warn(f"File client {_prefix} has already been set, will be replaced by newer config.")
            self._prefix_to_clients[_prefix] = fs_client

    def get_fs_client(self, target_path, safe=True) -> BaseFs:
        """ Get the client by input path.
        Every file system has its own identifier, default will use local file system to have a try.
        If copy needed, only do shallow copy.

        Args:
            target_path (str):
            safe (bool): In safe mode, get the copy of the client.
        """
        obj = None
        for prefix in sorted(list(self._prefix_to_clients.keys()), key=lambda a: -len(a)):
            if target_path.startswith(prefix):
                obj = self._prefix_to_clients[prefix]
                break
        if obj is not None:
            if safe:
                return obj.copy()
            else:
                return obj

        if not check_if_local_path(target_path):
            warnings.warn(f"{target_path} is not a local path, use LocalFs may cause an error.")

        if safe:
            return self._default_client.copy()
        else:
            return self._default_client

    def add_target_local_map(self, target_dir, local_dir):
        """ Map target directory to local file system directory

        Args:
            target_dir (str): Target directory.
            local_dir (str): Directory in local file system.
        """
        with self.get_fs_client(target_dir, safe=False) as client:
            client.add_target_local_map(target_dir, local_dir)

    def make_dir(self, target_dir):
        """ Make a directory.
                If target_dir is already exists, return True.

                Args:
                    target_dir (str):

                Returns:
                    True if target_dir exists or created.
                """
        with self.get_fs_client(target_dir) as client:
            return client.make_dir(target_dir)

    def exists(self, target_path):
        """ Check if target_path exists.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_path) as client:
            return client.exists(target_path)

    def map_to_local(self, target_path):
        """ Map target path to local file path. (NO IO HERE).

        Args:
            target_path (str): Target file path.

        Returns:
            A local path and a flag indicates if the local path is a temporary file.
        """
        with self.get_fs_client(target_path) as client:
            local_path, is_tmp = client.map_to_local(target_path)
            return local_path, is_tmp

    def put_dir_from_local_dir(self, local_dir, target_dir):
        """ Upload all contents in local_dir to target_dir, keep the file tree.

        Args:
            local_dir (str):
            target_dir (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_dir) as client:
            return client.put_dir_from_local_dir(local_dir, target_dir)

    def is_local_client(self, target_path) -> bool:
        """ Check if the client support read or write to target_path is a LocalFs.

        Args:
            target_path (str):

        Returns:
            Bool.
        """
        with self.get_fs_client(target_path) as client:
            return type(client) is LocalFs

    @contextmanager
    def get_from(self, target_path, local_path=None):
        with self.get_fs_client(target_path) as client:
            local_path = client.get_object_to_local_file(target_path, local_path=local_path)
            if local_path is None:
                raise ReadException(f"Failed to fetch {target_path} to {local_path}")
            yield local_path

    @contextmanager
    def put_to(self, target_path):
        with self.get_fs_client(target_path) as client:
            local_path, is_tmp = client.map_to_local(target_path)
            if is_tmp:
                client.add_temp_file(local_path)
            yield local_path
            if not osp.exists(local_path):
                raise WriteException(f"{local_path} is not exists.")
            status = client.put_object_from_local_file(local_path, target_path)
            if not status:
                raise WriteException(f"Failed to upload from {local_path} to {target_path}")

    @contextmanager
    def open(self, target_path, mode='r', encoding=None):
        """ Mock builtin open function.

        Args:
            target_path (str): Local or remote path.
            mode (str): Support 'r', 'w', 'rb', 'wb' only.
            encoding (Optional[str]):
        """
        assert mode in ('r', 'w', 'rb', 'wb')
        read_only = 'r' in mode

        with self.get_fs_client(target_path) as client:
            if read_only:
                local_path = client.get_object_to_local_file(target_path)
                if local_path is None:
                    raise ReadException(f"Failed to fetch {target_path}")
                try:
                    handler = open(local_path, mode=mode, encoding=encoding)
                    yield handler
                finally:
                    handler.close()
            else:
                local_path, is_tmp = client.map_to_local(target_path)
                if is_tmp:
                    client.add_temp_file(local_path)
                try:
                    handler = open(local_path, mode=mode, encoding=encoding)
                    yield handler
                finally:
                    handler.close()
                status = client.put_object_from_local_file(local_path, target_path)
                if not status:
                    raise WriteException(f"Failed to write to {target_path}")

    def __repr__(self) -> str:
        s = "Support prefix list:\n"
        for prefix in sorted(list(self._prefix_to_clients.keys()), key=lambda a: -len(a)):
            s += f"\t{prefix} -> {self._prefix_to_clients[prefix]}\n"
        return s


# global instance, easy to use
FS = FileSystem()
