# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .local_fs import LocalFs
from .registry import FILE_SYSTEMS

from ..typing import check_dict_of_str_dict


class FS(object):
    """ File system, module single instance. Should only be inited once.
    """
    _inited = False
    _inited_clients = {}
    _default_client = LocalFs()

    @staticmethod
    def init_fs_client(cfg=None):
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
        if FS._inited:
            return
        fs_cfg_list = cfg or []
        if isinstance(fs_cfg_list, dict):
            if check_dict_of_str_dict(fs_cfg_list, contains_type=True):
                fs_cfg_list = list(fs_cfg_list.values())
            else:
                fs_cfg_list = [fs_cfg_list]

        fs_cfg_list = [t for t in fs_cfg_list if t.get("type") != "LocalFs"]

        for fs_cfg in fs_cfg_list:
            fs_client = FILE_SYSTEMS.build(fs_cfg)
            FS._inited_clients[fs_client.get_prefix()] = fs_client

        FS._inited = True

    @staticmethod
    def get_fs_client(path: str):
        """ Get the client by input path.
        Every file system has its own identifier, default will use local file system to have a try.
        """
        for prefix, client in FS._inited_clients.items():
            if path.startswith(prefix):
                return client

        return FS._default_client
