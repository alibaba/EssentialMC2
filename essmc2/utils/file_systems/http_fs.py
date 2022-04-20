# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os
import os.path as osp
import urllib.parse as parse
import urllib.request
from typing import Optional

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


@FILE_SYSTEMS.register_class()
class HttpFs(BaseFs):
    def __init__(self, retry_times=10):
        super(HttpFs, self).__init__()
        self._retry_times = retry_times

    def get_prefix(self) -> str:
        return "http"

    def support_write(self) -> bool:
        return False

    def support_link(self) -> bool:
        return False

    def basename(self, target_path) -> str:
        url = parse.unquote(target_path)
        url = url.split("?")[0]
        return osp.basename(url)

    def get_object_to_local_file(self, target_path, local_path=None) -> Optional[str]:
        if local_path is None:
            local_path, is_tmp = self.map_to_local(target_path)
        else:
            is_tmp = False

        os.makedirs(osp.dirname(local_path), exist_ok=True)

        retry = 0
        while retry < self._retry_times:
            try:
                target_url = urllib.parse.quote(target_path, safe=":/?#[]@!$&'()*+,;=%")
                urllib.request.urlretrieve(target_url, local_path)
                if osp.exists(local_path):
                    break
            except:
                retry += 1

        if retry >= self._retry_times:
            return None

        if is_tmp:
            self.add_temp_file(local_path)
        return local_path

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        raise NotImplemented

    def make_link(self, target_link_path, target_path) -> bool:
        raise NotImplemented

    def make_dir(self, target_dir) -> bool:
        raise NotImplemented

    def remove(self, target_path) -> bool:
        raise NotImplemented

    def get_logging_handler(self, target_logging_path):
        raise NotImplemented

    def put_dir_from_local_dir(self, local_dir, target_dir) -> bool:
        raise NotImplemented

    def exists(self, target_path) -> bool:
        req = urllib.request.Request(target_path)
        req.get_method = lambda: 'HEAD'

        try:
            urllib.request.urlopen(req)
            return True
        except Exception as e:
            return False

    def isfile(self, target_path) -> bool:
        # Well for a http url, it should only be a file.
        return True

    def isdir(self, target_path) -> bool:
        return False
