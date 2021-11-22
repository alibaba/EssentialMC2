# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os
import os.path as osp
import random
import datetime
import tempfile
import urllib.parse as parse
import urllib.request as request

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


def _get_http_url_basename(url):
    url = parse.unquote(url)
    url = url.split("?")[0]
    return osp.basename(url)


@FILE_SYSTEMS.register_class()
class HttpFs(BaseFs):
    def __init__(self, retry_times=10):
        super(HttpFs, self).__init__()
        self.retry_times = retry_times

    def get_object_to_local_file(self, path) -> str:
        basename = _get_http_url_basename(path)
        randname = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())+''.join([str(random.randint(1,10)) for i in range(5)])
        tmp_file = osp.join(tempfile.gettempdir(), randname + '_' + basename)
        retry = 0
        while retry < self.retry_times:
            try:
                request.urlretrieve(path, tmp_file)
                if osp.exists(tmp_file):
                    break
            except Exception:
                retry += 1
        self.to_removes.add(tmp_file)
        return tmp_file

    def get_object_to_memory(self, path) -> bytes:
        tmp_file = self.get_object_to_local_file(path)
        with open(tmp_file, "rb") as f:
            content = f.read()
        self.remove_local_file(tmp_file)
        self.to_removes.remove(tmp_file)
        return content

    def remove_local_file(self, local_path):
        try:
            os.remove(local_path)
            if local_path in self.to_removes:
                self.to_removes.remove(local_path)
        except:
            pass

    def put_object_from_local_file(self, local_path, target_path):
        raise NotImplemented

    def get_prefix(self):
        return "http"

    def support_write(self):
        return False

    def get_logging_handler(self, logging_path):
        raise NotImplemented
