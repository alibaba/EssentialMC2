# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import logging
import os
import os.path as osp
import random
import datetime
import tempfile

import oss2
from typing import Optional

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


class OssLoggingHandler(logging.StreamHandler):
    def __init__(self, bucket, log_file):
        super(OssLoggingHandler, self).__init__()
        self._bucket: oss2.Bucket = bucket
        self._log_file = log_file
        self._pos = self._bucket.append_object(self._log_file, 0, '')

    def emit(self, record):
        msg = self.format(record) + "\n"
        self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)


@FILE_SYSTEMS.register_class()
class AliyunOssFs(BaseFs):
    def __init__(self, endpoint, bucket, ak, sk, retry_times=10):
        super(AliyunOssFs, self).__init__()
        self.endpoint = endpoint
        self.bucket_name = bucket
        self.ak = ak
        self.sk = sk
        self.bucket: Optional[oss2.Bucket] = None
        self.retry_times = retry_times
        self.prefix = f"oss://{self.bucket_name}/"

    def _init_bucket(self):
        if self.bucket is None:
            self.bucket = oss2.Bucket(oss2.Auth(self.ak, self.sk), self.endpoint, self.bucket_name)

    def get_object_to_local_file(self, path) -> str:
        self._init_bucket()

        key = path[len(self.prefix):]
        basename = osp.basename(path)
        randname = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())+''.join([str(random.randint(1,10)) for i in range(5)])
        tmp_file = osp.join(tempfile.gettempdir(), randname + '_' + basename)
        retry = 0
        while retry < self.retry_times:
            try:
                self.bucket.get_object_to_file(key, tmp_file)
                if osp.exists(tmp_file):
                    break
            except oss2.exceptions.NoSuchKey as e:
                raise e
            except Exception as e:
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
        self._init_bucket()

        key = target_path[len(self.prefix):]
        retry = 0
        while retry < self.retry_times:
            try:
                self.bucket.put_object_from_file(key, local_path)
                if self.bucket.object_exists(key):
                    break
            except Exception:
                retry += 1

    def get_prefix(self):
        return self.prefix

    def support_write(self):
        return True

    def get_logging_handler(self, logging_path):
        self._init_bucket()
        oss_key = osp.relpath(logging_path, self.prefix)
        return OssLoggingHandler(self.bucket, oss_key)
