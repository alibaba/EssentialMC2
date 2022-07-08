# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import logging
import io
import os
import os.path as osp
import random
import tempfile
import warnings
from typing import Optional

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


class MinioLoggingHandler(logging.StreamHandler):
    def __init__(self, client, bucket_name, log_file):
        super(MinioLoggingHandler, self).__init__()
        import minio
        
        self._bucket_name = bucket_name
        self._client: minio.Minio = client
        self._log_file = log_file
        self._msg = ""

    def emit(self, record):
        self._msg += self.format(record) + "\n"
        self._client.put_object(self._bucket_name, self._log_file, io.BytesIO(self._msg.encode('utf-8')), length=len(self._msg))


@FILE_SYSTEMS.register_class()
class MinioFs(BaseFs):
    def __init__(self, endpoint, bucket, ak, sk, prefix=None, writable=True, check_writable=False, retry_times=10):
        super(MinioFs, self).__init__()
        import minio

        self._bucket_name = bucket
        self._endpoint = endpoint
        self._client: minio.Minio = minio.Minio(self._endpoint, access_key=ak, secret_key=sk, secure=False)
        self._fs_prefix = f"{self._bucket_name}/" + ("" if prefix is None else prefix)
        self._prefix = f"{self._bucket_name}/"
        try:
            self._client.list_objects(self._bucket_name)
        except Exception as e:
            warnings.warn(f"Cannot list objects in {self._prefix}, please check auth information. \n{e}")
        self._retry_times = retry_times

        self._writable = writable
        

    def get_prefix(self) -> str:
        return self._fs_prefix

    def support_write(self) -> bool:
        return self._writable

    def support_link(self) -> bool:
        return self._writable

    def get_object_to_local_file(self, target_path, local_path=None) -> Optional[str]:
        if local_path is None:
            local_path, is_tmp = self.map_to_local(target_path)
        else:
            is_tmp = False

        key = osp.relpath(target_path, self._prefix)
        local_dir = osp.dirname(local_path)
        if local_dir != "": os.makedirs(local_dir, exist_ok=True)

        retry = 0
        while retry < self._retry_times:
            try:
                result = self._client.fget_object(self._bucket_name, key, local_path)
                if osp.exists(local_path):
                    break
            except Exception as e:
                print(e)
                retry += 1

        if retry >= self._retry_times:
            return None

        if is_tmp:
            self.add_temp_file(local_path)

        return local_path

    def put_object_from_local_file(self, local_path, target_path) -> bool:
        key = osp.relpath(target_path, self._prefix)
        retry = 0
        while retry < self._retry_times:
            try:
                stat = self._client.fput_object(self._bucket_name, key, local_path)
                stat = self._client.stat_object(self._bucket_name, key)
                break
            except Exception as e:
                print(e)
                retry += 1

        if retry >= self._retry_times:
            return False

        return True

    def make_link(self, target_link_path, target_path) -> bool:
        from minio.commonconfig import CopySource

        if not self.support_link():
            return False
        link_key = osp.relpath(target_link_path, self._prefix)
        target_key = osp.relpath(target_path, self._prefix)
        try:
            result = self._client.copy_object(self._bucket_name, link_key, CopySource(self._bucket_name, target_key))
        except Exception as e:
            print(e)
            return False
        return True

    def make_dir(self, target_dir) -> bool:
        # OSS treat file path as a key, it will create directory automatically when putting a file.
        return True

    def remove(self, target_path) -> bool:
        key = osp.relpath(target_path, self._prefix)
        try:
            self._client.remove_object(self._bucket_name, key)
            return True
        except Exception as e:
            print(e)
            return False

    def get_logging_handler(self, target_logging_path):
        minio_key = osp.relpath(target_logging_path, self._prefix)
        return MinioLoggingHandler(self._client, self._bucket_name, minio_key)

    def put_dir_from_local_dir(self, local_dir, target_dir) -> bool:
        for folder, sub_folders, files in os.walk(local_dir):
            for file in files:
                file_abs_path = osp.join(folder, file)
                file_rel_path = osp.relpath(file_abs_path, local_dir)
                target_path = osp.join(target_dir, file_rel_path)
                status = self.put_object_from_local_file(file_abs_path, target_path)
                if not status:
                    return False
        return True

    def exists(self, target_path) -> bool:
        try:
            stat = self._client.stat_object(self._bucket_name, osp.relpath(target_path, self._prefix))
            return True
        except Exception as e:
            print(e)
            return False

    def isfile(self, target_path) -> bool:
        if target_path.endswith('/'):
            return False
        # maybe other conditions should in considerations
        return True

    def isdir(self, target_path) -> bool:
        if not target_path.endswith('/'):
            target_path += '/'
        return self.exists(target_path)
