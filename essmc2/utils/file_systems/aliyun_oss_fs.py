# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import logging
import os
import os.path as osp
import random
import tempfile
import warnings
from typing import Optional

import oss2

from .base_fs import BaseFs
from .registry import FILE_SYSTEMS


class OssLoggingHandler(logging.StreamHandler):
    def __init__(self, bucket, log_file):
        super(OssLoggingHandler, self).__init__()
        self._bucket: oss2.Bucket = bucket
        self._log_file = log_file
        if self._bucket.object_exists(self._log_file):
            size = self._bucket.get_object_meta(self._log_file).content_length
        else:
            size = 0
        self._pos = self._bucket.append_object(self._log_file, size, '')

    def emit(self, record):
        msg = self.format(record) + "\n"
        try:
            self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)
        except oss2.exceptions.PositionNotEqualToLength as e:
            self._pos = self._bucket.get_object_meta(self._log_file).content_length
            self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)


@FILE_SYSTEMS.register_class()
class AliyunOssFs(BaseFs):
    def __init__(self, endpoint, bucket, ak, sk, prefix=None, writable=True, check_writable=False, retry_times=10):
        super(AliyunOssFs, self).__init__()
        self._bucket: oss2.Bucket = oss2.Bucket(oss2.Auth(ak, sk), endpoint, bucket)
        self._fs_prefix = f"oss://{bucket}/" + ("" if prefix is None else prefix)
        self._prefix = f"oss://{bucket}/"
        try:
            self._bucket.list_objects(max_keys=1)
        except Exception as e:
            warnings.warn(f"Cannot list objects in {self._prefix}, please check auth information. \n{e}")
        self._retry_times = retry_times

        self._writable = writable
        if check_writable:
            self._writable = self._test_write(bucket, prefix)

    def _test_write(self, bucket, prefix) -> bool:
        local_tmp_file = osp.join(tempfile.gettempdir(),
                                  f"oss_{bucket}_{'' if prefix is None else prefix}_try_test_write" + ''.join(
                                      [str(random.randint(1, 10) for _ in range(5))]))
        with open(local_tmp_file, "w") as f:
            f.write("Try to write")
        target_tmp_file = osp.join(self._prefix, osp.basename(local_tmp_file))
        status = self.put_object_from_local_file(local_tmp_file, target_tmp_file)
        if status:
            self.remove(target_tmp_file)
        return status

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
        os.makedirs(osp.dirname(local_path), exist_ok=True)

        retry = 0
        while retry < self._retry_times:
            try:
                self._bucket.get_object_to_file(key, local_path)
                if osp.exists(local_path):
                    break
            except oss2.exceptions.NoSuchKey as e:
                return None
            except Exception as e:
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
                self._bucket.put_object_from_file(key, local_path)
                if self._bucket.object_exists(key):
                    break
            except Exception as e:
                retry += 1

        if retry >= self._retry_times:
            return False

        return True

    def make_link(self, target_link_path, target_path) -> bool:
        if not self.support_link():
            return False
        link_key = osp.relpath(target_link_path, self._prefix)
        target_key = osp.relpath(target_path, self._prefix)
        try:
            self._bucket.put_symlink(target_key, link_key)
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
            self._bucket.delete_object(key)
            return True
        except Exception as e:
            print(e)
            return False

    def get_logging_handler(self, target_logging_path):
        oss_key = osp.relpath(target_logging_path, self._prefix)
        return OssLoggingHandler(self._bucket, oss_key)

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
            return self._bucket.object_exists(osp.relpath(target_path, self._prefix))
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
