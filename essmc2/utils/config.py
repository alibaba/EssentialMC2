# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

# Config class partially modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import os
import sys
import json
from importlib import import_module

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

_SECURE_KEYWORDS = [
    "privateKey",
    "access-key",
    "ak",
    "access_id",
    "access_key",
    "accessKey",
    "password",
    "pwd",
    "secret",
    "appkey",
    "appKey",
    "scret",
    "ak.value",
    "sk"
]


class ConfigDict(Dict):
    """ Class Dict will set default value {} if key not exists, which is not expected.
    We can set value via non-exist key:
    >>> cfg = ConfigDict({})
    >>> cfg.key = 100
    >>> print(cfg.key)
    >>> 100
    We can get value by key or key sequences:
    >>> cfg.key = dict(a=dict(b=dict(c=100)))
    >>> print(cfg.key.a.b.c)
    >>> 100
    We cannot set value by key sequences:
    >>> cfg.key.d.e = 100
    >>> KeyError: 'd'
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        return super(ConfigDict, self).__getattr__(name)

    @staticmethod
    def merge_a_into_b(a, b):
        """ Merge ConfigDict object a into ConfigDict object b
        Args:
            a (ConfigDict): Source ConfigDict object to merge
            b (ConfigDict): Target ConfigDict object to be merged into

        Returns:
            A new ConfigDict.
        """
        b = copy.deepcopy(b)
        for key, value in a.items():
            if key in b.keys():
                value_b = b[key]
                if type(value) is ConfigDict:
                    if type(value_b) is not ConfigDict:
                        raise ValueError(f"Except source type {type(value_b)}, got {type(value)}")
                    b[key] = ConfigDict.merge_a_into_b(value, value_b)
                else:
                    if type(value) != type(value_b):
                        raise ValueError(f"Except source type {type(value_b)}, got {type(value)}")
                    b[key] = value
            else:
                b[key] = value

        return b


class Config(object):
    """ Config parser and pretty printer.

    Currently, it only support python and json config file.

    Example:
        >>> cfg = Config(dict(a=1, b=(1, )))
        >>> cfg.dumps()
        a = 1\nb = (1, )\n
        >>> cfg.a
        1
        >>> cfg = Config.load_file("config.py")
    """

    def __init__(self, cfg_dict=None, source=None, secure=True, secure_keys=None):
        _cfg_dict = cfg_dict if cfg_dict is not None else {}
        super(Config, self).__setattr__("_cfg_dict", ConfigDict(_cfg_dict))
        super(Config, self).__setattr__("_source", source)
        super(Config, self).__setattr__("_secure", secure)
        super(Config, self).__setattr__("_secure_keys", set(secure_keys or _SECURE_KEYWORDS))

    def __repr__(self):
        return (f"Config from {self._source}: \n" if self._source is not None else "") + f"{self.dumps()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return self._cfg_dict.__getattr__(name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        elif isinstance(value, (list, tuple)):
            value = type(value)(
                ConfigDict(item) if isinstance(item, dict) else
                item for item in value)
        self._cfg_dict.__setitem__(key, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def get(self, name, default=None):
        """ Get value by name

        Args:
            name (Any):
            default (Any): Default value if name missed.

        Returns:
            value (Any, None): Will only return primitive type dict, not ConfigDict
        """
        if name not in self._cfg_dict:
            return default
        else:
            value = self._cfg_dict.__getitem__(name)
            if type(value) is ConfigDict:
                value = value.to_dict()
            elif isinstance(value, (list, tuple)):
                value = type(value)(
                    item.to_dict() if isinstance(item, ConfigDict) else
                    item for item in value)

            return value

    @staticmethod
    def merge_a_into_b(a, b):
        """ Merge Config object a into Config object b
        Args:
            a (Config): Source Config object to merge
            b (Config): Target Config object to be merged into

        Returns:
            A new Config.
        """
        cfg_dict = ConfigDict.merge_a_into_b(a.__getattribute__("_cfg_dict"), b.__getattribute__("_cfg_dict"))
        return Config(cfg_dict=cfg_dict)

    def make_global_visible(self):
        """ Assign this object to cfg object, so that we can use `from essmc2.utils.config import cfg` everywhere
        except sub process workers.
        """
        global cfg
        cfg._cfg_dict = self._cfg_dict

    def set_secure(self, flag):
        """ Set secure flag, if true, value whose key name in self._secure_keys will be replaced by ***
        Args:
            flag (bool): Secure flag.
        """
        self._secure = flag

    def set_secure_keys(self, secure_keys):
        """ Add new secure keys

        Args:
            secure_keys (iterable): New secure keys.
        """

        self._secure_keys.update(secure_keys)

    @staticmethod
    def load_file(filename):
        """ Load config from filename, currently only support python file

        Args:
            filename (str): config file path.

        Returns:
            Config instance.

        Raises:
            An exception will raise if unsupported type file occurs.
        """
        if filename.endswith(".py"):
            return Config._parse_python_file(filename)
        elif filename.endswith(".json"):
            return Config._parse_json_file(filename)
        raise Exception(f"Unsupported filename {filename}")

    @staticmethod
    def loads(s, loads_format="json"):
        assert loads_format in ("json",)
        if loads_format == "json":
            return Config._loads_json(s)
        raise Exception(f"Unsupported type {loads_format}, except ('json', )")

    @staticmethod
    def _parse_python_file(filename):
        """ Parse python file and load basic elements without functions, modules, classes, etc...
        """
        filepath = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filepath):
            raise Exception(f"File {filepath} not found")
        if not filepath.endswith(".py"):
            raise Exception(f"File {filepath} is not python file")
        file_dir = os.path.dirname(filepath)
        sys.path.insert(0, file_dir)
        module_name = os.path.basename(filepath).replace('.py', '')
        module = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {name: value for name, value in module.__dict__.items() if not name.startswith("__") and
                    not inspect.isfunction(value) and not inspect.ismodule(value) and not inspect.isclass(value)}
        del sys.modules[module_name]
        return Config(cfg_dict=cfg_dict, source=filename)

    @staticmethod
    def _parse_json_file(filename):
        """ Parse json file
        """
        with open(filename) as f:
            content = f.read()
        return Config._loads_json(content, filename)

    @staticmethod
    def _loads_json(s, filename=None):
        cfg_dict = json.loads(s)
        if type(cfg_dict) is not dict:
            raise Exception(f"Json should contains dict type, get {type(cfg_dict)}")
        return Config(cfg_dict=cfg_dict, source=filename)

    def dump(self, file):
        """ Dump current config to a file, currently only support python file

        Args:
            file (str): file to be dumped

        Returns:

        Raises:
            An exception will raise if unsupported type file inputs.
        """
        if file.endswith(".py"):
            s = self.dumps(dump_format="py")
        elif file.endswith(".json"):
            s = self.dumps(dump_format="json")
        else:
            raise Exception(f"Unsupported filename {file}, except format (.py, .json)")

        with open(file, "w") as f_out:
            f_out.write(s)

    def dumps(self, dump_format="py"):
        """ Dump current config to a string, currently only support python file

        Args:
            dump_format (str): format name, currently only support 'py'

        Returns:
            str, which contains config content

        Raises:
            An exception will raise if unsupported format occurs.
        """
        if dump_format == "py":
            return self._dumps_python()
        elif dump_format == "json":
            return self._dumps_json()
        else:
            raise ValueError(f"Unsupported dump format {dump_format}")

    def _dumps_python(self):
        s = Config._dump_dict(self._cfg_dict, True, self._secure, self._secure_keys)
        style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
            DEDENT_CLOSING_BRACKETS=True)
        text, _ = FormatCode(s, style_config=style, verify=True)
        return text

    def _dumps_json(self):
        deep_copy = copy.deepcopy(self._cfg_dict)

        secure_keys = set(self._secure_keys)

        def make_secure(value):
            if isinstance(value, dict):
                for key in value.keys():
                    if key in secure_keys and type(value[key]) is str:
                        value[key] = "***"
                    else:
                        value[key] = make_secure(value[key])
            elif isinstance(value, list):
                value = [make_secure(t) for t in value]

            return value

        if self._secure:
            deep_copy = make_secure(deep_copy)

        return json.dumps(deep_copy, ensure_ascii=False, indent=2)

    @staticmethod
    def _dump_list(v):
        v_str = '['
        s = []
        for vv in v:
            if isinstance(vv, dict):
                tmp = Config._dump_dict(vv)
                o_str = f'dict({tmp})'
                s.append(o_str)
            elif isinstance(vv, list):
                s.append(Config._dump_list(vv))
            else:
                s.append(f"'{vv}'" if isinstance(vv, str) else str(vv))
        v_str += ', '.join(s)
        v_str += ']'
        return v_str

    @staticmethod
    def _dump_dict(input_dict, root_level=False, secure=True, secure_keys=None) -> str:
        if secure_keys is None:
            secure_keys = set(_SECURE_KEYWORDS)
        s = []

        for i, (k, v) in enumerate(input_dict.items()):
            # Secure mode, replace value by ***
            if secure and k in secure_keys and type(v) is str:
                v = "***"
            k_str = f"{k}" if isinstance(k, str) else str(k)
            end = '' if i == len(input_dict) - 1 or root_level else ' '
            if isinstance(v, dict):
                v_str = Config._dump_dict(v)
                o_str = f'{k_str}=dict({v_str}' + ')' + end
            elif isinstance(v, list):
                o_str = f'{k_str}={Config._dump_list(v)}' + end
            else:
                v_str = f"'{v}'" if isinstance(v, str) else str(v)
                o_str = f"{k_str}={v_str}"
            s.append(o_str)

        return '\n'.join(s) if root_level else ', '.join(s)


cfg = Config()
