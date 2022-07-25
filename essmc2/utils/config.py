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
import json
import numbers
import os
import os.path as osp
import sys
import warnings
from importlib import import_module
from typing import Optional, Type

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

_BASE_KEY = "_base_"
_DELETE_KEY = "_delete_"
_RESERVE_KEY = "_reserve_"


class ValueComment(object):
    """ Add a comment to the value.
    """

    def __init__(self, value, comment):
        self.value = value
        self.comment = comment

    def __repr__(self):
        return f"{self.value} # {self.comment}"


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
                if type(value_b) is type(None):
                    b[key] = value
                elif type(value) is type(None):
                    b[key] = None
                elif isinstance(value, numbers.Number) and isinstance(value_b, numbers.Number):
                    b[key] = value
                elif type(value) == type(value_b):
                    if type(value) is ConfigDict:
                        b[key] = ConfigDict.merge_a_into_b(value, value_b)
                    else:
                        b[key] = value
                else:
                    raise ValueError(f"Except source type {type(value_b)}, got {type(value)}")
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
        warnings.warn("Config.load_file is deprecated, use Config.load instead. Will be removed after v0.1.0.")
        return Config.load(filename)

    @staticmethod
    def load(filename):
        """ Load config from filename, currently only support python file

        Args:
            filename (str): config file path.

        Returns:
            Config instance.

        Raises:
            An exception will raise if unsupported type file occurs.
        """
        if filename.endswith((".py", ".pysc", ".pyc", ".pyo", ".pyd", ".pyx")):
            cfg_dict = Config._parse_python_file(filename)
        elif filename.endswith(".json"):
            cfg_dict = Config._parse_json_file(filename)
        else:
            raise Exception(f"Unsupported filename {filename}")

        cfg_dict = Config._process_predefined_vars(cfg_dict, filename=filename)

        return Config(cfg_dict=cfg_dict, source=filename)

    @staticmethod
    def _parse_python_file(filename) -> dict:
        """ Parse python file and load basic elements without functions, modules, classes, etc...
        """
        filepath = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filepath):
            raise Exception(f"File {filepath} not found")
        if not filepath.endswith((".py", ".pysc", ".pyc", ".pyo", ".pyd", ".pyx")):
            raise Exception(f"File {filepath} is not python file")
        file_dir = os.path.dirname(filepath)
        sys.path.insert(0, file_dir)
        module_name, _ = os.path.splitext(os.path.basename(filepath))
        module = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {name: value for name, value in module.__dict__.items() if not name.startswith("__") and
                    not inspect.isfunction(value) and not inspect.ismodule(value) and not inspect.isclass(value)}
        del sys.modules[module_name]
        return cfg_dict

    @staticmethod
    def _parse_json_file(filename) -> dict:
        """ Parse json file
        """
        with open(filename) as f:
            content = f.read()
        return Config._loads_json(content)

    @staticmethod
    def loads(s, loads_format="json"):
        assert loads_format in ("json",)
        if loads_format == "json":
            cfg_dict = Config._loads_json(s)
        else:
            raise Exception(f"Unsupported type {loads_format}, except ('json', )")

        cfg_dict = Config._process_predefined_vars(cfg_dict)

        return Config(cfg_dict=cfg_dict)

    @staticmethod
    def _loads_json(s) -> dict:
        cfg_dict = json.loads(s)
        if type(cfg_dict) is not dict:
            raise Exception(f"Json should contains dict type, get {type(cfg_dict)}")
        return cfg_dict

    @staticmethod
    def _process_predefined_vars(cfg_dict: dict, filename: Optional[str] = None) -> Type[dict]:
        # process _base_ field
        if _BASE_KEY in cfg_dict:
            _base_list = cfg_dict.pop(_BASE_KEY)
            if isinstance(_base_list, (list, tuple)):
                _base_list = list(_base_list)
            elif isinstance(_base_list, str):
                _base_list = [_base_list]
            else:
                raise ValueError('Input '
                                 + ('' if filename is None else f'{filename} ')
                                 + f'contains {_BASE_KEY} filed, which should be type [list, tuple, str], '
                                   f'get {type(_base_list)}'
                                 )
            # When filename is not None, _base_ fields contains relative pathes to filename
            # Else contains absolute path
            if filename is not None:
                b = Config.load(osp.abspath(osp.expanduser(osp.join(osp.dirname(filename), _base_list[0]))))
            else:
                b = Config.load(_base_list[0])
            for _a in _base_list[1:]:
                if filename is not None:
                    a = Config.load(osp.abspath(osp.expanduser(osp.join(osp.dirname(filename), _a))))
                else:
                    a = Config.load(_a)
                b = Config.merge_a_into_b(a, b)
            b = Config.merge_a_into_b(Config(cfg_dict=cfg_dict), b)
            cfg_dict = b.__getattribute__('_cfg_dict')

        # process _delete_ or _reserve_ field
        if _DELETE_KEY in cfg_dict or _RESERVE_KEY in cfg_dict:
            if _DELETE_KEY in cfg_dict and _RESERVE_KEY in cfg_dict:
                warnings.warn(
                    f"Get {_DELETE_KEY}: {cfg_dict[_DELETE_KEY]} and {_RESERVE_KEY}: {cfg_dict[_RESERVE_KEY]} in"
                    + ('' if filename is None else f'{filename} ')
                    + f" at the same time, will ONLY use {_RESERVE_KEY} and drop {_DELETE_KEY}")
                cfg_dict.pop(_DELETE_KEY)

            # process _delete_ field
            if _DELETE_KEY in cfg_dict:
                _del_list = cfg_dict.pop(_DELETE_KEY)
                if isinstance(_del_list, (list, tuple)):
                    _del_list = list(_del_list)
                elif isinstance(_del_list, str):
                    _del_list = [_del_list]
                else:
                    raise ValueError('Input '
                                     + ('' if filename is None else f'{filename} ')
                                     + f'contains {_DELETE_KEY} filed, which should be type [list, tuple, str], '
                                       f'get {type(_base_list)}'
                                     )
                for del_key in _del_list:
                    if del_key in cfg_dict:
                        cfg_dict.pop(del_key)
            else:
                # process _reserve_ field
                _reserve_list = cfg_dict.pop(_RESERVE_KEY)
                if isinstance(_reserve_list, (list, tuple)):
                    _reserve_list = list(_reserve_list)
                elif isinstance(_reserve_list, str):
                    _reserve_list = [_reserve_list]
                else:
                    raise ValueError('Input '
                                     + ('' if filename is None else f'{filename} ')
                                     + f'contains {_RESERVE_KEY} filed, which should be type [list, tuple, str], '
                                       f'get {type(_base_list)}'
                                     )
                new_dict = {}
                for reserve_key in _reserve_list:
                    if reserve_key in cfg_dict:
                        new_dict[reserve_key] = cfg_dict[reserve_key]
                cfg_dict = new_dict

        return cfg_dict

    def dump(self, file):
        """ Dump current config to a file, currently only support python file

        Args:
            file (str): file to be dumped

        Returns:

        Raises:
            An exception will raise if unsupported type file inputs.
        """
        if file.endswith((".py", ".pysc", ".pyc", ".pyo", ".pyd", ".pyx")):
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
            DEDENT_CLOSING_BRACKETS=True,
            COLUMN_LIMIT=120,
        )
        text, _ = FormatCode(s, style_config=style, verify=True)
        return text

    @staticmethod
    def _dump_list(v):
        v_str = '['

        s = []
        s_comments = []

        for vv in v:
            # check if comment exists
            comment_str = ''
            if isinstance(vv, ValueComment):
                comment_str = vv.comment
                comment_str = comment_str.replace('\n', '# \n')
                vv = vv.value

            if isinstance(vv, dict):
                tmp = Config._dump_dict(vv)
                o_str = f'dict({tmp})'
                s.append(o_str)
            elif isinstance(vv, list):
                s.append(Config._dump_list(vv))
            else:
                s.append(f"'{vv}'" if isinstance(vv, str) else str(vv))

            s_comments.append(comment_str)

        for _s, _comment in zip(s, s_comments):
            if len(_comment) > 0:
                v_str += (f"{_s}, " + f"  # {_comment}\n")
            else:
                v_str += f"{_s}, "
        v_str += ']'

        return v_str

    @staticmethod
    def _dump_dict(input_dict, root_level=False, secure=True, secure_keys=None) -> str:
        if secure_keys is None:
            secure_keys = set(_SECURE_KEYWORDS)
        s = []
        s_comments = []

        for i, (k, v) in enumerate(input_dict.items()):
            # Secure mode, replace value by ***
            if secure and k in secure_keys and type(v) is str:
                v = "***"
            k_str = f"{k}" if isinstance(k, str) else str(k)
            end = '' if i == len(input_dict) - 1 or root_level else ' '

            comment_str = ''
            if isinstance(v, ValueComment):
                comment_str = v.comment
                comment_str = comment_str.replace('\n', '\n# ')
                v = v.value

            if isinstance(v, dict):
                v_str = Config._dump_dict(v)
                o_str = f'{k_str}=dict({v_str}' + ')' + end
            elif isinstance(v, list):
                o_str = f'{k_str}={Config._dump_list(v)}' + end
            else:
                v_str = f"'{v}'" if isinstance(v, str) else str(v)
                o_str = f"{k_str}={v_str}"
            s.append(o_str)
            s_comments.append(comment_str)

        ret = ""
        for _s, _s_comment in zip(s, s_comments):
            if root_level:
                if len(_s_comment) > 0:
                    ret += (_s + "  # " + _s_comment + '\n')
                else:
                    ret += (_s + '\n')
            else:
                if len(_s_comment) > 0:
                    ret += (_s + ",  # " + _s_comment + "\n")
                else:
                    ret += (_s + ', \n')

        return ret

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


cfg = Config()
