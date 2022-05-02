import os.path as osp

from essmc2.utils.config import Config


def get_train_base_config() -> Config:
    return Config.load(osp.join(osp.dirname(__file__), '_standard_train_config.py'))


def get_test_base_config() -> Config:
    return Config.load(osp.join(osp.dirname(__file__), '_standard_test_config.py'))
