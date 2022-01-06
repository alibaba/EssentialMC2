from essmc2.utils.config import Config


def test_load_py_config():
    py_file = "../data/utils/config_test.py"
    cfg = Config.load_file(py_file)
    assert cfg.a == 100


def test_load_json_config():
    json_file = "../data/utils/config_test.json"
    cfg = Config.load_file(json_file)
    assert cfg.a == 100


def test_loads_json_config():
    json_str = """
    {
      "a": 100,
      "b": false
    }
    """
    cfg = Config.loads(json_str, loads_format="json")
    assert cfg.a == 100


def test_dumps_config():
    py_file = "../data/utils/config_test.py"
    cfg = Config.load_file(py_file)

    py_s = cfg.dumps(dump_format="py")
    assert len(py_s) > 0

    json_s = cfg.dumps(dump_format="json")

    assert len(json_s) > 0



def test_set_value():
    json_str = """
        {
          "a": 100,
          "b": false
        }
        """
    cfg = Config.loads(json_str, loads_format="json")
    cfg.c = dict(d=100, e=dict(v=200))
    cfg.c.e.v = 300
    assert cfg.c.e.v == 300


def test_merge():
    json_file = "../data/utils/config_test.json"
    cfg = Config.load_file(json_file)
    cfg.none_value = None
    cfg.not_none_value = "Not none"

    json_str = """
    {
        "a": 200,
        "f": {
            "type": "InceptionResnet"
        },
        "h": "NewKey"
    }
    """
    cfg_add = Config.loads(json_str, loads_format="json")
    cfg_add.none_value = 1
    cfg_add.not_none_value = None

    cfg_new = Config.merge_a_into_b(cfg_add, cfg)

    assert cfg_new.a == 200
    assert cfg_new.f.type == "InceptionResnet"
    assert cfg_new.h == "NewKey"
    assert cfg_new.g == cfg.g
    assert cfg_new.f.mean == cfg.f.mean
    assert cfg_new.none_value == 1
    assert cfg_new.not_none_value is None
