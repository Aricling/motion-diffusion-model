import yaml
from types import SimpleNamespace
import os

cfg=None

def _dict_to_namespace(d):
    """递归将 dict 转换为支持点访问的 SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

def init_diy_config(diy_cfg_path=None):
    if diy_cfg_path is None:
        # 加载 config.yaml
        config_path = os.path.join(os.path.dirname(__file__), 'z_config.yaml')
    else:
        config_path=diy_cfg_path

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 转为全局对象
    global cfg
    cfg = _dict_to_namespace(config_dict)

def get_diy_config():
    global cfg
    return cfg
