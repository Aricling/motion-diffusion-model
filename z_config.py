import yaml
from types import SimpleNamespace
import os

def _dict_to_namespace(d):
    """递归将 dict 转换为支持点访问的 SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

# 加载 config.yaml
config_path = os.path.join(os.path.dirname(__file__), 'z_config.yaml')

with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# 转为全局对象
cfg = _dict_to_namespace(config_dict)
