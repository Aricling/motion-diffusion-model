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

def save_config(config_path, cfg, overwrite=False):
    """
    将配置对象（可能为 SimpleNamespace 或嵌套结构）保存为 YAML 文件。

    Args:
        config_path (str): 要保存的 YAML 配置文件路径。
        cfg: 配置对象，通常是 SimpleNamespace 或 dict 的嵌套结构。
        overwrite (bool): 是否允许覆盖已存在的配置文件。默认为 False。

    Raises:
        ValueError: 如果文件已存在且不允许覆盖时抛出。
    """
    def namespace_to_dict(ns):
        if isinstance(ns, SimpleNamespace):
            return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
        elif isinstance(ns, dict):
            return {k: namespace_to_dict(v) for k, v in ns.items()}
        else:
            return ns

    if os.path.exists(config_path) and not overwrite:
        raise ValueError(f"Config file already exists at {config_path} and overwrite is not enabled.")
    else:
        with open(config_path, 'w') as f:
            cfg_dict = namespace_to_dict(cfg)
            yaml.safe_dump(cfg_dict, f, sort_keys=False)