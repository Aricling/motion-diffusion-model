


def remove_module_prefix(state_dict):
    """移除 state_dict 中的 'module.' 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict