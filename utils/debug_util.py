import numpy as np
import os

import numpy as np
import os
import torch
from typing import Any, Dict, Optional

def save_debug_tensors(save_path: str, filename: str, **kwargs):
    """
    将任意类型的输入（tensor, list, int, float, str, None 等）保存为 .npz 文件
    自动处理：
        - GPU tensors -> .cpu()
        - requires_grad tensors -> .detach()
        - list/tuple (包括 str list, int list, tensor list)
        - None 值
        - 字符串、标量等
    参数:
        save_path: 保存目录（如 "./debug"）
        filename: 文件名（如 "batch_0.npz"）
        **kwargs: 要保存的变量，如 model_output=xxx, db_key=yyy
    """
    def to_numpy(x):
        if x is None:
            return None

        # 处理 PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.numpy()

        # 处理 list 或 tuple
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return np.array(x, dtype=object)

            # 检查是否包含 tensor
            if any(isinstance(item, torch.Tensor) for item in x):
                processed = []
                for item in x:
                    if isinstance(item, torch.Tensor):
                        processed.append(item.detach().cpu().numpy())
                    else:
                        processed.append(item)
                return np.array(processed, dtype=object)
            else:
                # 普通 list（如 str, int），转为 object 类型 numpy array
                return np.array(x, dtype=object)

        # 处理 numpy 数组
        if isinstance(x, np.ndarray):
            return x

        # 处理标量（int, float, bool）
        if isinstance(x, (int, float, bool)):
            return np.array(x)

        # 处理字符串
        if isinstance(x, str):
            return np.array(x, dtype=object)

        # 其他类型（尝试转为 object）
        try:
            return np.array(x, dtype=object)
        except Exception as e:
            print(f"Warning: Failed to convert {type(x)} to numpy. Saving as string.")
            return np.array(str(x), dtype=object)

    # 创建目录
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)

    # 转换所有输入
    saved_data = {}
    for key, value in kwargs.items():
        try:
            saved_data[key] = to_numpy(value)
        except Exception as e:
            print(f"Error converting {key}: {e}")
            saved_data[key] = np.array(str(value), dtype=object)

    # 保存为 .npz 文件
    np.savez(file_path, **saved_data)
    print(f"[Debug] Saved debug data to: {file_path}")
    print(f"[Debug] Keys saved: {list(saved_data.keys())}")