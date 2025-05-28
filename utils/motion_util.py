import torch

# 假设 joints 是原始张量，shape 为 [64, 196, 22, 3]
def pad_joints_to_24(joints_22):
    B, T, J, C = joints_22.shape  # B:64, T:196, J:22, C:3
    assert J == 22 and C == 3, "输入的 joints 应该是 [B, T, 22, 3]"

    # 初始化一个全 0 的张量 [B, T, 24, 3]
    joints_24 = torch.zeros((B, T, 24, 3), dtype=joints_22.dtype, device=joints_22.device)

    # 前 22 个关节直接复制
    joints_24[:, :, :22, :] = joints_22

    # 最后两个关节保持为 0（默认填充即可）
    return joints_24