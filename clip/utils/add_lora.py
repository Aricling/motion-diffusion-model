import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from utils import dist_util
import z_config

def prepare_lora_clip_model(
    clip_model,
    use_lora: bool = True,
    lora_mlp: bool = True,
    lora_attn: bool = True,
    num_new_tokens: int = 3,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    对 CLIP 文本编码器进行 LoRA 改造，并扩展词表，冻结原始模型部分参数，仅允许新增 token 和 LoRA 参数更新。

    Args:
        clip_model: 原始的 CLIP 模型（文本部分）
        use_lora: 是否应用 LoRA
        lora_mlp: 是否在 MLP 层应用 LoRA
        lora_attn: 是否在 Attention 层应用 LoRA
        num_new_tokens: 扩展多少个新 token（如用于特殊标记）
        save_dir: （可选）保存可训练参数名称的路径
        device: 设备（默认使用模型所在设备）

    Returns:
        处理完成的 CLIP 模型（已添加 LoRA、扩展 embedding、设置梯度钩子）
    """
    # 获取设备
    dev = dist_util.dev() if device is None else device

    # 1. 克隆原始模型用于推理（冻结）
    import copy
    clip_model_ori = copy.deepcopy(clip_model).eval()
    clip_model = copy.deepcopy(clip_model)
    for param in clip_model_ori.parameters():
        param.requires_grad = False

    # 2. 对原始模型应用 LoRA
    if use_lora:
        from clip.utils.lora_util import apply_lora_attn_mlp  # 替换为你的 LoRA 实现路径
        clip_model = apply_lora_attn_mlp(clip_model, encoder_type='text', mlp=lora_mlp, attn=lora_attn)

    # 3. 保存原始 token embedding 权重
    old_weight = clip_model.token_embedding.weight.detach().clone()
    
    unfreeze_last_n_layers = z_config.get_diy_config().model.clip_unfreeze_last_n_layers
    if unfreeze_last_n_layers > 0:
        total_layers = len(clip_model.transformer.resblocks)
        assert unfreeze_last_n_layers <= total_layers, \
            f"unfreeze_last_n_layers ({unfreeze_last_n_layers}) > total layers ({total_layers})"
        
        for i in range(total_layers - unfreeze_last_n_layers, total_layers):
            block = clip_model.transformer.resblocks[i]
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfrozen last {unfreeze_last_n_layers} transformer layers.")

    # 4. 扩展 embedding 层
    old_vocab_size, embedding_dim = old_weight.shape
    new_vocab_size = old_vocab_size + num_new_tokens

    # 创建新的 embedding 层
    new_token_embedding = nn.Embedding(new_vocab_size, embedding_dim).to(dev)

    # 初始化新权重
    with torch.no_grad():
        new_token_embedding.weight[:old_vocab_size].copy_(old_weight)
        nn.init.normal_(new_token_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)

    # 设置整个 embedding 可训练（后续通过 mask 控制梯度）
    new_token_embedding.weight.requires_grad = True

    # 替换 embedding 层
    clip_model.token_embedding = new_token_embedding

    # 5. 创建 mask 并注册梯度钩子：只更新新增的 token
    mask = torch.zeros_like(clip_model.token_embedding.weight, device=dev)
    mask[old_vocab_size:] = 1.0  # 只允许最后 num_new_tokens 更新

    def selective_grad_hook(grad):
        return grad * mask

    clip_model.token_embedding.weight.register_hook(selective_grad_hook)

    # 6. 确保 token_projection 可训练（如果存在）
    if hasattr(clip_model, 'token_projection'):
        for param in clip_model.token_projection.parameters():
            param.requires_grad = True

    # 7. （可选）保存可训练参数名用于调试
    if save_dir is not None:
        from clip.utils.lora_util import save_trainable_parameter_names, count_parameters  # 替换为实际导入路径

        save_trainable_parameter_names(
            models_dict={
                "clip_model": clip_model,
                "clip_model_ori": clip_model_ori,
            },
            save_dir=save_dir,
            filename_prefix="trainable_params"
        )

        # 打印参数统计
        print("=== Trainable Parameters ===")
        count_parameters(clip_model)
        count_parameters(clip_model_ori)

    # 8. 返回处理后的模型
    return clip_model