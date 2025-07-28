import loratorch as lora
import os

def apply_lora_attn_mlp(model, encoder_type="text", rank=16, lora_alpha=32, mlp=True, attn=True):
    if encoder_type == 'visual':
        encoder = model.visual.transformer
    elif encoder_type == 'text':
        encoder = model.transformer
    else:
        raise ValueError("Invalid encoder_type. Choose 'visual' or 'text'.")

    enable_lora=['q', 'k', 'v', 'o']
    for i, resblock in enumerate(encoder.resblocks):
        if hasattr(resblock, 'attn') and attn:
            multihead = resblock.attn
            lora_multihead = lora.MultiheadAttention(r=rank,
                                    lora_alpha=lora_alpha,
                                    enable_lora=enable_lora,
                                    embed_dim=multihead.embed_dim,
                                    num_heads=multihead.num_heads,
                                    dropout=multihead.dropout,
                                    bias=True if hasattr(multihead, "in_proj_bias") else False,
                                    add_bias_kv=False if multihead.bias_k==None else True,
                                    add_zero_attn=multihead.add_zero_attn,
                                    kdim=multihead.kdim,
                                    vdim=multihead.vdim,
                                    batch_first=multihead.batch_first)
            missing_keys, unexpected_keys = lora_multihead.load_state_dict(multihead.state_dict(), strict=False)
            resblock.attn = lora_multihead

        if hasattr(resblock, 'mlp') and mlp:
            old_mlp_fc=resblock.mlp.c_fc
            old_mlp_proj=resblock.mlp.c_proj
            new_mlp_fc = lora.Linear(
                old_mlp_fc.in_features,
                old_mlp_fc.out_features,
                bias=True if hasattr(old_mlp_fc, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            new_mlp_proj = lora.Linear(
                old_mlp_proj.in_features,
                old_mlp_proj.out_features,
                bias=True if hasattr(old_mlp_proj, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            c, d = new_mlp_fc.load_state_dict(old_mlp_fc.state_dict(),strict=False)
            e,f = new_mlp_proj.load_state_dict(old_mlp_proj.state_dict(),strict=False)
            resblock.mlp.c_fc = new_mlp_fc
            resblock.mlp.c_proj = new_mlp_proj

    lora.mark_only_lora_as_trainable(model)
    return model

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision_params = sum(p.numel() for p in model.visual.transformer.parameters() if p.requires_grad)
    text_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    embed_params = sum(p.numel() for p in model.token_embedding.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters - Full model: {trainable_params:,}")
    print(f"Trainable parameters - Vision: {vision_params:,}")
    print(f"Trainable parameters - Text: {text_params:,}")
    print(f"Trainable parameters - embedding: {embed_params:,}")

def save_trainable_parameter_names(models_dict, save_dir, filename_prefix="trainable_params"):
    """
    收集多个模型中所有 requires_grad=True 的参数名称，并保存到 save_dir 下的 .txt 和 .json 文件中。

    Args:
        models_dict (dict): 模型名字到模型对象的字典，例如：
                            {
                                "clip_model": self.clip_model,
                                "clip_model_ori": self.clip_model_ori
                            }
        save_dir (str): 保存文件的目录路径
        filename_prefix (str, optional): 保存文件的前缀，默认是 "trainable_params"
    """
    os.makedirs(save_dir, exist_ok=True)

    all_trainable_names = {}

    for model_name, model in models_dict.items():
        trainable_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
        all_trainable_names[model_name] = trainable_names

    # ===== 保存为 txt 文件 =====
    txt_filename = f"{filename_prefix}.txt"
    txt_path = os.path.join(save_dir, txt_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        for model_name, names in all_trainable_names.items():
            f.write(f"===== {model_name} (可训练参数) =====\n")
            for name in names:
                f.write(f"{name}\n")
            f.write("\n")  # 模型之间空一行

    print(f"[INFO] 可训练参数(txt)已保存到: {txt_path}")