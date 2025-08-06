import loratorch as lora
import os
import torch.nn as nn
from utils import dist_util
import copy
from z_config import cfg

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

def load_clip_w_lora(model, state_dict, lora_dict):
    def remove_prefix_from_state_dict(state_dict, prefix, filter_prefix=None):
        new_state_dict = {}
        for k, v in state_dict.items():
            if filter_prefix and k.startswith(filter_prefix):
                continue  # skip filtered keys
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
        # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
        # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    if state_dict.get('sequence_pos_encoder.pe', None):
        del state_dict['equence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    if state_dict.get('embed_timestep.sequence_pos_encoder.pe', None):
        del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models

    state_dict = remove_prefix_from_state_dict(state_dict, "clip_model.", filter_prefix = "clip_model_ori.")
    lora_dict = remove_prefix_from_state_dict(lora_dict, "clip_model.")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys_lora, unexpected_keys_lora = model.load_state_dict(lora_dict, strict=False)
    assert len(unexpected_keys) == 0 and len(unexpected_keys_lora) == 0
    assert all("lora" or "text_projection" in k for k in missing_keys), f"Error: Not all missing keys are LoRA-related. Found keys without 'lora': { [k for k in missing_keys if 'lora' not in k] }"
    assert all("lora" not in k for k in missing_keys_lora), f"Missing keys contain LoRA layers: { [k for k in missing_keys_lora if 'lora' in k] }"

def init_finetuned_clip_and_freeze(clip_model, clip_model_path=cfg.model.clip_model_path):

    old_weight = clip_model.token_embedding.weight
    new_vocab_size = old_weight.shape[0] + 3  # 假设新增3个token
    embedding_dim = old_weight.shape[1]

    clip_model.token_embedding = nn.Embedding(new_vocab_size, embedding_dim)

    state_dict = dist_util.load_state_dict(
        clip_model_path, map_location=dist_util.dev())
    resume_lora_checkpoint = clip_model_path.replace('model', 'lora')
    lora_dict = dist_util.load_state_dict(
        resume_lora_checkpoint, map_location=dist_util.dev()
    )

    if 'model_avg' in state_dict:
        clip_model_avg = copy.deepcopy(clip_model)
        print('loading both model and model_avg')
        state_dict, state_dict_avg = state_dict['model'], state_dict[
            'model_avg']
        lora_dict, lora_dict_avg = lora_dict['lora'], lora_dict['lora_avg']
        load_clip_w_lora(clip_model, state_dict, lora_dict)
        load_clip_w_lora(clip_model_avg, state_dict_avg, lora_dict_avg)
        for p in clip_model.parameters():
            p.requires_grad = False
        for p in clip_model_avg.parameters():
            p.requires_grad = False
        return clip_model, clip_model_avg
    else:
        load_clip_w_lora(clip_model, state_dict)
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model