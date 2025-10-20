import torch
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from data_loaders.humanml_utils import HML_EE_JOINT_NAMES
import importlib
import z_config
from utils import dist_util

def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model') or 'sequence_pos_encoder' or k.startswith('lra_') in k for k in missing_keys])
    return missing_keys


def create_model_and_diffusion(args, data):
    try:
        model_module_name = z_config.get_diy_config().model.MDM_model_file
    except (AttributeError, KeyError):
        model_module_name = None
    # 如果配置不存在或为空，使用默认模块
    if not model_module_name:
        model_module_name = 'mdm'  # 默认使用 model.mdm
    module_path = f"model.{model_module_name}"
    try:
        # 动态导入模块
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"无法导入模型模块 '{module_path}'. "
                          f"请检查是否存在 model/{model_module_name}.py 文件。错误: {e}")
    MDM=getattr(module, 'MDM')
    model = MDM(**get_model_args(args, data))

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    all_goal_joint_names = []

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0
    
    emb_policy = args.__dict__.get('emb_policy', 'add')
    multi_target_cond = args.__dict__.get('multi_target_cond', False)
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi')
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            }



def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model

def load_model_with_lora(model, ckpt_path, use_ema=False):
    """
    加载模型权重和对应的 LoRA 权重到指定的 model。
    
    Args:
        model: 要加载权重的目标模型
        ckpt_path (str): 主模型 checkpoint 路径 (e.g., ".../modelXXXX.pt")
        use_ema (bool): 是否加载 EMA 权重（默认 False）

    Returns:
        None
    """
    import os

    # 加载 state dict
    state_dict = dist_util.load_state_dict(ckpt_path, map_location=dist_util.dev())

    # 推导 lora 路径
    lora_path = 'lora'.join(ckpt_path.rsplit('model', 1))
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"对应的 LoRA 权重文件不存在: {lora_path}")
    lora_dict = dist_util.load_state_dict(lora_path, map_location=dist_util.dev())

    if "model_avg" in state_dict:
        print("Loading both model and model_avg ...")

        # 拆分主模型和 avg
        state_dict_main, state_dict_avg = state_dict["model"], state_dict["model_avg"]
        lora_dict_main, lora_dict_avg = lora_dict["lora"], lora_dict["lora_avg"]

        # 加载主模型权重
        missing_keys_0 = load_model_wo_clip(model, state_dict_main)
        missing_keys_lora_0, unexpected_keys_lora_0 = model.load_state_dict(lora_dict_main, strict=False)
        assert len(unexpected_keys_lora_0) == 0

        # 如果需要的话加载 avg 权重（覆盖 model 当前参数）
        if use_ema:
            print("Replacing model with EMA weights ...")
            missing_keys_1 = load_model_wo_clip(model, state_dict_avg)
            missing_keys_lora_1, unexpected_keys_lora_1 = model.load_state_dict(lora_dict_avg, strict=False)
            assert len(unexpected_keys_lora_1) == 0

    else:
        print("Loading model (no model_avg found in checkpoint) ...")
        missing_keys = load_model_wo_clip(model, state_dict)

        # LoRA 权重
        missing_keys_lora, unexpected_keys_lora = model.load_state_dict(lora_dict, strict=False)
        assert len(unexpected_keys_lora) == 0

        # 如果需要 EMA，但 checkpoint 没有，直接复制一份普通模型参数
        if use_ema:
            print("Using model parameters as EMA (legacy checkpoint).")