# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from utils.pth_util import remove_module_prefix
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from MotionBERT.lib.utils.tools import *
from MotionBERT.lib.utils.learning import *
import z_config
from types import SimpleNamespace
from vae_all.options.denoiser_option import arg_parse   ## Salad也是用的denoiser_option来初始化VAE的
from os.path import join as pjoin
from utils.get_opt_vae import get_opt
from vae_all.vae.model import VAE

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns
    
def load_and_freeze_vae(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name_vae, opt.vae_name, 'opt.txt')
    vae_opt = get_opt(opt_path, opt.device)

    model = VAE(vae_opt)
    # ckpt = torch.load(pjoin('vae_all', vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            # map_location='cpu')
    ckpt = torch.load(pjoin('vae_all', vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_mse.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.to(dist_util.dev())
    print(f'Loading VAE Model {opt.vae_name}')
    return model
    
def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    ## 存储args
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    ## 存储我的z_config
    z_config.init_diy_config()
    cfg=z_config.get_diy_config()
    config_path = os.path.join(args.save_dir, 'config.yaml')
    z_config.save_config(config_path, cfg, cfg.overwrite_config)

    dist_util.setup_dist(args.device)

    ## 初始化VAE
    vae_opt = arg_parse(True)
    vae_opt.vae_name = cfg.vae_model.vae_name
    vae = load_and_freeze_vae(vae_opt)


    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              multi_view=args.multiview,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, vae).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
