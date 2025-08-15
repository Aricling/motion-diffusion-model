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

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns
    
def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    z_config.init_diy_config()
    cfg=z_config.get_diy_config()
    config_path = os.path.join(args.save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        cfg_dict = namespace_to_dict(cfg)
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    dist_util.setup_dist(args.device)

    print("initialize motionbert...")
    args_motionbert = get_config(args.motionbert_config)
    print(args_motionbert)

    model_backbone = load_backbone(args_motionbert)
    model_backbone = model_backbone.to(dist_util.dev())

    for param in model_backbone.parameters():
        param.requires_grad = False

    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: MotionBERT  parameter count:', model_params)
    print('Loading checkpoint', args.evaluate_motionbert)
    checkpoint = torch.load(args.evaluate_motionbert, map_location=lambda storage, loc: storage)   ## 这里目前默认是使用162M的模型
    model_backbone.load_state_dict(remove_module_prefix(checkpoint['model_pos']), strict=True)


    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              multi_view=args.multiview,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)   ## 和DIP没有什么关系，就是经典的MDM架构
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, model_backbone).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
