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

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        # raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
        pass
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("initialize motionbert...")
    args_motionbert = get_config(args.motionbert_config)
    print(args_motionbert)

    model_backbone = load_backbone(args_motionbert)

    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: MotionBERT Trainable parameter count:', model_params)

    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model_backbone)    ## 就是因为这一步会在每一个模块前面加上module
    #     model_backbone = model_backbone.cuda()

    print('Loading checkpoint', args.evaluate_motionbert)
    checkpoint = torch.load(args.evaluate_motionbert, map_location=lambda storage, loc: storage)   ## 这里目前默认是使用162M的模型
    model_backbone.load_state_dict(remove_module_prefix(checkpoint['model_pos']), strict=True)


    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
