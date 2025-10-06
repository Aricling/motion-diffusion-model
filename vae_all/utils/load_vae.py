import torch
from utils import dist_util

from os.path import join as pjoin
from vae_all.utils.get_opt_vae import get_opt
from vae_all.vae.model import VAE

def load_and_freeze_vae(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name_vae, opt.vae_name, 'opt.txt')
    vae_opt = get_opt(opt_path, opt.device)

    model = VAE(vae_opt)
    # ckpt = torch.load(pjoin('vae_all', vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            # map_location='cpu')
    ckpt = torch.load(pjoin('vae_all', vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.to(dist_util.dev())
    print(f'Loading VAE Model {opt.vae_name}')
    return model