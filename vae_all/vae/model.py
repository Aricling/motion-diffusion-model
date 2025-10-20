import torch
import torch.nn as nn

from vae_all.skeleton.linear import MultiLinear
from vae_all.vae.encdec import MotionEncoder, MotionDecoder, STConvEncoder, STConvDecoder

import z_config

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        
        self.opt = opt

        # motion encoder and decoder
        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)

        # skeleto-temporal convolutional encoder and decoder
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, 7)
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.motion_enc(x)
        x = self.conv_enc(x)

        # latent space
        x = self.dist(x)
        if z_config.get_diy_config().debug:
            mu, _ = x.chunk(2, dim=-1)
            # 👇 加载你之前保存的 logvar，并移动到 mu 的 device 上
            loaded_logvar = torch.load('/home/mengqing/usr/motion-diffusion-model/z_utils/saved_logvar.pt')         
            loaded_logvar = loaded_logvar.to(mu.device)          
            loaded_logvar = loaded_logvar.to(mu.dtype)       
            logvar = loaded_logvar
        else:
            mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        
        return z, {"loss_kl": loss_kl}
    
    def decode(self, x):
        x = self.conv_dec(x)
        x = self.motion_dec(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """
        
        # encode
        x = x.detach().float()
        z, loss_dict = self.encode(x)
        out = self.decode(z)

        return out, loss_dict