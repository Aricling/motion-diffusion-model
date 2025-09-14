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
        x = self.motion_enc(x)  ## 它的职责其实只是把每一个关节点变成到一样的维度 [bs, T=64, 17, 32]
        x = self.conv_enc(x)    ## 真正的图卷积，时间卷积和降维其实都是这里做的 [bs, T/4=16, 7, 32]

        # latent space
        x = self.dist(x)    ## 为什么需要这一层，就是为了将其映射到latent space中
        if z_config.get_diy_config().training.use_encoder_output_wo_reparam:
            z=x.chunk(2, dim=-1)[0]
            loss_kl =0
            return z, {"loss_kl": loss_kl}
        
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)
        
        return z, {"loss_kl": loss_kl}
    
    def decode(self, x):
        x = self.conv_dec(x)    ## len也是2, x.shape=[32,196,22,32],!!好像训练和测试不一样，训练是64
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