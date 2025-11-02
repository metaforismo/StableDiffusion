from inspect import getargs
import torch
from torch import nn
from torch.nn import Conv2d, functional as F
from vae_decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(B, C, H, W) --> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            #(B, 128, H, W) --> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) --> (B, 128, H/2, W/2), not exactly halved, but somewhat
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (B, 128, H/2, W/2) --> (B, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (B, 256, H/2, W/2) --> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            # (B, 256, H/2, W/2) --> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (B, 256, H/4, W/4) --> (B, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (B, 512, H/4, W/4) --> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/4, W/4) --> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (B, 512, H/8, W/8) --> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/8, W/8) --> (B, 512, H/8, W/8)
            VAE_AttentionBlock(512),
            # (B, 512, H/8, W/8) --> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/8, W/8) --> (B, 512, H/8, W/8)
            nn.GroupNorm(32, 512),
            # (B, 512, H/8, W/8) --> (B, 512, H/8, W/8)
            nn.SiLU(),
            # (B, 512, H/8, W/8) --> (B, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (B, 8, H/4, W/4) --> (B, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x.shape = (B, C, H, W)
        # noise.shape = (B, Channels_out, H/8, W/8)
        for layer in self:
            if getattr(layer, 'stride', None) == (2,2):
                # apply asymmetric padding: only on right and bottom of the image
                x = F.pad(x, (0,1,0,1))
            x = layer(x)
        # (B, 8, H/8, W/8) --> divide into 2 along dim=1 of shape (B, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20) # clamp log variance between a workable range, no shape change
        var = log_var.exp() # exponentiate log_var to get var, no shape change
        std = var.sqrt() # get standard deviation from variance, no shape change

        # because we learn mean and variance of latent space:
        # sampling from N(0,1) and convert to N(mean, std) (make life easier, haha!)
        # transform X=N(0,1) to Y=N(mean, std) -> Y=aX+b, a(0) + b = mean, a^2*(1)^2=std^2
        # => Y = std*X + mean
        z = std*noise + mean
        z *= 0.18215 # some scaling "divine benevolence" ig :)

        return z
