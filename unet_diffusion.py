import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):

        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
        # self.silu = nn.SiLU()

    def forward(self, x):
        # x.shape = (1, 320) --> (1, 1280)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # x = self.linear_2(self.silu(self.linear_1(x)))

        return x

class SwitchSequential(nn.Sequential):
    # used in the layers of UNET for ease of argument passing
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (B, C, H, W) --> (B, C, H*2, W*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)

    def forward(self, feature, time):
        # feature.shape: (B, C, H, W)
        # time.shape: (1, 1280)

        residue = feature # skip connect

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head*n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x.shape: (B, C, H, W)
        # context: (B, seq_len, d_context)

        long_residue = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        b, c, h, w = x.shape
        # (B, C, H, W) --> (B, C, H*W)
        x = x.view(b, c, h*w)
        # (B, C, H*W) --> (B, H*W, C)
        x = x.transpose(-1 ,-2)

        # Pre-Norm + SelfAttn + Short skip connection
        x = x + self.attention_1(self.layernorm_1(x))

        # Pre-Norm + CrossAttn + Short skip connect
        x = x + self.attention_2(self.layernorm_2(x), context)

        # Pre-Norm + FFN with GeGLU + Skip connect
        short_residue = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x* F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += short_residue
        # (B, H*W, C) --> (B, C, H*W)
        x = x.transpose(-1, -2)
        x = x.view((b,c,h,w))

        return self.conv_output(x) + long_residue

class UNET(nn.Module):
    def __init__(self, x, context, time):
        super().__init__()

        self.encoders = nn.Module([\
            # note: where there's no note about the shapes, there's no shape change from that layer,\
            # all shape changes are specified above the layer which changes the shape\
            # (B, 4, H/8, W/8) --> (B, 320, H/8, W/8)\
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),\
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),\
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),\
\
            # (B, 320, H/8, W/8) --> (B, 320, H/16, W/16)\
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),\
\
            # (B, 320, H/16, W/16) --> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)\
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),\
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),\
\
            # (B, 640, H/16, W/16) --> (B, 640, H/32, W/32)\
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),\
\
            # (B, 640, H/32, W/32) --> (B, 1280, H/32, W/32) --> (B, 1280, H/32, W/32)\
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),\
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),\
\
            # (B, 1280, H/32, W/32) --> (B, 1280, H/64, W/64)\
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),\
\
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),\
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),\
        ])

        self.bottleneck = SwitchSequential(
            # (B, 1280, H/64, W/64) --> (B, 1280, H/64, W/64)
            UNET_ResidualBlock(1280, 1280),
            # (B, 1280, H/64, W/64) --> (B, 1280, H/64, W/64)
            UNET_AttentionBlock(8, 160),
            # (B, 1280, H/64, W/64) --> (B, 1280, H/64, W/64)
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([\
            # (B, 2560, H/64, W/64) --> (B, 1280, H/64, W/64)\
            # the input from last layer goes from 1280 to 2560 because of adding the residual connection\
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),\
\
            # (B, 2560, H/64, W/64) --> (B, 1280, H/64, W/64)\
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),\
\
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64) -> (B, 1280, H/32, W/32)\
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),\
\
            # (B, 2560, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)\
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),\
\
            # (B, 2560, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)\
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),\
\
            # (B, 1920, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/16, W/16)\
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),\
\
            # (B, 1920, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)\
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),\
\
            # (B, 1280, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)\
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),\
\
            # (B, 960, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/8, W/8)\
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),\
\
            # (B, 960, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)\
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),\
\
            # (B, 640, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)\
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),\
\
            # (B, 640, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)\
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),\
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x.shape = (B, 320, H/8, W/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x # (B, 4, H/8, W/8)

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent.shape = (B, 4, H/8, W/8)
        # context.shape = (B, T, n_embd)
        # time.shape = (1, 320)

        time = self.time_embedding(time)

        # (B, 4, H/8, W/8) --> (B, 320, H/8, W/8)
        output = self.final(self.unet(latent, context, time))
        # (B, 320, H/8, W/8) --> (B, 4, H/8, W/8)
        # output = self.final(output)

        return output
