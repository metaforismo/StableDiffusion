from this import d
import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed)
        self.n_heads = n_heads # number of heads
        self.d_head = d_embed // n_heads # head_dimension

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x.shape = (B, T, C)
        input_shape = x.shape
        # here B = batch_sie, T = seq_len, C = d_embed
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        # (B, T, C) --> (B, T, C*3) --> 3 tensors x (B, T, C)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # q = Q*W_Q, also for K and V
        # (B, T, C) --> (B, T, n_heads, d_head) --> (B, n_heads, T, d_head)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) = (B, n_heads, T, T)
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.d_head))

        if causal_mask:
            mask = torch.ones_like(att, dtype=torch.bool).triu(1)
            att.masked_fill_(mask, -torch.inf)

        att = F.softmax(att, dim=-1)
        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) --> (B, n_heads, T, d_head)
        y = att @ v
        # (B, n_heads, T, d_head) --> (B, T, n_heads, d_head) --> (B, T, n_heads*d_head) = (B, T, C)
        y = y.transpose(1, 2).contiguous().view(input_shape)
        # wo matrix to mix info. across heads after attention
        output = self.out_proj(y)
        # (B, T, C), same as input
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # defining the matrices as three separate here, unlike SelfAttn
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x, y):
        # x (latent): (B, T_Q, C_Q)
        # y (prompt/context): (B, T_KV, C_KV) = (B, 77, 768)
        input_shape = x.shape
        b, seq_len, d_embed = input_shape
        interim_shape = (b, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.d_head))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(input_shape)

        out = self.out_proj(out)

        return out
