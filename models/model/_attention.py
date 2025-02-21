import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight

class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1]**0.5)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, use_flash_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.use_flash_attn = use_flash_attn

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

        self.norm = RMSNorm(dim)

    def forward(self, x):
        b, c, h, w,d = x.shape

        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c ) x y z-> b h c (x y z)', h=self.heads),
            qkv)

        if not self.use_flash_attn:
            # print(q.shape)
            q = q * self.scale
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            # print(sim.shape)
            try:
                attn = torch.Logsoftmax(sim,dim=-1)
                # print('normal soft')
                # attn = attn.softmax(dim=-1)#backward error,
            except:
                attn = sim
            # attn = torch.nan_to_num(attn)
            if 0 in attn:
              print(attn.shape)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x=h, y=w)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
            out = rearrange(out, 'b h d (x y z) -> b (h d) x y z', x=h, y=w)

        return self.to_out(out) + x
