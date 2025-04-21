import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead', 'AdaLNCrossAttn', 'CrossAttention']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError: pass
def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm

        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    def forward(self, x, attn_bias):
        B, L, C = x.shape
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1   # q or k or v: BLHc
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2   # q or k or v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim,
        norm_layer, num_heads, mlp_ratio=4.,
        drop=0., attn_drop=0., drop_path=0.,
        attn_l2_norm=False, flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim,
            num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=drop, attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available
        )
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            drop=drop, fused_if_available=fused_if_available
        )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_add_norm_fn = None

    def forward(self, x, attn_bias):
        x = x + self.drop_path(self.attn(self.ln_wo_grad(x), attn_bias=attn_bias))
        x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))
        return x

class AdaLNBeforeHead(nn.Module):
    def __init__(self, embed_dim, norm_layer):  
        super().__init__()
        self.C = embed_dim
        self.ln = norm_layer(embed_dim, elementwise_affine=True)  # Keep learnable affine parameters

    def forward(self, x_BLC: torch.Tensor):
        return self.ln(x_BLC)
    
class CrossAttention(nn.Module):
    def __init__(
        self, block_idx, dim=1024, num_heads=16, proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super(CrossAttention, self).__init__()

        self.attn_l2_norm = attn_l2_norm
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, dim // num_heads
        self.scale = 0.25 / math.sqrt(self.head_dim)
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()

        self.mat_q = nn.Linear(dim, dim, bias=False)
        self.mat_kv = nn.Linear(dim, dim * 2, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
        self.register_buffer('zero_k_bias', torch.zeros(dim))

    def forward(self, f_d, f_var, attn_bias=None):
        B, Lq, C = f_d.shape
        Lkv = f_var.shape[1]

        f_q = F.linear(input=f_d, weight=self.mat_q.weight, bias=self.q_bias).view(B, Lq, 1, self.num_heads, self.head_dim)
        f_kv = F.linear(input=f_var, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(B, Lkv, 2, self.num_heads, self.head_dim)
        main_type = f_kv.dtype

        q = f_q.permute(2, 0, 3, 1, 4)          # [1, B, num_heads, Lq, head_dim]
        k, v = f_kv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # [B, num_heads, Lkv, head_dim]
        
        dropout_p = 0.0
        using_flash = self.using_flash and attn_bias is None and f_kv.dtype != torch.float32

        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                dropout_p=dropout_p, softmax_scale=self.scale
            ).view(B, Lq, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, Lq, Lkv),
                p=dropout_p, scale=self.scale
            ).view(B, Lq, C)
        else:
            oup = slow_attn(
                query=q, key=k, value=v,
                scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p
            ).transpose(1, 2).reshape(B, Lq, C)

        return self.proj_drop(self.proj(oup))


    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'
    
class AdaLNCrossAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = CrossAttention(
            block_idx=block_idx, dim=embed_dim, num_heads=num_heads,
            proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.ffn = FFN(
            in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),
            drop=drop, fused_if_available=fused_if_available
        )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)

        self.fused_add_norm_fn = None

    def forward(self, x, f, attn_bias=None):
        # Cross-Attention: q from x, k/v from f
        x = x + self.drop_path(
            self.attn(self.ln_wo_grad(x), self.ln_wo_grad(f), attn_bias=attn_bias)
        )

        # Feed-Forward Network
        x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))

        return x