from typing import Tuple
import torch.nn as nn

from .quant import VectorQuantizer2
from .var import VAR
from .vqvae import VQVAE
from .var2 import VAR2
from .var3 import VAR3

def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,
    # VAR args
    depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VQVAE, VAR]:
    """
    3 stage model:
        * encode image
        * encode mask
        * attention on iamge and mask vectors
    returns -> vqvae_image, vqvae_mask, var
    """
    print("model1 => 3stage model, both attention")
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local_img = VQVAE(in_channels=3 ,vocab_size=Vimg, z_channels=Cvae, ch=160, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    vae_local_mask = VQVAE(in_channels=1 ,vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR(
        vae_local_mask = vae_local_mask, vae_local_img = vae_local_img,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local_img, vae_local_mask, var_wo_ddp

def build_vae_var2(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=256, Cvae=64, ch=160, share_quant_resi=4,
    # VAR args
    depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VAR2]:
    """
    2 stage model:
        * encode mask
        * extract image features and attention 
    returns -> vqvae_mask, var
    """
    print("model2 => 2stage model, both attentions, trainable encoding")
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(in_channels=1, vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR2(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp

def build_vae_var3(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=256, Vimg=1024, Cvae=64, ch=160, share_quant_resi=4,
    # VAR args
    depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VQVAE, VAR]:
    """
    3 stage model:
        * encode image
        * encode mask
        * attention on iamge and mask vectors
    returns -> vqvae_image, vqvae_mask, var
    """
    print("model3 => 3stage model, only cross attention, last level of image")
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local_img = VQVAE(in_channels=3 ,vocab_size=Vimg, z_channels=Cvae, ch=160, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    vae_local_mask = VQVAE(in_channels=1 ,vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR3(
        vae_local_mask = vae_local_mask, vae_local_img = vae_local_img,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local_img, vae_local_mask, var_wo_ddp