import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNCrossAttn, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR3(nn.Module):
    def __init__(
        self, vae_local_mask: VQVAE, vae_local_img: VQVAE,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local_mask.Cvae, vae_local_mask.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant_mask: VectorQuantizer2 = vae_local_mask.quantize
        self.vae_proxy_mask: Tuple[VQVAE] = (vae_local_mask,)
        self.vae_quant_proxy_mask: Tuple[VectorQuantizer2] = (quant_mask,)
        self.word_embed_mask = nn.Linear(self.Cvae, self.C)

        quant_img: VectorQuantizer2 = vae_local_img.quantize
        self.vae_proxy_img: Tuple[VQVAE] = (vae_local_img,)
        self.vae_quant_proxy_img: Tuple[VectorQuantizer2] = (quant_img,)
        self.word_embed_img = nn.Linear(self.Cvae, self.C)
        
        # 3. Static start token
        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 4. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 5. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.cross_blocks = nn.ModuleList([
            AdaLNCrossAttn(
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        # self.head_nm = AdaLNBeforeHead(self.C, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(h.float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, input_img: torch.Tensor, g_seed: Optional[int] = None, more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 1, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param input_img: image for feature fusion
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        B = input_img.shape[0]
        img_tokens = self.vae_proxy_img[0].img_to_idxBl(input_img)
        img_last = self.word_embed_img(self.vae_quant_proxy_img[0].embedding(img_tokens[-1]))

        next_token_map = self.pos_start.expand(B, self.first_l, -1) + self.pos_1LC[:, :self.first_l]
        cur_L = 0
        f_hat = torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=next_token_map.device)
        
        # for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            
            x = next_token_map
            # if(len(img_tokens[si][0]) <= 10):print("image:",img_tokens[si][0])
            
            AdaLNCrossAttn.forward
            for c in self.cross_blocks:
                x = c(x=x, f=img_last, attn_bias=None)
            logits_BlV = self.get_logits(x)
            
            idx_Bl = logits_BlV.argmax(dim=-1)
            # if(len(idx_Bl[0]) <= 10): print("mask:",idx_Bl[0])

            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy_mask[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy_mask[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy_mask[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed_mask(next_token_map) + self.pos_1LC[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
        
        # for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy_mask[0].fhat_to_img(f_hat)#.add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def autoregressive_forward(self, input_img: torch.Tensor,) -> torch.Tensor:
        B = input_img.shape[0]
        img_tokens = self.vae_proxy_img[0].img_to_idxBl(input_img)
        img_last = self.word_embed_img(self.vae_quant_proxy_img[0].embedding(img_tokens[-1]))

        next_token_map = self.pos_start.expand(B, self.first_l, -1) + self.pos_1LC[:, :self.first_l]
        cur_L = 0
        f_hat = torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=next_token_map.device)
        
        # for b in self.blocks: b.attn.kv_caching(True)
        logits = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            
            x = next_token_map
            
            # if(len(img_tokens[si][0]) <= 10):print("image:",img_tokens[si][0])
            AdaLNCrossAttn.forward
            for c in self.cross_blocks:
                x = c(x=x, f=img_last, attn_bias=None)
            logits_BlV = self.get_logits(x)
            logits.append(logits_BlV)

            idx_Bl = logits_BlV.argmax(dim=-1) # making prediction deterministic instead of one from top k
            # if(len(idx_Bl[0]) <= 10): print("mask:",idx_Bl[0])

            h_BChw = self.vae_quant_proxy_mask[0].embedding(idx_Bl)   # B, l, Cvae
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy_mask[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed_mask(next_token_map) + self.pos_1LC[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
        
        # for b in self.blocks: b.attn.kv_caching(False)
        # return self.vae_proxy_mask[0].fhat_to_img(f_hat)
        return torch.concat(logits, dim=1)
    
    def forward(self, x_BLCv_wo_first_l: torch.Tensor, input_img: torch.Tensor, teacher_force: bool = True) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        if not teacher_force:
            return self.autoregressive_forward(input_img=input_img)
        
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        # with torch.amp.autocast('cuda',enabled=False):
        sos = self.pos_start.expand(B, self.first_l, -1)

        if self.prog_si == 0: x_BLC = sos
        else: x_BLC = torch.cat((sos, self.word_embed_mask(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        img_tokens = self.vae_proxy_img[0].img_to_idxBl(input_img)[-1]
        img_tokens = self.vae_quant_proxy_img[0].embedding(img_tokens)
        img_tokens = self.word_embed_img(img_tokens)
        
        # AdaLNSelfAttn.forward
        AdaLNCrossAttn.forward
        for c in self.cross_blocks:
            x_BLC = c(x=x_BLC, f=img_tokens, attn_bias=None) #cross attention does not need bias
        x_BLC = self.get_logits(x_BLC.float())
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        # if isinstance(self.head_nm, AdaLNBeforeHead):
        #     self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
        #     if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
        #         self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # depth = len(self.blocks)
        # for block_idx, sab in enumerate(self.blocks):
        #     sab: AdaLNCrossAttn
        #     sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
        #     sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
        #     if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
        #         nn.init.ones_(sab.ffn.fcg.bias)
        #         nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
        #     if hasattr(sab, 'ada_lin'):
        #         sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
        #         sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
        #         if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
        #             sab.ada_lin[-1].bias.data.zero_()
        #     elif hasattr(sab, 'ada_gss'):
        #         sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
        #         sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR3, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
