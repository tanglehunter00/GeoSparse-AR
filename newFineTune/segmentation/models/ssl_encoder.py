"""
与 newFullPretrain/newNewModel.py 在 GeoPriorGen3DOn=1、HybridSparseAttnOn=1 时同构的 ViT 编码器。
分割阶段仅改输出头：向 UNETR 提供 patch token（去掉 BOS/EOS），注意力图与预训练一致。
"""
import torch
import warnings

warnings.filterwarnings("ignore")

from timm.models.layers import trunc_normal_
from monai.networks.blocks.mlp import MLPBlock
from typing import Optional, Tuple, Union
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

# 与 full pretrain（GASP）一致：必须开启
GeoPriorGen3DOn = 1
HybridSparseAttnOn = 1

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class HybridSparseMaskGen3D(nn.Module):
    def __init__(self, window_size=2, max_pow=3):
        super().__init__()
        self.window_size = window_size
        self.max_pow = max_pow

    def forward(self, t: int, h: int, w: int, device, dtype):
        N = t * h * w
        idx_t = torch.arange(t, device=device)
        idx_h = torch.arange(h, device=device)
        idx_w = torch.arange(w, device=device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing="ij")
        grid = torch.stack(grid, dim=-1).reshape(N, 3)

        abs_diff = (grid[:, None, :] - grid[None, :, :]).abs()
        mask_local = (abs_diff <= self.window_size).all(dim=-1)

        mask_sparse = torch.zeros((N, N), dtype=torch.bool, device=device)
        for p in range(self.max_pow):
            offset = 2**p
            for d in range(3):
                match = abs_diff[..., d] == offset
                for od in range(3):
                    if od != d:
                        match &= abs_diff[..., od] == 0
                mask_sparse |= match

        mask_combined = mask_local | mask_sparse
        res_mask = torch.zeros((N, N), dtype=dtype, device=device)
        res_mask.masked_fill_(~mask_combined, torch.finfo(dtype).min)
        return res_mask


class GeoPriorGen3D(nn.Module):
    def __init__(self, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

    def forward(self, t: int, h: int, w: int):
        idx_t = torch.arange(t).to(self.decay.device)
        idx_h = torch.arange(h).to(self.decay.device)
        idx_w = torch.arange(w).to(self.decay.device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing="ij")
        grid = torch.stack(grid, dim=-1).reshape(t * h * w, 3)
        dist = grid[:, None, :] - grid[None, :, :]
        dist = dist.abs().sum(dim=-1)
        return dist.unsqueeze(0) * self.decay[:, None, None]


class SinCosPosEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: int, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 3 == 0, embed_dim
        grid_t = torch.arange(t).float()
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_t, grid_h, grid_w)
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([3, 1, t, h, w])
        emb_t = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
        return torch.concatenate([emb_t, emb_h, emb_w], dim=1)

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.concatenate([emb_sin, emb_cos], dim=1)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 64,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = (img_size, img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                if attention_mask.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)} or "
                        f"{(bsz, self.num_heads, q_len, kv_seq_len)}, got {attention_mask.size()}"
                    )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLPBlock(config.hidden_size, config.intermediate_size, 0.0)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class PretrainMatchedEncoder(nn.Module):
    """与 ReconModel.model（GASP full pretrain）同构；forward 与 newNewModel.BaseModel 一致。"""

    def __init__(self, config):
        super().__init__()
        self.pos_type = config.pos_type
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_tokens = nn.Embedding(4, config.hidden_size)
        self.patchifier = PatchEmbed(
            embed_dim=config.hidden_size,
            img_size=(self.img_size[0], self.img_size[1], self.img_size[2]),
            patch_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]),
        )
        if self.pos_type == "sincos3d":
            self.pos_embed = SinCosPosEmbed()
        elif self.pos_type == "learnable":
            ntok = (
                self.img_size[0]
                * self.img_size[1]
                * self.img_size[2]
                // self.patch_size[0]
                // self.patch_size[1]
                // self.patch_size[2]
                + 2
            )
            self.pos_embed_learn = nn.Parameter(torch.zeros(ntok, config.hidden_size))
            trunc_normal_(self.pos_embed_learn, std=0.02)

        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.geo_prior_gen = GeoPriorGen3D(config.num_attention_heads)
        self.hybrid_sparse_gen = HybridSparseMaskGen3D(window_size=2, max_pow=3)
        self.init_proj()
        self.apply(self._init_weights)

    def init_proj(self):
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_pretrain(self, input_image: torch.Tensor):
        """与 newNewModel.BaseModel.forward 相同的嵌入与注意力构造。"""
        batch_size = input_image.shape[0]
        input_image = input_image.reshape(batch_size, 1, self.img_size[0], self.img_size[1], self.img_size[2])
        image_embeds = self.patchifier(input_image)
        t, h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        seq_length = t * h * w + 2

        input_ids = torch.empty(batch_size, seq_length, dtype=torch.int64, device=input_image.device)
        input_ids[:, 0] = BOS_TOKEN_ID
        input_ids[:, -1] = EOS_TOKEN_ID

        starts_embeds = self.embed_tokens(input_ids[..., :1])
        ends_embeds = self.embed_tokens(input_ids[..., -1:])
        image_embeds = torch.cat((starts_embeds, image_embeds, ends_embeds), 1)

        if self.pos_type == "sincos3d":
            pos_embed = self.pos_embed(t, h, w, embed_dim)
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
            pos_embed = torch.concatenate([pos_embed, torch.zeros([1, embed_dim])], dim=0)
        else:
            pos_embed = self.pos_embed_learn
        pos_embed = pos_embed.to(image_embeds.device)
        inputs_embeds = image_embeds + pos_embed[None, ...]

        attention_mask = torch.zeros(
            batch_size, 1, seq_length, seq_length, dtype=inputs_embeds.dtype, device=input_image.device
        )
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask, (batch_size, seq_length), inputs_embeds, 0
        )

        sparse_mask = self.hybrid_sparse_gen(t, h, w, attention_mask.device, inputs_embeds.dtype)
        full_sparse_mask = torch.zeros((seq_length, seq_length), dtype=inputs_embeds.dtype, device=attention_mask.device)
        full_sparse_mask[1:-1, 1:-1] = sparse_mask
        attention_mask = attention_mask + full_sparse_mask.unsqueeze(0).unsqueeze(0)

        geo_bias = self.geo_prior_gen(t, h, w)
        full_geo_bias = torch.zeros(
            (geo_bias.shape[0], seq_length, seq_length), dtype=inputs_embeds.dtype, device=attention_mask.device
        )
        full_geo_bias[:, 1:-1, 1:-1] = geo_bias
        attention_mask = attention_mask + full_geo_bias.unsqueeze(0)

        hidden_states = inputs_embeds
        all_hidden_states = ()
        for decoder_layer in self.layers:
            all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = self.norm(hidden_states)
        all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states

    def forward(self, input_image: torch.Tensor):
        """
        UNETR 接口：返回 patch 特征与 skip（与 baseline 下游索引 3/6/9 对齐）。
        patch 对应预训练重建监督的 shift_logits[..., :-2, :] 区间。
        """
        hidden_states, all_hidden_states = self._encode_pretrain(input_image)
        patch_tokens = hidden_states[:, 1:-1, :]
        skips = [x[:, 1:-1, :] for x in all_hidden_states[1:]]
        return patch_tokens, skips
