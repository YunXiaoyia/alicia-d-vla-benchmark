"""Standalone DINOv3-ViT fallback for environments with transformers < 4.56 (e.g., lerobot conda env)."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers.activations import ACT2FN


def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device) / num_patches_h
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device) / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    return 2.0 * coords - 1.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)
    return q, k


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)


class DINOv3ViTRopePositionEmbedding(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size
        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )
            angles = 2.0 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DINOv3ViTAttention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = float(getattr(config, "attention_dropout", 0.0))

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, patches, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTLayer(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual
        return hidden_states


class DINOv3ViTEncoder(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        return SimpleNamespace(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class StandaloneDINOv3ViTModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.model = DINOv3ViTEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "StandaloneDINOv3ViTModel":
        cfg_path = os.path.join(model_name_or_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = SimpleNamespace(**cfg_dict)
        model = cls(config)
        st_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.exists(st_path):
            raw_sd = load_file(st_path, device="cpu")
            state_dict = {
                (f"model.{k}" if k.startswith("layer.") else k): v
                for k, v in raw_sd.items()
            }
            model.load_state_dict(state_dict, strict=True)
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> SimpleNamespace:
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)
        output = self.model(hidden_states, position_embeddings, output_hidden_states=output_hidden_states)
        sequence_output = self.norm(output.last_hidden_state)
        pooled_output = sequence_output[:, 0, :]
        return SimpleNamespace(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=output.hidden_states,
            attentions=None,
        )
