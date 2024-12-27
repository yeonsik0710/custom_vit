# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


class CustomAttention(torch.nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn

    def forward(self, x):
        # 기존 Attention 연산 (Q, K, V 계산)
        qkv = self.original_attn.qkv(x).reshape(
            x.shape[0], x.shape[1], 3, self.original_attn.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Grouping 작업 (cls token 제외)
        num_patches = k.shape[2] - 1  # cls token 제외한 토큰 수 (196-1=195)
        group_size = 49
        num_groups = num_patches // group_size  # 49개씩 4그룹

        final_attn = []
        for batch_idx in range(q.shape[0]):
            query = q[batch_idx]  # [num_heads, num_patches, embed_dim]
            key = k[batch_idx]    # [num_heads, num_patches, embed_dim]
            current_attn = []

            for token_idx in range(query.shape[1]):
                current_query = query[:, token_idx]  # [num_heads, embed_dim]

                group_scores = []
                # torch.no_grad() 사용하여 불필요한 그래디언트 추적 방지
                with torch.no_grad():
                    for group_idx in range(num_groups):
                        group_start = group_idx * group_size + 1
                        group_end = group_start + group_size

                        group_k = key[:, group_start:group_end]  # [num_heads, group_size, embed_dim]
                        group_max = torch.amax(group_k, dim=1)  # [num_heads, embed_dim]
                        group_min = torch.amin(group_k, dim=1)  # [num_heads, embed_dim]

                        elementwise_max = torch.max(current_query * group_max, current_query * group_min)
                        group_score = elementwise_max.sum(dim=-1)  # [num_heads]
                        group_scores.append(group_score)

                group_scores = torch.stack(group_scores, dim=0)  # [num_groups, num_heads]
                top_k = int(0.5 * num_groups)
                _, top_group_idx = torch.topk(group_scores, top_k, dim=0, largest=True)

                selected_keys_mask = torch.zeros_like(key)  # [num_heads, num_patches, embed_dim]
                for head_idx in range(top_group_idx.shape[1]):
                    for group_idx in top_group_idx[:, head_idx]:
                        group_start = int(group_idx.item() * group_size + 1)
                        group_end = int(group_start + group_size)
                        selected_keys_mask[head_idx, group_start:group_end, :] = 1

                selected_keys_mask[:, 0, :] = 1  # CLS token always selected
                if token_idx == 0:
                    selected_keys_mask[:, :, :] = 1

                selected_key = key * selected_keys_mask  # Select key

                # Attention Score 계산
                attn = (current_query.unsqueeze(1) @ selected_key.transpose(-2, -1)) * self.original_attn.scale
                attn = attn.softmax(dim=-1)  # Softmax in-place
                attn = self.original_attn.attn_drop(attn)

                current_attn.append(attn)

            current_attn = torch.stack(current_attn, dim=1)  # [num_heads, num_patches, num_selected_keys]
            final_attn.append(current_attn)

        final_attn = torch.stack(final_attn, dim=0).squeeze(3)  # [batch_size, num_heads, num_patches, num_selected_keys]

        # Value 연산
        x = (final_attn @ v).transpose(2, 3).contiguous()  
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)
        x = self.original_attn.proj(x)
        x = self.original_attn.proj_drop(x)
        return x



@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
