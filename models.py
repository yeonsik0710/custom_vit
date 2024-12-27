# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
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

# CustomAttention 클래스
class CustomAttention(nn.Module):
    def __init__(self, original_attn, window_size=3, stride=1):
        super().__init__()
        self.original_attn = original_attn
        self.window_size = window_size  # 윈도우 크기
        self.stride = stride  # 윈도우 이동 간격
        

    def forward(self, x):
        # Q, K, V 계산
        qkv = self.original_attn.qkv(x).view(x.shape[0], x.shape[1], 3, self.original_attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        batch_size, num_heads, num_patches, embed_dim = q.shape

        # Attention score 계산
        attn_scores = (q @ k.transpose(-2, -1)) * self.original_attn.scale  # [batch_size, num_heads, num_patches, num_patches]

        # 윈도우 기반 마스킹
        #############################################
        ##############  cls token 제거 ##############
        #############################################
        img_size = int((num_patches-1) ** 0.5)  # 14x14 이미지 가정  
        mask = torch.zeros(batch_size, num_heads, num_patches, num_patches, device=q.device)

        # 벡터화된 2x2 윈도우 마스킹 설정
        center_indices = torch.arange(1, num_patches, device=q.device).view(img_size, img_size)

        # CLS 토큰 마스킹
        mask[:, :, 0, :] = 1  # CLS 토큰이 모든 패치와 Attention
        mask[:, :, :, 0] = 1  # 모든 패치가 CLS 토큰과 Attention

        # #### window size가 짝수인 경우 ####
        # # 각 패치에 대해 2x2 윈도우 마스크 벡터화
        # for i in range(img_size):
        #     for j in range(img_size):
        #         query_idx = center_indices[i, j]

        #         # 2x2 윈도우 좌표 계산
        #         wi_end, wj_end = min(img_size, i + self.window_size), min(img_size, j + self.window_size)
                
        #         # 벡터화된 윈도우 인덱스 설정
        #         window_indices = center_indices[i:wi_end, j:wj_end].flatten()
        #         mask[:, :, query_idx, window_indices] = 1

        #         print(f"current q : {query_idx} k_index : {window_indices}")

        #### window size 홀수인 경우 ####
        # GPU 속도 최적화를 위해 벡터화된 마스킹 설정
        half_window = self.window_size // 2

        for i in range(img_size):
            for j in range(img_size):
                query_idx = center_indices[i, j]

                # 3x3 윈도우 좌표 계산
                wi_start = max(0, i - half_window)
                wi_end = min(img_size, i + half_window + 1)
                wj_start = max(0, j - half_window)
                wj_end = min(img_size, j + half_window + 1)

                # 마스크 설정: 3x3 윈도우 내의 key 인덱스들에 대해 마스킹
                window_indices = center_indices[wi_start:wi_end, wj_start:wj_end].flatten()
                mask[:, :, query_idx, window_indices] = 1

                # print(f"current q : {query_idx} k_index : {window_indices}")


        # mask 적용
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Attention score를 softmax로 정규화
        attn_scores = attn_scores.softmax(dim=-1)

        # 드롭아웃 적용
        attn_scores = self.original_attn.attn_drop(attn_scores)

        # value와의 연산을 통해 최종 결과 계산
        x = (attn_scores @ v).transpose(2, 3).contiguous()  # [batch_size, num_heads, embed_dim, num_patches]

        # 최종 결과 정리 및 투영
        x = x.view(batch_size, -1, x.shape[-1]).permute(0, 2, 1)  # [batch_size, embed_dim, num_patches]

        x = self.original_attn.proj(x)  # 최종 투영 연산
        x = self.original_attn.proj_drop(x)  # 드롭아웃 적용

        return x  # 최종 결과 반환


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
