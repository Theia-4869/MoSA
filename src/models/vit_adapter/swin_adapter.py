#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn

from ..vit_backbones.swin_transformer import Mlp, SwinTransformerBlock, BasicLayer, PatchMerging, SwinTransformer
from ...utils import logging
logger = logging.get_logger("MOSA")


class AdaptedMlp(Mlp):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
        adapter_config=None, adapter_scalar=1.0, dropout=0.0
    ):
        super(AdaptedMlp, self).__init__(in_features, hidden_features, out_features, act_layer, drop)
        self.adapter_config = adapter_config
        
        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar
        self.dropout = dropout

        out_features = out_features or in_features
        self.adapter_down = nn.Linear(
            in_features,
            adapter_config.BOTTLENECK_SIZE
        )
        self.adapter_up = nn.Linear(
            adapter_config.BOTTLENECK_SIZE,
            out_features
        )
        self.adapter_act_fn = nn.ReLU()

        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)

        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        # same as reguluar Mlp block

        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # start to insert adapter layers...
        adpt = self.adapter_down(h)
        adpt = self.adapter_act_fn(adpt)
        adpt = nn.functional.dropout(adpt, p=self.dropout, training=self.training)
        adpt = self.adapter_up(adpt)
        
        x = adpt * self.adapter_scale + x
        # ...end
        
        return x


class AdaptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_config=None
    ):
        super(AdaptedSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer
        )
        self.adapter_config = adapter_config
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = AdaptedMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop, adapter_config=adapter_config
        )


class AdaptedSwinTransformer(SwinTransformer):
    def __init__(
        self, adapter_config, img_size=224, patch_size=4, in_chans=3, 
        num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, 
        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, **kwargs
    ):
        super(AdaptedSwinTransformer, self).__init__(
            img_size, patch_size, in_chans, num_classes, embed_dim, depths,
            num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
            use_checkpoint, **kwargs
        )
        self.adapter_config = adapter_config
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                        input_resolution=(
                            self.patches_resolution[0] // (2 ** i_layer),
                            self.patches_resolution[1] // (2 ** i_layer)),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                        use_checkpoint=use_checkpoint,
                        block_module=AdaptedSwinTransformerBlock,
                        adapter_config=adapter_config
                    )
            self.layers.append(layer)
