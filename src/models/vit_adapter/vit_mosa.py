#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("MOSA")


class MoSABlock(nn.Module):
    def __init__(self, config, vis, adapter_config, adapter_scalar=1.0, adapter_dropout=0.0):
        super(MoSABlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.bottleneck_size = adapter_config.BOTTLENECK_SIZE
        self.style = adapter_config.STYLE
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        
        self.expert_num = adapter_config.EXPERT_NUM
        self.share = adapter_config.SHARE
        
        if adapter_config.STYLE == "AdaptFormer":
            self.adapter_norm = LayerNorm(config.hidden_size, eps=1e-6)
            if self.share == "down":
                self.adapter_down = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            else:
                self.adapter_down = nn.ModuleList([
                    nn.Linear(
                        config.hidden_size,
                        adapter_config.BOTTLENECK_SIZE
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_down_merge = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            if self.share == "up":
                self.adapter_up = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            else:
                self.adapter_up = nn.ModuleList([
                    nn.Linear(
                        adapter_config.BOTTLENECK_SIZE,
                        config.hidden_size
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_up_merge = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            self.adapter_act_fn = nn.GELU()
            self.adapter_dropout = adapter_dropout
            if adapter_scalar is None:
                self.adapter_scale = nn.Parameter(torch.ones(1))
            else:
                self.adapter_scale = adapter_scalar
        
        elif adapter_config.STYLE == "Pfeiffer":
            self.adapter_norm = LayerNorm(config.hidden_size, eps=1e-6)
            if self.share == "down":
                self.adapter_down = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            else:
                self.adapter_down = nn.ModuleList([
                    nn.Linear(
                        config.hidden_size,
                        adapter_config.BOTTLENECK_SIZE
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_down_merge = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            if self.share == "up":
                self.adapter_up = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            else:
                self.adapter_up = nn.ModuleList([
                    nn.Linear(
                        adapter_config.BOTTLENECK_SIZE,
                        config.hidden_size
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_up_merge = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            self.adapter_act_fn = nn.ReLU()

        elif adapter_config.STYLE == "Houlsby":
            if self.share == "down":
                self.adapter_down_attn = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            else:
                self.adapter_down_attn = nn.ModuleList([
                    nn.Linear(
                        config.hidden_size,
                        adapter_config.BOTTLENECK_SIZE
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_down_attn_merge = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            if self.share == "up":
                self.adapter_up_attn = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            else:
                self.adapter_up_attn = nn.ModuleList([
                    nn.Linear(
                        adapter_config.BOTTLENECK_SIZE,
                        config.hidden_size
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_up_attn_merge = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            if self.share == "down":
                self.adapter_down_ffn = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            else:
                self.adapter_down_ffn = nn.ModuleList([
                    nn.Linear(
                        config.hidden_size,
                        adapter_config.BOTTLENECK_SIZE
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_down_ffn_merge = nn.Linear(
                    config.hidden_size,
                    adapter_config.BOTTLENECK_SIZE
                )
            if self.share == "up":
                self.adapter_up_ffn = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            else:
                self.adapter_up_ffn = nn.ModuleList([
                    nn.Linear(
                        adapter_config.BOTTLENECK_SIZE,
                        config.hidden_size
                    ) for _ in range(self.expert_num)
                ])
                self.adapter_up_ffn_merge = nn.Linear(
                    adapter_config.BOTTLENECK_SIZE,
                    config.hidden_size
                )
            self.adapter_act_fn = nn.SiLU()

        if self.share == "down":
            if adapter_config.STYLE == "AdaptFormer" or adapter_config.STYLE == "Pfeiffer":
                nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down.bias)
            elif adapter_config.STYLE == "Houlsby":
                nn.init.kaiming_uniform_(self.adapter_down_attn.weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down_attn.bias)
                nn.init.kaiming_uniform_(self.adapter_down_ffn.weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down_ffn.bias)
        else:
            if adapter_config.STYLE == "AdaptFormer" or adapter_config.STYLE == "Pfeiffer":
                nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down[0].bias)
                for i in range(1, self.expert_num):
                    self.adapter_down[i] = copy.deepcopy(self.adapter_down[0])
                self.adapter_down_merge = copy.deepcopy(self.adapter_down[0])
            elif adapter_config.STYLE == "Houlsby":
                nn.init.kaiming_uniform_(self.adapter_down_attn[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down_attn[0].bias)
                nn.init.kaiming_uniform_(self.adapter_down_ffn[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_down_ffn[0].bias)
                for i in range(1, self.expert_num):
                    self.adapter_down_attn[i] = copy.deepcopy(self.adapter_down_attn[0])
                    self.adapter_down_ffn[i] = copy.deepcopy(self.adapter_down_ffn[0])
                self.adapter_down_attn_merge = copy.deepcopy(self.adapter_down_attn[0])
                self.adapter_down_ffn_merge = copy.deepcopy(self.adapter_down_ffn[0])
        if self.share == "up":
            if adapter_config.STYLE == "AdaptFormer" or adapter_config.STYLE == "Pfeiffer":
                nn.init.zeros_(self.adapter_up.weight)
                nn.init.zeros_(self.adapter_up.bias)
            elif adapter_config.STYLE == "Houlsby":
                nn.init.zeros_(self.adapter_up_attn.weight)
                nn.init.zeros_(self.adapter_up_attn.bias)
                nn.init.zeros_(self.adapter_up_ffn.weight)
                nn.init.zeros_(self.adapter_up_ffn.bias)
        else:
            if adapter_config.STYLE == "AdaptFormer" or adapter_config.STYLE == "Pfeiffer":
                nn.init.zeros_(self.adapter_up[0].weight)
                nn.init.zeros_(self.adapter_up[0].bias)
                for i in range(1, self.expert_num):
                    self.adapter_up[i] = copy.deepcopy(self.adapter_up[0])
                self.adapter_up_merge = copy.deepcopy(self.adapter_up[0])
            elif adapter_config.STYLE == "Houlsby":
                nn.init.zeros_(self.adapter_up_attn[0].weight)
                nn.init.zeros_(self.adapter_up_attn[0].bias)
                nn.init.zeros_(self.adapter_up_ffn[0].weight)
                nn.init.zeros_(self.adapter_up_ffn[0].bias)
                for i in range(1, self.expert_num):
                    self.adapter_up_attn[i] = copy.deepcopy(self.adapter_up_attn[0])
                    self.adapter_up_ffn[i] = copy.deepcopy(self.adapter_up_ffn[0])
                self.adapter_up_attn_merge = copy.deepcopy(self.adapter_up_attn[0])
                self.adapter_up_ffn_merge = copy.deepcopy(self.adapter_up_ffn[0])
        

    def forward(self, x):
        if self.style == "AdaptFormer":
            if self.training:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)

                # start to insert adapter layers...
                adpt = self.adapter_norm(h)
                if self.share == "down":
                    adpt = self.adapter_down(adpt)
                else:
                    adpt = self.adapter_down[A](adpt)
                adpt = self.adapter_act_fn(adpt)
                adpt = nn.functional.dropout(adpt, p=self.adapter_dropout, training=self.training)
                if self.share == "up":
                    adpt = self.adapter_up(adpt)
                else:
                    adpt = self.adapter_up[B](adpt)
                
                x = adpt * self.adapter_scale + x
                # ...end

                x = x + h
            
            else:
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)

                # start to insert adapter layers...
                adpt = self.adapter_norm(h)
                if self.share == "down":
                    adpt = self.adapter_down(adpt)
                else:
                    adpt = self.adapter_down_merge(adpt)
                adpt = self.adapter_act_fn(adpt)
                adpt = nn.functional.dropout(adpt, p=self.adapter_dropout, training=self.training)
                if self.share == "up":
                    adpt = self.adapter_up(adpt)
                else:
                    adpt = self.adapter_up_merge(adpt)
                
                x = adpt * self.adapter_scale + x
                # ...end

                x = x + h
        
        elif self.style == "Pfeiffer":
            if self.training:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)

                # start to insert adapter layers...
                adpt = self.adapter_norm(x)
                if self.share == "down":
                    adpt = self.adapter_down(adpt)
                else:
                    adpt = self.adapter_down[A](adpt)
                adpt = self.adapter_act_fn(adpt)
                if self.share == "up":
                    adpt = self.adapter_up(adpt)
                else:
                    adpt = self.adapter_up[B](adpt)
                x = adpt + x
                # ...end

                x = x + h
            
            else:
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)

                # start to insert adapter layers...
                adpt = self.adapter_norm(x)
                if self.share == "down":
                    adpt = self.adapter_down(adpt)
                else:
                    adpt = self.adapter_down_merge(adpt)
                adpt = self.adapter_act_fn(adpt)
                if self.share == "up":
                    adpt = self.adapter_up(adpt)
                else:
                    adpt = self.adapter_up_merge(adpt)
                
                x = adpt + x
                # ...end

                x = x + h
        
        elif self.style == "Houlsby":
            if self.training:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                C, D = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h
                
                # start to insert adapter layers...
                if self.share == "down":
                    adpt_attn = self.adapter_down_attn(x)
                else:
                    adpt_attn = self.adapter_down_attn[A](x)
                adpt_attn = self.adapter_act_fn(adpt_attn)
                if self.share == "up":
                    adpt_attn = self.adapter_up_attn(adpt_attn)
                else:
                    adpt_attn = self.adapter_up_attn[B](adpt_attn)
                x = adpt_attn + x
                # ...end

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)
                x = x + h

                # start to insert adapter layers...
                if self.share == "down":
                    adpt_ffn = self.adapter_down_ffn(x)
                else:
                    adpt_ffn = self.adapter_down_ffn[C](x)
                adpt_ffn = self.adapter_act_fn(adpt_ffn)
                if self.share == "up":
                    adpt_ffn = self.adapter_up_ffn(adpt_ffn)
                else:
                    adpt_ffn = self.adapter_up_ffn[D](adpt_ffn)
                x = adpt_ffn + x
                # ...end
            
            else:
                # same as reguluar ViT block
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h
                
                # start to insert adapter layers...
                if self.share == "down":
                    adpt_attn = self.adapter_down_attn(x)
                else:
                    adpt_attn = self.adapter_down_attn_merge(x)
                adpt_attn = self.adapter_act_fn(adpt_attn)
                if self.share == "up":
                    adpt_attn = self.adapter_up_attn(adpt_attn)
                else:
                    adpt_attn = self.adapter_up_attn_merge(adpt_attn)
                x = adpt_attn + x
                # ...end

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)
                x = x + h

                # start to insert adapter layers...
                if self.share == "down":
                    adpt_ffn = self.adapter_down_ffn(x)
                else:
                    adpt_ffn = self.adapter_down_ffn_merge(x)
                adpt_ffn = self.adapter_act_fn(adpt_ffn)
                if self.share == "up":
                    adpt_ffn = self.adapter_up_ffn(adpt_ffn)
                else:
                    adpt_ffn = self.adapter_up_ffn_merge(adpt_ffn)
                x = adpt_ffn + x
                # ...end
            
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
    
    def merge(self, mode='add'):
        if self.share != "down":
            if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                self.adapter_down_merge.weight.data = torch.zeros_like(self.adapter_down_merge.weight.data)
                self.adapter_down_merge.bias.data = torch.zeros_like(self.adapter_down_merge.bias.data)
            elif self.style == "Houlsby":
                self.adapter_down_attn_merge.weight.data = torch.zeros_like(self.adapter_down_attn_merge.weight.data)
                self.adapter_down_attn_merge.bias.data = torch.zeros_like(self.adapter_down_attn_merge.bias.data)
                self.adapter_down_ffn_merge.weight.data = torch.zeros_like(self.adapter_down_ffn_merge.weight.data)
                self.adapter_down_ffn_merge.bias.data = torch.zeros_like(self.adapter_down_ffn_merge.bias.data)
        if self.share != "up":
            if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                self.adapter_up_merge.weight.data = torch.zeros_like(self.adapter_up_merge.weight.data)
                self.adapter_up_merge.bias.data = torch.zeros_like(self.adapter_up_merge.bias.data)
            elif self.style == "Houlsby":
                self.adapter_up_attn_merge.weight.data = torch.zeros_like(self.adapter_up_attn_merge.weight.data)
                self.adapter_up_attn_merge.bias.data = torch.zeros_like(self.adapter_up_attn_merge.bias.data)
                self.adapter_up_ffn_merge.weight.data = torch.zeros_like(self.adapter_up_ffn_merge.weight.data)
                self.adapter_up_ffn_merge.bias.data = torch.zeros_like(self.adapter_up_ffn_merge.bias.data)
        
        if mode == 'add':
            for i in range(self.expert_num):
                if self.share != "down":
                    if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                        self.adapter_down_merge.weight.data += self.adapter_down[i].weight.data
                        self.adapter_down_merge.bias.data += self.adapter_down[i].bias.data
                    elif self.style == "Houlsby":
                        self.adapter_down_attn_merge.weight.data += self.adapter_down_attn[i].weight.data
                        self.adapter_down_attn_merge.bias.data += self.adapter_down_attn[i].bias.data
                        self.adapter_down_ffn_merge.weight.data += self.adapter_down_ffn[i].weight.data
                        self.adapter_down_ffn_merge.bias.data += self.adapter_down_ffn[i].bias.data
                if self.share != "up":
                    if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                        self.adapter_up_merge.weight.data += self.adapter_up[i].weight.data
                        self.adapter_up_merge.bias.data += self.adapter_up[i].bias.data
                    elif self.style == "Houlsby":
                        self.adapter_up_attn_merge.weight.data += self.adapter_up_attn[i].weight.data
                        self.adapter_up_attn_merge.bias.data += self.adapter_up_attn[i].bias.data
                        self.adapter_up_ffn_merge.weight.data += self.adapter_up_ffn[i].weight.data
                        self.adapter_up_ffn_merge.bias.data += self.adapter_up_ffn[i].bias.data

        elif mode == 'joint':
            for i in range(self.expert_num):
                if self.share != "down":
                    if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                        mask = self.adapter_down[i].weight.mask > 0.5
                        self.adapter_down_merge.weight.data[mask] = self.adapter_down[i].weight.data[mask]
                        mask = self.adapter_down[i].bias.mask > 0.5
                        self.adapter_down_merge.bias.data[mask] = self.adapter_down[i].bias.data[mask]
                    elif self.style == "Houlsby":
                        mask = self.adapter_down_attn[i].weight.mask > 0.5
                        self.adapter_down_attn_merge.weight.data[mask] = self.adapter_down_attn[i].weight.data[mask]
                        mask = self.adapter_down_attn[i].bias.mask > 0.5
                        self.adapter_down_attn_merge.bias.data[mask] = self.adapter_down_attn[i].bias.data[mask]
                        mask = self.adapter_down_ffn[i].weight.mask > 0.5
                        self.adapter_down_ffn_merge.weight.data[mask] = self.adapter_down_ffn[i].weight.data[mask]
                        mask = self.adapter_down_ffn[i].bias.mask > 0.5
                        self.adapter_down_ffn_merge.bias.data[mask] = self.adapter_down_ffn[i].bias.data[mask]
                if self.share != "up":
                    if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                        mask = self.adapter_up[i].weight.mask > 0.5
                        self.adapter_up_merge.weight.data[mask] = self.adapter_up[i].weight.data[mask]
                        mask = self.adapter_up[i].bias.mask > 0.5
                        self.adapter_up_merge.bias.data[mask] = self.adapter_up[i].bias.data[mask]
                    elif self.style == "Houlsby":
                        mask = self.adapter_up_attn[i].weight.mask > 0.5
                        self.adapter_up_attn_merge.weight.data[mask] = self.adapter_up_attn[i].weight.data[mask]
                        mask = self.adapter_up_attn[i].bias.mask > 0.5
                        self.adapter_up_attn_merge.bias.data[mask] = self.adapter_up_attn[i].bias.data[mask]
                        mask = self.adapter_up_ffn[i].weight.mask > 0.5
                        self.adapter_up_ffn_merge.weight.data[mask] = self.adapter_up_ffn[i].weight.data[mask]
                        mask = self.adapter_up_ffn[i].bias.mask > 0.5
                        self.adapter_up_ffn_merge.bias.data[mask] = self.adapter_up_ffn[i].bias.data[mask]
            
        else:
            raise NotImplementedError(f'Unknown merge mode: {mode}')
        
        if self.share != "down":
            if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                self.adapter_down_merge.weight.data /= self.expert_num
                self.adapter_down_merge.bias.data /= self.expert_num
            elif self.style == "Houlsby":
                self.adapter_down_attn_merge.weight.data /= self.expert_num
                self.adapter_down_attn_merge.bias.data /= self.expert_num
                self.adapter_down_ffn_merge.weight.data /= self.expert_num
                self.adapter_down_ffn_merge.bias.data /= self.expert_num
        if self.share != "up":
            if self.style == "AdaptFormer" or self.style == "Pfeiffer":
                self.adapter_up_merge.weight.data /= self.expert_num
                self.adapter_up_merge.bias.data /= self.expert_num
            elif self.style == "Houlsby":
                self.adapter_up_attn_merge.weight.data /= self.expert_num
                self.adapter_up_attn_merge.bias.data /= self.expert_num
                self.adapter_up_ffn_merge.weight.data /= self.expert_num
                self.adapter_up_ffn_merge.bias.data /= self.expert_num


class MoSAEncoder(nn.Module):
    def __init__(self, config, mid, vis, adapter_cfg):
        super(MoSAEncoder, self).__init__()
        self.mid = mid
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = MoSABlock(config, vis, adapter_cfg, adapter_scalar=adapter_cfg.SCALAR, adapter_dropout=adapter_cfg.DROPOUT)
            self.layer.append(layer)

    def forward(self, states):
        hidden_states = []
        attn_weights = []
        for layer_block in self.layer:
            states, weights = layer_block(states)
            if self.mid:
                hidden_states.append(states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(states)
        return encoded, hidden_states, attn_weights


class MoSATransformer(nn.Module):
    def __init__(self, config, img_size, mid, vis, adapter_cfg):
        super(MoSATransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = MoSAEncoder(config, mid, vis, adapter_cfg)

    def forward(self, input_ids):
        embedded = self.embeddings(input_ids)

        encoded, hidden_states, attn_weights = self.encoder(embedded)
        return encoded, hidden_states, attn_weights


class MoSAVisionTransformer(nn.Module):
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, mid=False, vis=False, adapter_cfg=None
    ):
        super(MoSAVisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = MoSATransformer(config, img_size, mid, vis, adapter_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, mid=False, vis=False):
        x, hidden_states, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not mid and not vis:
            return logits
        elif mid and not vis:
            return logits, hidden_states
        elif vis and not mid:
            return logits, attn_weights
        else:
            return logits, hidden_states, attn_weights


    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
    
    def merge(self, mode='add'):
        for blk in self.transformer.encoder.layer:
            blk.merge(mode)
