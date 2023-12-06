#!/usr/bin/env python3
"""
vit with lora
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


class MoSLAttention(Attention):
    def __init__(self, config, vis, lora_config, lora_scalar=1.0, lora_dropout=0.0):
        super(MoSLAttention, self).__init__(config, vis)
        self.rank = lora_config.RANK
        self.mode = lora_config.MODE
        
        self.expert_num = lora_config.EXPERT_NUM
        self.share = lora_config.SHARE
        
        if 'q' in self.mode:
            if self.share == "down":
                self.lora_A_q = nn.Linear(config.hidden_size, self.rank, bias=False)
            else:
                self.lora_A_q = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.rank, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_A_q_merge = nn.Linear(config.hidden_size, self.rank, bias=False)
            if self.share == "up":
                self.lora_B_q = nn.Linear(self.rank, self.all_head_size, bias=False)
            else:
                self.lora_B_q = nn.ModuleList([
                    nn.Linear(self.rank, self.all_head_size, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_B_q_merge = nn.Linear(self.rank, self.all_head_size, bias=False)
        if 'k' in self.mode:
            if self.share == "down":
                self.lora_A_k = nn.Linear(config.hidden_size, self.rank, bias=False)
            else:
                self.lora_A_k = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.rank, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_A_k_merge = nn.Linear(config.hidden_size, self.rank, bias=False)
            if self.share == "up":
                self.lora_B_k = nn.Linear(self.rank, self.all_head_size, bias=False)
            else:
                self.lora_B_k = nn.ModuleList([
                    nn.Linear(self.rank, self.all_head_size, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_B_k_merge = nn.Linear(self.rank, self.all_head_size, bias=False)
        if 'v' in self.mode:
            if self.share == "down":
                self.lora_A_v = nn.Linear(config.hidden_size, self.rank, bias=False)
            else:
                self.lora_A_v = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.rank, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_A_v_merge = nn.Linear(config.hidden_size, self.rank, bias=False)
            if self.share == "up":
                self.lora_B_v = nn.Linear(self.rank, self.all_head_size, bias=False)
            else:
                self.lora_B_v = nn.ModuleList([
                    nn.Linear(self.rank, self.all_head_size, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_B_v_merge = nn.Linear(self.rank, self.all_head_size, bias=False)
        if 'o' in self.mode:
            if self.share == "down":
                self.lora_A_o = nn.Linear(config.hidden_size, self.rank, bias=False)
            else:
                self.lora_A_o = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.rank, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_A_o_merge = nn.Linear(config.hidden_size, self.rank, bias=False)
            if self.share == "up":
                self.lora_B_o = nn.Linear(self.rank, config.hidden_size, bias=False)
            else:
                self.lora_B_o = nn.ModuleList([
                    nn.Linear(self.rank, config.hidden_size, bias=False) for _ in range(self.expert_num)
                ])
                self.lora_B_o_merge = nn.Linear(self.rank, config.hidden_size, bias=False)
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scalar = lora_scalar
        
        if 'q' in self.mode:
            if self.share == "down":
                nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lora_A_q[0].weight, a=math.sqrt(5))
                for i in range(1, self.expert_num):
                    self.lora_A_q[i] = copy.deepcopy(self.lora_A_q[0])
                self.lora_A_q_merge = copy.deepcopy(self.lora_A_q[0])
            if self.share == "up":
                nn.init.zeros_(self.lora_B_q.weight)
            else:
                nn.init.zeros_(self.lora_B_q[0].weight)
                for i in range(1, self.expert_num):
                    self.lora_B_q[i] = copy.deepcopy(self.lora_B_q[0])
                self.lora_B_q_merge = copy.deepcopy(self.lora_B_q[0])
        if 'k' in self.mode:
            if self.share == "down":
                nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lora_A_k[0].weight, a=math.sqrt(5))
                for i in range(1, self.expert_num):
                    self.lora_A_k[i] = copy.deepcopy(self.lora_A_k[0])
                self.lora_A_k_merge = copy.deepcopy(self.lora_A_k[0])
            if self.share == "up":
                nn.init.zeros_(self.lora_B_k.weight)
            else:
                nn.init.zeros_(self.lora_B_k[0].weight)
                for i in range(1, self.expert_num):
                    self.lora_B_k[i] = copy.deepcopy(self.lora_B_k[0])
                self.lora_B_k_merge = copy.deepcopy(self.lora_B_k[0])
        if 'v' in self.mode:
            if self.share == "down":
                nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lora_A_v[0].weight, a=math.sqrt(5))
                for i in range(1, self.expert_num):
                    self.lora_A_v[i] = copy.deepcopy(self.lora_A_v[0])
                self.lora_A_v_merge = copy.deepcopy(self.lora_A_v[0])
            if self.share == "up":
                nn.init.zeros_(self.lora_B_v.weight)
            else:
                nn.init.zeros_(self.lora_B_v[0].weight)
                for i in range(1, self.expert_num):
                    self.lora_B_v[i] = copy.deepcopy(self.lora_B_v[0])
                self.lora_B_v_merge = copy.deepcopy(self.lora_B_v[0])
        if 'o' in self.mode:
            if self.share == "down":
                nn.init.kaiming_uniform_(self.lora_A_o.weight, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lora_A_o[0].weight, a=math.sqrt(5))
                for i in range(1, self.expert_num):
                    self.lora_A_o[i] = copy.deepcopy(self.lora_A_o[0])
                self.lora_A_o_merge = copy.deepcopy(self.lora_A_o[0])
            if self.share == "up":
                nn.init.zeros_(self.lora_B_o.weight)
            else:
                nn.init.zeros_(self.lora_B_o[0].weight)
                for i in range(1, self.expert_num):
                    self.lora_B_o[i] = copy.deepcopy(self.lora_B_o[0])
                self.lora_B_o_merge = copy.deepcopy(self.lora_B_o[0])

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        if self.training:
            if 'q' in self.mode:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                if self.share == "down":
                    mixed_query_layer += self.lora_B_q[B](self.lora_dropout(self.lora_A_q(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_query_layer += self.lora_B_q(self.lora_dropout(self.lora_A_q[A](hidden_states))) * self.lora_scalar
                else:
                    mixed_query_layer += self.lora_B_q[B](self.lora_dropout(self.lora_A_q[A](hidden_states))) * self.lora_scalar
            if 'k' in self.mode:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                if self.share == "down":
                    mixed_key_layer += self.lora_B_k[B](self.lora_dropout(self.lora_A_k(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_key_layer += self.lora_B_k(self.lora_dropout(self.lora_A_k[A](hidden_states))) * self.lora_scalar
                else:
                    mixed_key_layer += self.lora_B_k[B](self.lora_dropout(self.lora_A_k[A](hidden_states))) * self.lora_scalar
            if 'v' in self.mode:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                if self.share == "down":
                    mixed_value_layer += self.lora_B_v[B](self.lora_dropout(self.lora_A_v(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_value_layer += self.lora_B_v(self.lora_dropout(self.lora_A_v[A](hidden_states))) * self.lora_scalar
                else:
                    mixed_value_layer += self.lora_B_v[B](self.lora_dropout(self.lora_A_v[A](hidden_states))) * self.lora_scalar
        else:
            if 'q' in self.mode:
                if self.share == "down":
                    mixed_query_layer += self.lora_B_q_merge(self.lora_dropout(self.lora_A_q(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_query_layer += self.lora_B_q(self.lora_dropout(self.lora_A_q_merge(hidden_states))) * self.lora_scalar
                else:
                    mixed_query_layer += self.lora_B_q_merge(self.lora_dropout(self.lora_A_q_merge(hidden_states))) * self.lora_scalar
            if 'k' in self.mode:
                if self.share == "down":
                    mixed_key_layer += self.lora_B_k_merge(self.lora_dropout(self.lora_A_k(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_key_layer += self.lora_B_k(self.lora_dropout(self.lora_A_k_merge(hidden_states))) * self.lora_scalar
                else:
                    mixed_key_layer += self.lora_B_k_merge(self.lora_dropout(self.lora_A_k_merge(hidden_states))) * self.lora_scalar
            if 'v' in self.mode:
                if self.share == "down":
                    mixed_value_layer += self.lora_B_v_merge(self.lora_dropout(self.lora_A_v(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    mixed_value_layer += self.lora_B_v(self.lora_dropout(self.lora_A_v_merge(hidden_states))) * self.lora_scalar
                else:
                    mixed_value_layer += self.lora_B_v_merge(self.lora_dropout(self.lora_A_v_merge(hidden_states))) * self.lora_scalar

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        if self.training:
            if 'o' in self.mode:
                A, B = random.randint(0, self.expert_num - 1), random.randint(0, self.expert_num - 1)
                if self.share == "down":
                    attention_output += self.lora_B_o[B](self.lora_dropout(self.lora_A_o(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    attention_output += self.lora_B_o(self.lora_dropout(self.lora_A_o[A](hidden_states))) * self.lora_scalar
                else:
                    attention_output += self.lora_B_o[B](self.lora_dropout(self.lora_A_o[A](hidden_states))) * self.lora_scalar
        else:
            if 'o' in self.mode:
                if self.share == "down":
                    attention_output += self.lora_B_o_merge(self.lora_dropout(self.lora_A_o(hidden_states))) * self.lora_scalar
                elif self.share == "up":
                    attention_output += self.lora_B_o(self.lora_dropout(self.lora_A_o_merge(hidden_states))) * self.lora_scalar
                else:
                    attention_output += self.lora_B_o_merge(self.lora_dropout(self.lora_A_o_merge(hidden_states))) * self.lora_scalar
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

    def merge(self, mode='add'):
        if self.share != "down":
            if 'q' in self.mode:
                self.lora_A_q_merge.weight.data = torch.zeros_like(self.lora_A_q_merge.weight.data)
            if 'k' in self.mode:
                self.lora_A_k_merge.weight.data = torch.zeros_like(self.lora_A_k_merge.weight.data)
            if 'v' in self.mode:
                self.lora_A_v_merge.weight.data = torch.zeros_like(self.lora_A_v_merge.weight.data)
            if 'o' in self.mode:
                self.lora_A_o_merge.weight.data = torch.zeros_like(self.lora_A_o_merge.weight.data)
        if self.share != "up":
            if 'q' in self.mode:
                self.lora_B_q_merge.weight.data = torch.zeros_like(self.lora_B_q_merge.weight.data)
            if 'k' in self.mode:
                self.lora_B_k_merge.weight.data = torch.zeros_like(self.lora_B_k_merge.weight.data)
            if 'v' in self.mode:
                self.lora_B_v_merge.weight.data = torch.zeros_like(self.lora_B_v_merge.weight.data)
            if 'o' in self.mode:
                self.lora_B_o_merge.weight.data = torch.zeros_like(self.lora_B_o_merge.weight.data)
        
        if mode == 'add':
            for i in range(self.expert_num):
                if self.share != "down":
                    if 'q' in self.mode:
                        self.lora_A_q_merge.weight.data += self.lora_A_q[i].weight.data
                    if 'k' in self.mode:
                        self.lora_A_k_merge.weight.data += self.lora_A_k[i].weight.data
                    if 'v' in self.mode:
                        self.lora_A_v_merge.weight.data += self.lora_A_v[i].weight.data
                    if 'o' in self.mode:
                        self.lora_A_o_merge.weight.data += self.lora_A_o[i].weight.data
                if self.share != "up":
                    if 'q' in self.mode:
                        self.lora_B_q_merge.weight.data += self.lora_B_q[i].weight.data
                    if 'k' in self.mode:
                        self.lora_B_k_merge.weight.data += self.lora_B_k[i].weight.data
                    if 'v' in self.mode:
                        self.lora_B_v_merge.weight.data += self.lora_B_v[i].weight.data
                    if 'o' in self.mode:
                        self.lora_B_o_merge.weight.data += self.lora_B_o[i].weight.data

        elif mode == 'joint':
            for i in range(self.expert_num):
                if self.share != "down":
                    if 'q' in self.mode:
                        mask = self.lora_A_q[i].weight.mask > 0.5
                        self.lora_A_q_merge.weight.data[mask] = self.lora_A_q[i].weight.data[mask]
                    if 'k' in self.mode:
                        mask = self.lora_A_k[i].weight.mask > 0.5
                        self.lora_A_k_merge.weight.data[mask] = self.lora_A_k[i].weight.data[mask]
                    if 'v' in self.mode:
                        mask = self.lora_A_v[i].weight.mask > 0.5
                        self.lora_A_v_merge.weight.data[mask] = self.lora_A_v[i].weight.data[mask]
                    if 'o' in self.mode:
                        mask = self.lora_A_o[i].weight.mask > 0.5
                        self.lora_A_o_merge.weight.data[mask] = self.lora_A_o[i].weight.data[mask]
                if self.share != "up":
                    if 'q' in self.mode:
                        mask = self.lora_B_q[i].weight.mask > 0.5
                        self.lora_B_q_merge.weight.data[mask] = self.lora_B_q[i].weight.data[mask]
                    if 'k' in self.mode:
                        mask = self.lora_B_k[i].weight.mask > 0.5
                        self.lora_B_k_merge.weight.data[mask] = self.lora_B_k[i].weight.data[mask]
                    if 'v' in self.mode:
                        mask = self.lora_B_v[i].weight.mask > 0.5
                        self.lora_B_v_merge.weight.data[mask] = self.lora_B_v[i].weight.data[mask]
                    if 'o' in self.mode:
                        mask = self.lora_B_o[i].weight.mask > 0.5
                        self.lora_B_o_merge.weight.data[mask] = self.lora_B_o[i].weight.data[mask]
            
        else:
            raise NotImplementedError(f'Unknown merge mode: {mode}')
        
        if self.share != "down":
            if 'q' in self.mode:
                self.lora_A_q_merge.weight.data /= self.expert_num
            if 'k' in self.mode:
                self.lora_A_k_merge.weight.data /= self.expert_num
            if 'v' in self.mode:
                self.lora_A_v_merge.weight.data /= self.expert_num
            if 'o' in self.mode:
                self.lora_A_o_merge.weight.data /= self.expert_num
        if self.share != "up":
            if 'q' in self.mode:
                self.lora_B_q_merge.weight.data /= self.expert_num
            if 'k' in self.mode:
                self.lora_B_k_merge.weight.data /= self.expert_num
            if 'v' in self.mode:
                self.lora_B_v_merge.weight.data /= self.expert_num
            if 'o' in self.mode:
                self.lora_B_o_merge.weight.data /= self.expert_num


class MoSLBlock(nn.Module):
    def __init__(self, config, vis, lora_config, lora_scalar=1.0, lora_dropout=0.0):
        super(MoSLBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = MoSLAttention(config, vis, lora_config, lora_scalar, lora_dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
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


class MoSLEncoder(nn.Module):
    def __init__(self, config, mid, vis, lora_cfg):
        super(MoSLEncoder, self).__init__()
        self.mid = mid
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = MoSLBlock(config, vis, lora_cfg, lora_scalar=lora_cfg.SCALAR, lora_dropout=lora_cfg.DROPOUT)
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


class MoSLTransformer(nn.Module):
    def __init__(self, config, img_size, mid, vis, lora_cfg):
        super(MoSLTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = MoSLEncoder(config, mid, vis, lora_cfg)

    def forward(self, input_ids):
        embedded = self.embeddings(input_ids)

        encoded, hidden_states, attn_weights = self.encoder(embedded)
        return encoded, hidden_states, attn_weights


class MoSLVisionTransformer(nn.Module):
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, mid=False, vis=False, lora_cfg=None
    ):
        super(MoSLVisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = MoSLTransformer(config, img_size, mid, vis, lora_cfg)
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
            blk.attn.merge(mode)
