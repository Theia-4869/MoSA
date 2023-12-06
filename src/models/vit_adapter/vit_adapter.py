#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("MOSA")


class AdaptedBlock(nn.Module):
    def __init__(self, config, vis, adapter_config, adapter_scalar=1.0, dropout=0.0):
        super(AdaptedBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.bottleneck_size = adapter_config.BOTTLENECK_SIZE
        self.style = adapter_config.STYLE
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        
        if adapter_config.STYLE == "AdaptFormer":
            self.adapter_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.adapter_down = nn.Linear(
                config.hidden_size,
                adapter_config.BOTTLENECK_SIZE
            )
            self.adapter_up = nn.Linear(
                adapter_config.BOTTLENECK_SIZE,
                config.hidden_size
            )
            self.adapter_act_fn = nn.ReLU()
            self.dropout = dropout
            if adapter_scalar is None:
                self.adapter_scale = nn.Parameter(torch.ones(1))
            else:
                self.adapter_scale = adapter_scalar
            
            nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)
        
        elif adapter_config.STYLE == "Pfeiffer":
            self.adapter_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.adapter_down = nn.Linear(
                config.hidden_size,
                adapter_config.BOTTLENECK_SIZE
            )
            self.adapter_up = nn.Linear(
                adapter_config.BOTTLENECK_SIZE,
                config.hidden_size
            )
            self.adapter_act_fn = nn.ReLU()
            
            nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif adapter_config.STYLE == "Houlsby":
            self.adapter_down_attn = nn.Linear(
                config.hidden_size,
                adapter_config.BOTTLENECK_SIZE
            )
            self.adapter_up_attn = nn.Linear(
                adapter_config.BOTTLENECK_SIZE,
                config.hidden_size
            )
            self.adapter_down_ffn = nn.Linear(
                config.hidden_size,
                adapter_config.BOTTLENECK_SIZE
            )
            self.adapter_up_ffn = nn.Linear(
                adapter_config.BOTTLENECK_SIZE,
                config.hidden_size
            )
            self.adapter_act_fn = nn.SiLU()
            
            nn.init.kaiming_uniform_(self.adapter_down_attn.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down_attn.bias)
            nn.init.zeros_(self.adapter_up_attn.weight)
            nn.init.zeros_(self.adapter_up_attn.bias)
            nn.init.kaiming_uniform_(self.adapter_down_ffn.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down_ffn.bias)
            nn.init.zeros_(self.adapter_up_ffn.weight)
            nn.init.zeros_(self.adapter_up_ffn.bias)

    def forward(self, x):
        if self.style == "AdaptFormer":
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
            adpt = self.adapter_down(adpt)
            adpt = self.adapter_act_fn(adpt)
            adpt = nn.functional.dropout(adpt, p=self.dropout, training=self.training)
            adpt = self.adapter_up(adpt)
            x = adpt * self.adapter_scale + x
            # ...end

            x = x + h
        
        elif self.style == "Pfeiffer":
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
            adpt = self.adapter_down(adpt)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_up(adpt)
            x = adpt + x
            # ...end

            x = x + h
        
        elif self.style == "Houlsby":
            # same as reguluar ViT block
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            x = x + h

            # start to insert adapter layers...
            adpt_attn = self.adapter_down_attn(x)
            adpt_attn = self.adapter_act_fn(adpt_attn)
            adpt_attn = self.adapter_up_attn(adpt_attn)
            x = adpt_attn + x
            # ...end

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h

            # start to insert adapter layers...
            adpt_ffn = self.adapter_down_ffn(x)
            adpt_ffn = self.adapter_act_fn(adpt_ffn)
            adpt_ffn = self.adapter_up_ffn(adpt_ffn)
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


class AdaptedEncoder(nn.Module):
    def __init__(self, config, vis, adapter_cfg):
        super(AdaptedEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = AdaptedBlock(config, vis, adapter_cfg, adapter_scalar=adapter_cfg.SCALAR, dropout=adapter_cfg.DROPOUT)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class AdaptedTransformer(nn.Module):
    def __init__(self, config, img_size, vis, adapter_cfg):
        super(AdaptedTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = AdaptedEncoder(config, vis, adapter_cfg)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class AdaptedVisionTransformer(nn.Module):
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, vis=False, adapter_cfg=None
    ):
        super(AdaptedVisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = AdaptedTransformer(config, img_size, vis, adapter_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
        

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
