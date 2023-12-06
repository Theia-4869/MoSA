#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging
logger = logging.get_logger("MOSA")


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "adapter" not in cfg.MODEL.TRANSFER_TYPE \
            and "lora" not in cfg.MODEL.TRANSFER_TYPE and "mos" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt, adapter
            self.froze_enc = False
        
        if cfg.MODEL.TRANSFER_TYPE == "adapter" or cfg.MODEL.TRANSFER_TYPE == "mosa":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None
        
        if cfg.MODEL.TRANSFER_TYPE == "lora" or cfg.MODEL.TRANSFER_TYPE == "mosl":
            lora_cfg = cfg.MODEL.LORA
        else:
            lora_cfg = None

        self.build_backbone(cfg, adapter_cfg, lora_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)

    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, cfg, adapter_cfg, lora_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, cfg.MODEL.MODEL_ROOT, adapter_cfg, lora_cfg, load_pretrain, vis,
        )
        
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.layer.{}".format(total_layer - 3) not in k and "transformer.encoder.layer.{}".format(total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False
        
        # adapter
        elif transfer_type == "adapter" or transfer_type == "mosa":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k or "merge" in k:
                    p.requires_grad = False
        
        # lora
        elif transfer_type == "lora" or transfer_type == "mosl":
            for k, p in self.enc.named_parameters():
                if "lora" not in k or "merge" in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False, mid=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        if mid:
            x, h = self.enc(x, mid=True)
        else:
            x = self.enc(x)

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output
        
        logit = self.head(x)

        if return_feature:
            return x, logit
        if mid:
            return logit, h

        return logit

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class Swin(ViT):
    """Swin-related model."""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, cfg, adapter_cfg, lora_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, cfg.MODEL.MODEL_ROOT, adapter_cfg, lora_cfg, load_pretrain, vis
        )

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1, total_blocks - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(total_layer - 2) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False
        
        # adapter
        elif transfer_type == "adapter" or transfer_type == "mosa":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k or "merge" in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "sidetune":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.weight" not in k  and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))
