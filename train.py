#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from random import randint
from time import sleep
import wandb

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.build_pruner import build_pruner, log_pruned_model_info
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    bn = cfg.MODEL.ADAPTER.BOTTLENECK_SIZE
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_bn{bn}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        # sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader, val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("MOSA")
    if args.use_wandb:
        wandb.init(
            project='MOSA',
            name='{}_{}_{}'.format(cfg.DATA.NAME, cfg.MODEL.TRANSFER_TYPE, cfg.MODEL.HYPER.HYPER),
            config=cfg
        )

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    
    if args.sparse_train:
        logger.info("Constructing pruner...")
        pruner = build_pruner(cfg)
    
    # for k, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(k, p.shape)
    # raise ValueError("stop here")
    
    if args.sparse_train:
        logger.info("Pruning model...")
        if cfg.MODEL.TYPE == "vit":
            if cfg.MODEL.TRANSFER_TYPE == "adapter" or cfg.MODEL.TRANSFER_TYPE == "mosa":
                if cfg.MODEL.ADAPTER.MOE:
                    for blk in model.enc.transformer.encoder.layer:
                        if cfg.MODEL.ADAPTER.SHARE != "down":
                            if cfg.MODEL.ADAPTER.STYLE == "AdaptFormer" or cfg.MODEL.ADAPTER.STYLE == "Pfeiffer":
                                score = pruner.score(blk.adapter_down[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down[i].weight.mask = m
                                score = pruner.score(blk.adapter_down[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down[i].bias.mask = m
                            elif cfg.MODEL.ADAPTER.STYLE == "Houlsby":
                                score = pruner.score(blk.adapter_down_attn[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down_attn[i].weight.mask = m
                                score = pruner.score(blk.adapter_down_attn[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down_attn[i].bias.mask = m
                                score = pruner.score(blk.adapter_down_ffn[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down_ffn[i].weight.mask = m
                                score = pruner.score(blk.adapter_down_ffn[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_down_ffn[i].bias.mask = m
                        
                        if cfg.MODEL.ADAPTER.SHARE != "up":
                            if cfg.MODEL.ADAPTER.STYLE == "AdaptFormer" or cfg.MODEL.ADAPTER.STYLE == "Pfeiffer":
                                score = pruner.score(blk.adapter_up[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up[i].weight.mask = m
                                score = pruner.score(blk.adapter_up[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up[i].bias.mask = m
                            elif cfg.MODEL.ADAPTER.STYLE == "Houlsby":
                                score = pruner.score(blk.adapter_up_attn[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up_attn[i].weight.mask = m
                                score = pruner.score(blk.adapter_up_attn[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up_attn[i].bias.mask = m
                                score = pruner.score(blk.adapter_up_ffn[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up_ffn[i].weight.mask = m
                                score = pruner.score(blk.adapter_up_ffn[0].bias)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.adapter_up_ffn[i].bias.mask = m
                else:
                    for k, p in model.named_parameters():
                        if p.requires_grad and "head" not in k:
                            score = pruner.score(p)
                            mask = pruner.prune(score)
                            p.mask = mask
            
            elif cfg.MODEL.TRANSFER_TYPE == "lora" or cfg.MODEL.TRANSFER_TYPE == "mosl":
                if cfg.MODEL.LORA.MOE:
                    for blk in model.enc.transformer.encoder.layer:
                        if cfg.MODEL.LORA.SHARE != "down":
                            if "q" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_A_q[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_A_q[i].weight.mask = m
                            if "k" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_A_k[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_A_k[i].weight.mask = m
                            if "v" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_A_v[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_A_v[i].weight.mask = m
                            if "o" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_A_o[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_A_o[i].weight.mask = m
                        
                        if cfg.MODEL.LORA.SHARE != "up":
                            if "q" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_B_q[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_B_q[i].weight.mask = m
                            if "k" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_B_k[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_B_k[i].weight.mask = m
                            if "v" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_B_v[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_B_v[i].weight.mask = m
                            if "o" in cfg.MODEL.LORA.MODE:
                                score = pruner.score(blk.attn.lora_B_o[0].weight)
                                masks = pruner.divide(score)
                                for i, m in enumerate(masks):
                                    blk.attn.lora_B_o[i].weight.mask = m
                                
                else:
                    for k, p in model.named_parameters():
                        if p.requires_grad and "head" not in k:
                            score = pruner.score(p)
                            mask = pruner.prune(score)
                            p.mask = mask
        
        elif cfg.MODEL.TYPE == "swin":
            if cfg.MODEL.ADAPTER.MOE:
                for layer in model.enc.layers:
                    for blk in layer.blocks:
                        if cfg.MODEL.ADAPTER.SHARE != "down":
                            score = pruner.score(blk.mlp.adapter_down[0].weight)
                            masks = pruner.divide(score)
                            for i, m in enumerate(masks):
                                blk.mlp.adapter_down[i].weight.mask = m
                            score = pruner.score(blk.mlp.adapter_down[0].bias)
                            masks = pruner.divide(score)
                            for i, m in enumerate(masks):
                                blk.mlp.adapter_down[i].bias.mask = m
                        
                        if cfg.MODEL.ADAPTER.SHARE != "up":
                            score = pruner.score(blk.mlp.adapter_up[0].weight)
                            masks = pruner.divide(score)
                            for i, m in enumerate(masks):
                                blk.mlp.adapter_up[i].weight.mask = m
                            score = pruner.score(blk.mlp.adapter_up[0].bias)
                            masks = pruner.divide(score)
                            for i, m in enumerate(masks):
                                blk.mlp.adapter_up[i].bias.mask = m
            else:
                for k, p in model.named_parameters():
                    if p.requires_grad and "head" not in k:
                        score = pruner.score(p)
                        mask = pruner.prune(score)
                        p.mask = mask
        log_pruned_model_info(model, verbose=cfg.DBG)
    
    # for k, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(k, p.shape)
    # raise ValueError("stop here")

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, args, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0:
        trainer.eval_classifier(test_loader, "test", 0)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
