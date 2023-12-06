#!/usr/bin/env python3
"""
Pruner construction functions.
"""

from .pruner import Rand
from . import logging
logger = logging.get_logger("MOSA")
# Supported pruner types
_PRUNER_TYPES = {
    "random": Rand,
}


def build_pruner(cfg):
    """
    build pruner here
    """
    assert (
        cfg.PRUNER.TYPE in _PRUNER_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.PRUNER.TYPE)

    # Construct the pruner
    prune_type = cfg.PRUNER.TYPE
    pruner = _PRUNER_TYPES[prune_type](cfg)

    return pruner


def log_pruned_model_info(model, verbose=False):
    """Logs pruned model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(int(p.mask.sum()) if hasattr(p, 'mask') else p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))
