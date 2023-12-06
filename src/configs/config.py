#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.GPU_ID = 0
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = None

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = "linear"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.SAVE_CKPT = False

_C.MODEL.MODEL_ROOT = ""  # root folder for pretrained model weights

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Pruner options
# ----------------------------------------------------------------------
_C.PRUNER = CfgNode()

_C.PRUNER.TYPE = "random"
_C.PRUNER.NUM = 4

# ----------------------------------------------------------------------
# Adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.BOTTLENECK_SIZE = 64
_C.MODEL.ADAPTER.STYLE = "AdaptFormer"
_C.MODEL.ADAPTER.SCALAR = 0.1
_C.MODEL.ADAPTER.DROPOUT = 0.1
_C.MODEL.ADAPTER.MOE = False
_C.MODEL.ADAPTER.BIAS = True
_C.MODEL.ADAPTER.EXPERT_NUM = 4
_C.MODEL.ADAPTER.MERGE = "add"
_C.MODEL.ADAPTER.SHARE = None
_C.MODEL.ADAPTER.ADDITIONAL = False
_C.MODEL.ADAPTER.DEEPREG = False
_C.MODEL.ADAPTER.ADD_WEIGHT = 0.0
_C.MODEL.ADAPTER.REG_WEIGHT = 1.0

# ----------------------------------------------------------------------
# LoRA options
# ----------------------------------------------------------------------
_C.MODEL.LORA = CfgNode()
_C.MODEL.LORA.RANK = 16
_C.MODEL.LORA.MODE = "qv"
_C.MODEL.LORA.SCALAR = 1.0
_C.MODEL.LORA.DROPOUT = 0.0
_C.MODEL.LORA.MOE = False
_C.MODEL.LORA.BIAS = False
_C.MODEL.LORA.EXPERT_NUM = 4
_C.MODEL.LORA.MERGE = "add"
_C.MODEL.LORA.SHARE = None
_C.MODEL.LORA.ADDITIONAL = False
_C.MODEL.LORA.DEEPREG = False
_C.MODEL.LORA.ADD_WEIGHT = 0.0
_C.MODEL.LORA.REG_WEIGHT = 1.0

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.MERGE_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
