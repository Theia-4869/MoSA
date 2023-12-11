# launch final training for FGVC-nabirds.

gpu_id=${1}
data_path=/your/path/to/FGVC/nabirds
model_root=checkpoints
output_dir=${2}

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-file configs/finetune/nabirds.yaml \
    --sparse-train \
    DATA.BATCH_SIZE "128" \
    DATA.FEATURE "swinb_imagenet22k_224" \
    MODEL.ADAPTER.BOTTLENECK_SIZE "64" \
    MODEL.ADAPTER.EXPERT_NUM "4" \
    MODEL.ADAPTER.MOE "True" \
    MODEL.ADAPTER.MERGE "add" \
    MODEL.ADAPTER.SHARE "down" \
    MODEL.ADAPTER.ADDITIONAL "True" \
    MODEL.ADAPTER.DEEPREG "False" \
    MODEL.ADAPTER.ADD_WEIGHT "0.0" \
    MODEL.ADAPTER.REG_WEIGHT "1.0" \
    MODEL.TRANSFER_TYPE "mosa" \
    MODEL.TYPE "swin" \
    SEED "3407" \
    SOLVER.BASE_LR "0.0005" \
    SOLVER.WEIGHT_DECAY "0.01" \
    SOLVER.WARMUP_EPOCH "10" \
    DATA.DATAPATH "${data_path}" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    OUTPUT_DIR "${output_dir}"
