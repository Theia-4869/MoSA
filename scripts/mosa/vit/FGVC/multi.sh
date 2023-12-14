# launch final training for FGVC-multitask. The hyperparameters are the same from our paper.

gpu_id=${1}
data_path=/data/dataset/
model_root=checkpoints
style=${2}
output_dir=${3}

# rm -rf ${output_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-file configs/finetune/multitask.yaml \
    --sparse-train \
    DATA.BATCH_SIZE "128" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    DATA.NO_TEST "False" \
    MODEL.ADAPTER.BOTTLENECK_SIZE "64" \
    MODEL.ADAPTER.STYLE "${style}" \
    MODEL.ADAPTER.EXPERT_NUM "2" \
    MODEL.ADAPTER.MOE "True" \
    MODEL.ADAPTER.MERGE "add" \
    MODEL.ADAPTER.SHARE "none" \
    MODEL.ADAPTER.ADDITIONAL "True" \
    MODEL.ADAPTER.ADD_WEIGHT "0.0" \
    MODEL.ADAPTER.REG_WEIGHT "1.0" \
    MODEL.TRANSFER_TYPE "mosa" \
    MODEL.TYPE "vit" \
    SEED "3407" \
    SOLVER.BASE_LR "0.0005" \
    SOLVER.WEIGHT_DECAY "0.01" \
    SOLVER.WARMUP_EPOCH "10" \
    DATA.DATAPATH "${data_path}" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    OUTPUT_DIR "${output_dir}"
