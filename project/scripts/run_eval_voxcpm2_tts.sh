#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1

# ----- Workaround for multi-GPU (4 workers) torch.compile / inductor autotune deadlock -----
# Issue observed: when 4 worker processes simultaneously trigger torch.compile warmup,
# inductor's Triton autotune races on CUDA events ("Both events must be completed before
# calculating elapsed time."), and the warmup hangs past the 600s read timeout.
# Fix: disable inductor cudagraphs and torch.compile entirely so warmup is deterministic.
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCH_COMPILE_DISABLE=1
# Per-rank inductor cache to avoid cross-process cache contention.
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$$
# -------------------------------------------------------------------------------------------

#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=2
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en cv3_zero_shot_hard_en; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
    for dataset in cv3_zero_shot_zh cv3_zero_shot_hard_zh; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in minimax_tts_english; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
    for dataset in minimax_tts_chinese; do
        for model in voxcpm2-vc; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi

echo "success on `date`"
exit 0
