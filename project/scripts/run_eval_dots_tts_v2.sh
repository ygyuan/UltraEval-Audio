#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# dots.tts may use torch.compile / inductor when a registry entry enables
# optimize=True. The default dots-tts-mf config keeps optimize=False because
# BF16 compile warmup still has dtype-boundary issues, but we keep a dedicated
# cache path here so opt-in experiments do not share inductor artifacts across
# concurrent runs.
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dots_tts_$$

# NOTE: run dots.tts in single-process mode (no --use_model_pool / --workers).
# The model pool spawned multiple subprocesses but evaluation models
# (whisper / wavlm) and the dots.tts pipe protocol caused throughput
# regression and uneven GPU usage in practice; single-process is more
# predictable here.

stage=2
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en cv3_zero_shot_hard_en; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
    for dataset in cv3_zero_shot_zh cv3_zero_shot_hard_zh; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in minimax_tts_english; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
    for dataset in minimax_tts_chinese; do
        for model in dots-tts-mf; do
            for prompt in ""; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="3" python3 audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
fi

echo "success on `date`"
exit 0
