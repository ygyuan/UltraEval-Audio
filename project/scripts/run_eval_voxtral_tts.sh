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

# Voxtral 4B TTS 2603
# - Served via vllm-omni; preset voices only (no zero-shot voice clone).
# - Supported languages: English, French, Spanish, German, Italian,
#   Portuguese, Dutch, Arabic, Hindi (no Chinese).
# Reference: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603

stage=1
stop_stage=3

# Each Voxtral instance hosts its own vllm-omni HTTP server and, by default,
# reserves ~90% of a single GPU's memory. We therefore run exactly ONE
# instance per visible GPU (workers == #GPUs in CUDA_VISIBLE_DEVICES);
# putting multiple vllm servers on the same GPU triggers OOM/port races and
# was the root cause of the previous KeyboardInterrupt teardown.
GPU_LIST="0"
NUM_WORKERS=1  # equals number of GPUs in GPU_LIST

# ============================================================================
# Stage 1: SeedTTS-Eval English subset (only English is supported by Voxtral)
# ============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in voxtral-4b-tts; do
            for prompt in voxtral-en-male voxtral-en-female; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}-${prompt}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

# ============================================================================
# Stage 2: cv3 zero-shot English (English-only TTS quality benchmark)
# ============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en cv3_zero_shot_hard_en; do
        for model in voxtral-4b-tts; do
            for prompt in voxtral-en-male voxtral-en-female; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}-${prompt}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

# ============================================================================
# Stage 3: MiniMax TTS English benchmark
# ============================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage 3, stop stage ${stop_stage}"
    for dataset in minimax_tts_english; do
        for model in voxtral-4b-tts; do
            for prompt in voxtral-en-male voxtral-en-female; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                # [ ! -d res/${model}-${prompt}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

echo "success on `date`"
exit 0
