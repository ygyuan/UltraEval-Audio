#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1

# Higgs Audio v3's voice-clone path (vllm-omni serving_speech.py ->
# _build_higgs_audio_v3_params -> encode_reference_audio) lazily loads the
# v2 codec audio_tokenizer at first request time. By default vllm-omni
# pulls it from `k2-fsa/OmniVoice` via huggingface_hub.snapshot_download,
# which fails under HF_HUB_OFFLINE=1. We already have an equivalent copy
# locally at init_model/bosonai/higgs-audio-v2-tokenizer (boson-ai's
# standalone codec repo, identical weight layout); point vllm-omni at it.
export HIGGS_AUDIO_V2_TOKENIZER_DIR="${current_dir}/init_model/bosonai/higgs-audio-v2-tokenizer"
export ULTRAEVAL_AUDIO_ROOT="${current_dir}"
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# Higgs Audio v3 TTS (bosonai/higgs-audio-v3-tts-4b)
# - Served via sgl-omni (SGLang-Omni); OpenAI-compatible /v1/chat/completions
#   endpoint, zero-shot voice cloning via reference-audio injection.
# - Local weights: ./init_model/bosonai/higgs-audio-v3-tts-4b
# Reference:
#   https://huggingface.co/bosonai/higgs-audio-v3-tts-4b
#   https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html

stage=1
stop_stage=3

# Each Higgs v3 instance hosts its own sgl-omni HTTP server on a single GPU,
# so we run exactly ONE instance per visible GPU (workers == #GPUs in
# CUDA_VISIBLE_DEVICES) to avoid OOM / port races.
GPU_LIST="0"
NUM_WORKERS=1  # equals number of GPUs in GPU_LIST

# ============================================================================
# Stage 1: SeedTTS-Eval (English + Chinese)
# ============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-english; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
                # exit 0
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-chinese; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                #[ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

# ============================================================================
# Stage 2: CommonVoice-3 zero-shot (English + Chinese, normal + hard)
# ============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en cv3_zero_shot_hard_en; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-english; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
    for dataset in cv3_zero_shot_zh cv3_zero_shot_hard_zh; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-chinese; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

# ============================================================================
# Stage 3: MiniMax TTS benchmark (English + Chinese)
# ============================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage 3, stop stage ${stage}"
    for dataset in minimax_tts_english; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-english; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
    for dataset in minimax_tts_chinese; do
        for model in higgs-audio-v3-tts; do
            for prompt in higgs-audio-v3-voice-clone-chinese; do
                echo "dataset: ${dataset}, model: ${model}, prompt: ${prompt}"
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers ${NUM_WORKERS}
            done
        done
    done
fi

echo "success on `date`"
exit 0
