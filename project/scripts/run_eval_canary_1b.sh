#!/usr/bin/env bash
# Evaluation script for NVIDIA Canary-1B-v2.
#
# Reference: project/scripts/run_eval_qwen3_asr.sh
#            project/scripts/run_eval_parakeet_tdt.sh
# Model:     ./init_model/nvidia/canary-1b-v2
# Config:    registry/model/nemo_asr.yaml  (entry: canary-1b-v2)
# Prompt:    registry/prompt/nemo_asr.yaml (entry: canary)
#
# ============================================================================
# IMPORTANT — Language coverage of canary-1b-v2
# ============================================================================
# Per the official model card
# (init_model/nvidia/canary-1b-v2/README.md), this model only supports the
# following 25 EUROPEAN LANGUAGES:
#
#   bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk
#
# Mandadrin / Cantonese / Japanese / Korean / Arabic / Hindi / ... are NOT
# in the supported set.  Feeding Chinese audio (or even forcing
# language="zh" through transcribe(source_lang/target_lang)) yields
# garbage / errors, producing a WER ≈ 100% (or a hard exception).  This is
# a HARD MODEL LIMIT rather than a bug in the evaluation pipeline.
#
# We therefore restrict this script to its supported (European) languages
# only.  Use Qwen3-ASR / MegaASR for Chinese benchmarks instead.
# ============================================================================
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# Stages:
#   1: English ASR (canary-1b-v2's strongest configuration).
#   2: Other supported European languages (auto language detection).
stage=2
stop_stage=2

# ---------------------------------------------------------------------------
# Stage 1: English ASR (canary-1b-v2's strongest configuration).
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1: English ASR"
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in canary-1b-v2; do
            for prompt in canary; do
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
            done
        done
    done
fi

# ---------------------------------------------------------------------------
# Stage 2: Other supported European languages.
# canary-1b-v2 is multitask: source_lang / target_lang must both be set to
# the audio's true language, otherwise the model treats the request as
# speech translation INTO that language (e.g. forcing target_lang=en on
# German audio yields English translations and a WER ≈ 100%).
# We therefore use one language-specific registry entry per dataset.
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2: Multilingual European ASR"
    # Pairs of "<dataset>:<model-registry-entry>".
    for pair in \
        "fleurs-de_de:canary-1b-v2-de" \
        "fleurs-fr_fr:canary-1b-v2-fr" \
        "fleurs-ru_ru:canary-1b-v2-ru"; do
        dataset="${pair%%:*}"
        model="${pair##*:}"
        for prompt in canary; do
            CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
        done
    done
fi

echo "success on `date`"
exit 0
