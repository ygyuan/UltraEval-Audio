#!/usr/bin/env bash
# Evaluation script for NVIDIA Parakeet-TDT-0.6B-v3.
#
# Reference: project/scripts/run_eval_qwen3_asr.sh
# Model:     ./init_model/nvidia/parakeet-tdt-0.6b-v3
# Config:    registry/model/nemo_asr.yaml  (entries: parakeet-tdt-0.6b-v3,
#                                                    parakeet-tdt-0.6b-v3-en)
# Prompt:    registry/prompt/nemo_asr.yaml (entry: parakeet-tdt)
#
# ============================================================================
# IMPORTANT — Language coverage of parakeet-tdt-0.6b-v3
# ============================================================================
# Per the official model card
# (init_model/nvidia/parakeet-tdt-0.6b-v3/README.md), this model only
# supports the following 25 EUROPEAN LANGUAGES:
#
#   bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk
#
# Mandarin / Cantonese / Japanese / Korean / Arabic / Hindi / ... are NOT
# in the supported set.  The unified SentencePiece tokenizer (8192 tokens)
# was built only over EU-language transcripts, so feeding Chinese audio
# yields empty strings or transliterated Latin gibberish, producing a
# WER ≈ 100% across every Chinese benchmark.  This is a HARD MODEL LIMIT
# rather than a bug in the evaluation pipeline.
#
# We therefore restrict this script to its supported (European) languages
# and provide a separate Chinese-friendly script for Qwen3-ASR / MegaASR /
# canary-qwen-2.5b (which DO support Chinese to varying degrees).
# ============================================================================
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# Stages:
#   1: English benchmarks (tedlium-release1, fleurs-en_us)
#   2: Other supported European languages (fleurs-de_de / fr_fr / ru_ru)
stage=1
stop_stage=2

# ---------------------------------------------------------------------------
# Stage 1: English ASR (parakeet's strongest configuration).
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1: English ASR"
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in parakeet-tdt-0.6b-v3-en; do
            for prompt in parakeet-tdt; do
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
            done
        done
    done
fi

# ---------------------------------------------------------------------------
# Stage 2: Other supported European languages (auto language detection).
# Only fleurs-{de,fr,ru} are configured under registry/dataset/fleurs.yaml.
# Add more (e.g. fleurs-es_419, fleurs-it_it) as needed.
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2: Multilingual European ASR"
    for dataset in fleurs-de_de fleurs-fr_fr fleurs-ru_ru; do
        for model in parakeet-tdt-0.6b-v3; do
            for prompt in parakeet-tdt; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
            done
        done
    done
fi

echo "success on `date`"
exit 0
