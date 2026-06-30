#!/usr/bin/env bash
# Evaluation script for NVIDIA Canary-Qwen-2.5B (SALM).
#
# Reference: project/scripts/run_eval_qwen3_asr.sh
#            project/scripts/run_eval_parakeet_tdt.sh
# Model:     ./init_model/nvidia/canary-qwen-2.5b
# Config:    registry/model/canary_qwen.yaml (entry: canary-qwen-2.5b)
# Prompt:    registry/prompt/nemo_asr.yaml   (entry: canary-qwen)
#
# ============================================================================
# IMPORTANT — Language coverage of canary-qwen-2.5b
# ============================================================================
# Per the official model card
# (init_model/nvidia/canary-qwen-2.5b/README.md):
#
#   "English-only language support.  The model was trained using English
#    data only.  It may be able to spuriously transcribe other languages
#    as the underlying encoder was pretrained using German, French, and
#    Spanish speech in addition to English, but it's unlikely to be
#    reliable as a multilingual model."
#
# Mandarin / Cantonese / Japanese / Korean are NOT supported.  Feeding
# Chinese audio yields garbage transcriptions (English transliterations or
# empty strings), producing a WER ≈ 100%.  This is a HARD MODEL LIMIT
# rather than a bug in the evaluation pipeline.
#
# We therefore restrict this script to English benchmarks only.  Use
# Qwen3-ASR / MegaASR for Chinese benchmarks instead.
# ============================================================================
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# Stages:
#   1: English benchmarks (tedlium-release1, fleurs-en_us)
stage=1
stop_stage=1

# ---------------------------------------------------------------------------
# Stage 1: English ASR (canary-qwen-2.5b's only officially supported mode).
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1: English ASR"
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in canary-qwen-2.5b; do
            for prompt in canary-qwen; do
                # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
            done
        done
    done
fi

echo "success on `date`"
exit 0
