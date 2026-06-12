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

stage=2
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en cv3_zero_shot_hard_en; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
    for dataset in cv3_zero_shot_zh cv3_zero_shot_hard_zh; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stage}"
    for dataset in minimax_tts_english; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
    for dataset in minimax_tts_chinese; do
        for model in glmtts; do
                echo "dataset: ${dataset}, model: ${model}"
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --workers 4
        done
    done
fi

echo "success on `date`"
exit 0
