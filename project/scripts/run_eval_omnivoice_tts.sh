#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=1
stop_stage=1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-english; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-chinese; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-english; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
    for dataset in cv3_zero_shot_zh; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-chinese; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_hard_en; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-english; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
    for dataset in cv3_zero_shot_hard_zh; do
        for model in omnivoice; do
            for prompt in omnivoice-voice-clone-chinese; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 8
            done
        done
    done
fi

echo "success on `date`"
exit 0
