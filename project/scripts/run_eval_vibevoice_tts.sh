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
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-chinese; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en; do
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
    for dataset in cv3_zero_shot_zh; do
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-chinese; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stage}"
    for dataset in cv3_zero_shot_hard_en; do
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
    for dataset in cv3_zero_shot_hard_zh; do
        for model in vibevioce_tts; do
            for prompt in vibevoice-voice-clone-chinese; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6 
            done
        done
    done
fi

echo "success on `date`"
exit 0
