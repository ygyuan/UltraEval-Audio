#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=3
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1"
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
        for model in vibevoice-asr-zh; do
            for prompt in vibevoice-asr-zh; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4 
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2"
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in vibevoice-asr-en; do
            for prompt in vibevoice-asr-en; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4 
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage 3"
    # for dataset in asr_lianghui asr_shipinhao_fangyan asr_qunliao_eyi asr_shipinhao_long asr_badcase; do
    for dataset in asr_lianghui; do
        for model in vibevoice-asr-zh; do
            for prompt in vibevoice-asr-zh; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4 
            done
        done
    done
fi

echo "success on `date`"
exit 0
