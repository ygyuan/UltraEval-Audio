#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage 1"
    for dataset in fleurs-zh WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech; do
        for model in glm-asr-zh; do
            for prompt in glm-asr-zh; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4
	       # exit 0	
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage 2"
    for dataset in fleurs-en_us tedlium-release1; do
        for model in glm-asr-en; do
            for prompt in glm-asr-en; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4 
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage 3"
    for dataset in asr_lianghui asr_shipinhao_fangyan asr_qunliao_eyi asr_shipinhao_long asr_badcase; do
        for model in glm-asr-zh; do
            for prompt in glm-asr-zh; do
                CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --post_process asr_strip_tags --use_model_pool --workers 4 
            done
        done
    done
fi

echo "success on `date`"
exit 0
