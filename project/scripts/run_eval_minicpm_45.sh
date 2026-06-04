#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=3
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # for dataset in tedlium-release1 fleurs-en_us; do
    # for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
    for dataset in fleurs-zh WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech; do
        for model in MiniCPMo4_5-audio; do
            for prompt in mini-cpm-omni-asr-zh; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
                # exit 0
            done
        done
    done
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in MiniCPMo4_5-audio; do
            for prompt in mini-cpm-omni-asr-en; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en; do
        for model in MiniCPMo4_5-audio; do
            for prompt in mini-cpm-omni-s2tt-zh2en; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
    for dataset in covost2-en-zh; do
        for model in MiniCPMo4_5-audio; do
            for prompt in mini-cpm-omni-s2tt-en2zh; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in MiniCPMo4_5-audio; do
            for prompt in mini-cpm-omni-emotion_analysis; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
    for dataset in speech-cmmlu speech-web-questions speech-triviaqa; do
        for model in MiniCPMo4_5-speech; do
            for prompt in ""; do
                # [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 6
            done
        done
    done
fi

echo "success on `date`"
exit 0
