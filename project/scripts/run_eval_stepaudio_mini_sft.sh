#!/usr/bin/env bash
set -ex

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
current_dir=$(cd "${script_dir}/../.." && pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=4
stop_stage=4

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Chinese ASR datasets - use step_audio_2_mini_asr_zh prompt
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
        for model in Step-Audio-2-mini_sft; do
            [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt step_audio_2_mini_asr_zh --use_model_pool --workers 4
                # exit 0
        done
    done
    # English ASR datasets - use step_audio_2_mini_asr_en prompt
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in Step-Audio-2-mini_sft; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt step_audio_2_mini_asr_en --use_model_pool --workers 4
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Speech translation: zh->en
    for dataset in covost2-zh-en; do
        for model in Step-Audio-2-mini_sft; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt step_audio_2_mini_s2tt_en --use_model_pool --workers 4
        done
    done
    # Speech translation: en->zh
    for dataset in covost2-en-zh; do
        for model in Step-Audio-2-mini_sft; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt step_audio_2_mini_s2tt_zh --use_model_pool --workers 4
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in speech-cmmlu speech-web-questions speech-triviaqa; do
        for model in Step-Audio-2-mini_sft; do
            for prompt in ""; do
                # [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in Step-Audio-2-mini_sft; do
            for prompt in ""; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
    for dataset in asc-moan asc-multi; do
        for model in Step-Audio-2-mini_sft; do
            for prompt in ""; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi
echo "success on `date`"
exit 0
