#!/bin/bash
set -exo

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
current_dir=$(cd "${script_dir}/../.." && pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Chinese ASR datasets - use step_audio_r1_asr_zh prompt
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
        for model in step-audio-r1.1; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 --prompt step_audio_r1_asr_zh 
                # exit 0
        done
    done
    # English ASR datasets - use step_audio_r1_asr_en prompt
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in step-audio-r1.1; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 --prompt step_audio_r1_asr_en 
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Speech translation: zh->en
    for dataset in covost2-zh-en; do
        for model in step-audio-r1.1; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 --prompt step_audio_r1_s2tt_en 
        done
    done
    # Speech translation: en->zh
    for dataset in covost2-en-zh; do
        for model in step-audio-r1.1; do
            # [ ! -d res/${model}/${dataset} ] && \
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 --prompt step_audio_r1_s2tt_zh 
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in step-audio-r1.1; do
            for prompt in ""; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 
            done
        done
    done
    for dataset in speech-cmmlu speech-web-questions speech-triviaqa; do
        for model in step-audio-r1.1; do
            for prompt in ""; do
                # [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model} --use_model_pool --workers 8 
            done
        done
    done
    # for dataset in mmau-test-mini; do
    #     for model in step-audio-r1.1; do
    #         for prompt in step_audio_r1_mmau; do
    #             # [ ! -d res/${model}/${dataset} ] && \
    #                 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m audio_evals.main --dataset ${dataset} --model ${model}--use_model_pool --workers 8 --prompt ${prompt} 
    #         done
    #     done
    # done
fi

echo "success on `date`"
exit 0
