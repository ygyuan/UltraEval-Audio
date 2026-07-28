#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}

stage=4
stop_stage=4

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Chinese ASR datasets
    # for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
    for dataset in KeSpeech fleurs-zh; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-asr-zh; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
    # English ASR datasets
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-asr-en; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Speech Translation: Chinese to English
    for dataset in covost2-zh-en; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-s2tt-zh2en; do
                    echo ${prompt}
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
    # Speech Translation: English to Chinese
    for dataset in covost2-en-zh; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-s2tt-en2zh; do
                    echo ${prompt}
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Emotion Recognition
    for dataset in meld-emo; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-emotion; do
                    echo ${prompt}
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
    # Spoken QA datasets
    # speech-web-questions / speech-triviaqa default_task=s2s-aqa, use the
    # audio-only S2S prompt so the model answers with speech (downstream
    # extract_audio + speech2text post-processing is required).
    for dataset in speech-web-questions speech-triviaqa; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-speech-qa; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
    # speech-cmmlu default_task=speech-choice-aqa-zh (spoken multiple choice)
    for dataset in speech-cmmlu; do
        for model in funaudio_chat; do
            for prompt in funaudio_chat-speech-choice-qa-zh; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in asc-moan asc-multi; do
        for model in funaudio_chat; do
            for prompt in ""; do
                [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="0,1,2,3" python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
fi

echo "success on `date`"
exit 0
