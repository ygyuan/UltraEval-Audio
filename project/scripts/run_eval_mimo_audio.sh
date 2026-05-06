set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Chinese ASR datasets
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
        for model in mimo-audio; do
            for prompt in mimo-audio-asr-zh; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
    # English ASR datasets
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in mimo-audio; do
            for prompt in mimo-audio-asr-en; do
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Speech Translation: Chinese to English
    for dataset in covost2-zh-en; do
        for model in mimo-audio; do
            for prompt in mimo-audio-s2tt-zh2en; do
                    echo ${prompt}
                    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
    # Speech Translation: English to Chinese
    for dataset in covost2-en-zh; do
        for model in mimo-audio; do
            for prompt in mimo-audio-s2tt-en2zh; do
                    echo ${prompt}
                    #CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # Emotion Recognition
    for dataset in meld-emo; do
        for model in mimo-audio; do
            for prompt in mimo-audio-emotion; do
                    echo ${prompt}
                    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 6
            done
        done
    done
    # Spoken QA datasets
    # for dataset in speech-web-questions speech-triviaqa speech-chatbot-alpaca-eval; do
    for dataset in speech-cmmlu; do
        for model in mimo-audio; do
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python audio_evals/main.py --dataset ${dataset} --model ${model}
        done
    done
fi

echo "success on `date`"
exit 0
