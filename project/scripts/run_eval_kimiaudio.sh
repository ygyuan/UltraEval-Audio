set -exo
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=1
stop_stage=4

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # for dataset in tedlium-release1 fleurs-en_us; do
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
    # for dataset in fleurs-zh; do
        for model in kimiaudio; do
            for prompt in kimi-audio-asr-zh; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
    for dataset in tedlium-release1; do
    # for dataset in fleurs-en_us; do
        for model in kimiaudio; do
            for prompt in kimi-audio-asr-en; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en; do
        for model in kimiaudio; do
            for prompt in kimi-s2tt-zh2en; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
    for dataset in covost2-en-zh; do
        for model in kimiaudio; do
            for prompt in kimi-s2tt-en2zh; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in kimiaudio; do
            for prompt in kimi-emotion; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
    for dataset in speech-web-questions speech-triviaqa speech-cmmlu; do
        for model in kimiaudio-speech; do
            for prompt in ""; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
fi

echo "success on `date`"
exit 0
