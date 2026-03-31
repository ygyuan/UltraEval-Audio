set -exo
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # for dataset in tedlium-release1 fleurs-en_us; do
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
        for model in step-audio-r1.1; do
            for prompt in kimi-audio-asr-zh; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model}
                    # exit 0
            done
        done
    done
    for dataset in tedlium-release1; do
    # for dataset in fleurs-en_us; do
        for model in step-audio-r1.1; do
            for prompt in kimi-audio-asr-en; do
                # [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en covost2-en-zh; do
        for model in step-audio-r1.1; do
            # [ ! -d res/${model}/${dataset} ] && \
                python audio_evals/main.py --dataset ${dataset} --model ${model}
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in step-audio-r1.1; do
            for prompt in kimi-emotion; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model}
            done
        done
    done
    for dataset in speech-web-questions speech-triviaqa; do
        for model in step-audio-r1.1; do
            for prompt in ""; do
                # [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="6,7" python audio_evals/main.py --dataset ${dataset} --model ${model} --use_model_pool --workers 4
            done
        done
    done
    for dataset in mmau; do
        for model in step-audio-r1.1; do
            for prompt in step_audio_r1_mmau; do
                # [ ! -d res/${model}/${dataset} ] && \
                    CUDA_VISIBLE_DEVICES="6,7" python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 4
            done
        done
    done
fi

echo "success on `date`"
exit 0
