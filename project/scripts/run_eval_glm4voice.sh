#!/bin/bash
set -exo

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
current_dir=$(cd "${script_dir}/../.." && pwd)
cd ${current_dir}

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in KeSpeech fleurs-zh; do
        for model in glm-4-voice; do
            for prompt in glm4voice-asr-zh; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
                exit 0
            done
        done
    done

    for dataset in tedlium-release1 fleurs-en; do
        for model in glm-4-voice; do
            for prompt in glm4voice-asr-en; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en; do
        for model in glm-4-voice; do
            for prompt in glm4voice-s2tt-zh2en; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done

    for dataset in covost2-en-zh; do
        for model in glm-4-voice; do
            for prompt in glm4voice-s2tt-en2zh; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in glm-4-voice; do
            for prompt in glm4voice-emotion; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done

    for dataset in speech-web-questions speech-triviaqa speech-chatbot-alpaca-eval; do
        for model in glm-4-voice; do
            python -m audio_evals.main --dataset ${dataset} --model ${model}
        done
    done
fi

echo "success on `date`"
exit 0
